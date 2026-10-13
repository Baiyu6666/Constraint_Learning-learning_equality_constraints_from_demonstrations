from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.config_loader import apply_overrides, load_layered_config
from experiments.dataset_resolve import resolve_dataset
from evaluation.evaluator import (
    _clip_points_to_eval_bounds,
    _workspace_pose_analytic_dist_embed,
    compute_eps_stop,
    resolve_eval_cfg,
    sample_eval_seed_points,
)
from methods import ecomann as ecomann_base
from methods.utils.feature_normalizer import FeatureNormalizedModel, FeatureNormalizer
from planning.kinematics import workspace_embed_for_eval, wrap_workspace_pose_rpy_np


def _split_csv(text: str, cast: Any = str) -> list[Any]:
    return [cast(x.strip()) for x in str(text).split(",") if x.strip()]


def _cfg_from_mapping(mapping: dict[str, Any]) -> ecomann_base.Config:
    field_names = {f.name for f in fields(ecomann_base.Config)}
    kwargs = {k: mapping[k] for k in field_names if k in mapping}
    missing = sorted(field_names - set(kwargs))
    if missing:
        raise ValueError(f"missing ECoMaNN config keys: {missing}")
    cfg = ecomann_base.Config(**kwargs)
    for k, v in mapping.items():
        if not hasattr(cfg, k):
            setattr(cfg, k, v)
    return cfg


def _project_ecomann(
    *,
    model: Any,
    normalizer: FeatureNormalizer,
    x0: np.ndarray,
    eps_stop: float,
    tol: float,
    max_iter: int,
    step_size: float,
    diverge_ratio: float,
    max_dq_norm: float,
) -> tuple[np.ndarray, np.ndarray]:
    x_in_raw = np.asarray(x0, dtype=np.float32)
    x_in = normalizer.transform(x_in_raw) if normalizer.enabled else x_in_raw
    x_out = np.asarray(x_in, dtype=np.float32).copy()
    steps = np.zeros((len(x_out),), dtype=np.float32)
    effective_tol = max(float(tol), float(eps_stop))

    for i in range(len(x_out)):
        q = x_out[i].astype(np.float64, copy=True)
        y = np.asarray(model.y(q), dtype=np.float64).reshape(-1)
        y_norm = float(np.linalg.norm(y))
        y0 = float(diverge_ratio * max(y_norm, 1e-12))
        it = 0
        while (y_norm > effective_tol) and (it < max_iter) and (y_norm < y0):
            try:
                J = np.asarray(model.J(q), dtype=np.float64)
                dq = np.linalg.lstsq(J, y, rcond=None)[0]
                dq_norm = float(np.linalg.norm(dq))
                if np.isfinite(max_dq_norm) and max_dq_norm > 0.0 and dq_norm > max_dq_norm:
                    dq = dq * (max_dq_norm / max(dq_norm, 1e-12))
            except Exception:
                break
            q = q - float(step_size) * dq
            y = np.asarray(model.y(q), dtype=np.float64).reshape(-1)
            y_norm = float(np.linalg.norm(y))
            it += 1
        x_out[i] = q.astype(np.float32)
        steps[i] = float(it)

    if normalizer.enabled:
        x_out = normalizer.inverse_transform(x_out)
    x_out = wrap_workspace_pose_rpy_np(x_out.astype(np.float32))
    return x_out.astype(np.float32), steps.astype(np.float32)


def run_one(
    *,
    seed: int,
    n_local: float,
    traversal: int,
    lr: float,
    epochs: int,
    eval_points: int,
    eval_grid: int,
    n_train: int | None,
    aug_epsilon: float | None,
    extra_overrides: list[str],
    outdir: Path,
) -> dict[str, Any]:
    overrides = [
        f"seed={seed}",
        f"n_local_neighborhood_mult={n_local}",
        f"n_normal_space_traversal={traversal}",
        f"lr={lr}",
        f"epochs={epochs}",
        f"eval_proj_n_points={eval_points}",
        f"eval_gt_n_grid={eval_grid}",
    ]
    if n_train is not None:
        overrides.append(f"n_train={int(n_train)}")
    if aug_epsilon is not None:
        overrides.append(f"aug_epsilon_mult_override={aug_epsilon}")
    overrides.extend(extra_overrides)

    cfg_map, loaded_paths = load_layered_config(str(PROJECT_ROOT / "configs"), "ecomann", "6d_surf_normal")
    cfg_map = apply_overrides(cfg_map, overrides)
    cfg = _cfg_from_mapping(cfg_map)
    cfg.device = ecomann_base._choose_device(str(cfg.device))
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    ds = resolve_dataset("6d_surf_normal", cfg, optimize_ur5_train_only=True, ur5_backend="analytic")
    x_train = np.asarray(ds["x_train"], dtype=np.float32)
    true_codim = int(ds.get("true_codim", 3))
    normalizer = FeatureNormalizer.fit(x_train, dataset_name="6d_surf_normal", enable=True)
    x_model = normalizer.transform(x_train) if normalizer.enabled else x_train

    train_t0 = time.perf_counter()
    model, _hist, learned_codim = ecomann_base.train_ecomann(cfg, x_model, force_codim=true_codim)
    train_seconds = float(time.perf_counter() - train_t0)

    eval_model = FeatureNormalizedModel(model, normalizer).to(str(cfg.device)) if normalizer.enabled else model
    eval_cfg = resolve_eval_cfg(cfg, method_key="ecomann", dataset_name="6d_surf_normal")
    eval_cfg.eval_proj_n_points = int(eval_points)
    eval_cfg.eval_gt_n_grid = int(eval_grid)

    eps_stop = float(compute_eps_stop(eval_model, x_train, eval_cfg))
    x_eval = sample_eval_seed_points(x_train, eval_cfg, dataset_name="6d_surf_normal")
    x_eval = wrap_workspace_pose_rpy_np(x_eval.astype(np.float32))

    p_cfg = getattr(cfg, "projector", {}) or {}
    eval_t0 = time.perf_counter()
    proj, steps = _project_ecomann(
        model=model,
        normalizer=normalizer,
        x0=x_eval,
        eps_stop=eps_stop,
        tol=float(p_cfg.get("tol", 1e-5)),
        max_iter=int(p_cfg.get("max_iter", 120)),
        step_size=float(p_cfg.get("step_size", 1.0)),
        diverge_ratio=float(p_cfg.get("diverge_ratio", 2.0)),
        max_dq_norm=float(p_cfg.get("max_dq_norm", 1.0)),
    )
    proj = _clip_points_to_eval_bounds(proj, x_train, eval_cfg, dataset_name="6d_surf_normal")
    proj = wrap_workspace_pose_rpy_np(proj.astype(np.float32))
    embed = workspace_embed_for_eval("6d_surf_normal", proj).astype(np.float32)
    dist = _workspace_pose_analytic_dist_embed(embed[:, :6], eval_cfg).astype(np.float32)
    finite = np.isfinite(dist)
    eval_seconds = float(time.perf_counter() - eval_t0)

    row = {
        "seed": int(seed),
        "n_local_neighborhood_mult": float(n_local),
        "n_normal_space_traversal": int(traversal),
        "lr": float(lr),
        "epochs": int(epochs),
        "aug_epsilon_mult_override": "" if aug_epsilon is None else float(aug_epsilon),
        "feature_normalizer_enabled": bool(normalizer.enabled),
        "angle_dims": list(normalizer.angle_dims),
        "learned_codim": int(learned_codim),
        "eval_points": int(len(x_eval)),
        "n_train": int(len(x_train)),
        "eval_finite_ratio": float(np.mean(finite)) if len(finite) else float("nan"),
        "proj_manifold_dist_fast": float(np.mean(dist[finite])) if np.any(finite) else float("nan"),
        "proj_manifold_dist_fast_median": float(np.median(dist[finite])) if np.any(finite) else float("nan"),
        "proj_manifold_dist_fast_p95": float(np.percentile(dist[finite], 95)) if np.any(finite) else float("nan"),
        "proj_steps_mean": float(np.mean(steps[finite])) if np.any(finite) else float("nan"),
        "eval_eps_used": float(eps_stop),
        "train_seconds": float(train_seconds),
        "eval_seconds": float(eval_seconds),
        "seconds": float(train_seconds + eval_seconds),
        "loaded_config_paths": loaded_paths,
    }
    detail_path = outdir / f"seed{seed}_nlocal{str(n_local).replace('.', 'p')}.json"
    detail_path.write_text(json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8")
    return row


def main() -> None:
    p = argparse.ArgumentParser(description="Fast ECoMaNN SurfNormal scatter hyperparameter screen.")
    p.add_argument("--seeds", default="0,1")
    p.add_argument("--n-local", default="0.5,0.75,1.0,1.25,1.5")
    p.add_argument("--traversal", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--eval-points", type=int, default=256)
    p.add_argument("--eval-grid", type=int, default=1024)
    p.add_argument("--n-train", type=int, default=None)
    p.add_argument("--aug-epsilon", type=float, default=None)
    p.add_argument("--override", action="append", default=[])
    p.add_argument("--outdir", default="outputs/analysis/ecomann_surfnormal_scatter_fast_tune_2seed")
    args = p.parse_args()

    outdir = PROJECT_ROOT / str(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path = outdir / "results.csv"
    rows: list[dict[str, Any]] = []
    if csv_path.exists():
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    done = {
        (int(r["seed"]), float(r["n_local_neighborhood_mult"]), str(r.get("aug_epsilon_mult_override", "")))
        for r in rows
        if str(r.get("seed", "")).strip()
    }
    fieldnames: list[str] | None = None
    for seed in _split_csv(args.seeds, int):
        for n_local in _split_csv(args.n_local, float):
            key = (int(seed), float(n_local), "" if args.aug_epsilon is None else str(float(args.aug_epsilon)))
            if key in done:
                print(f"[skip] seed={seed} n_local={n_local}", flush=True)
                continue
            print(f"[run] seed={seed} n_local={n_local}", flush=True)
            row = run_one(
                seed=int(seed),
                n_local=float(n_local),
                traversal=int(args.traversal),
                lr=float(args.lr),
                epochs=int(args.epochs),
                eval_points=int(args.eval_points),
                eval_grid=int(args.eval_grid),
                n_train=(None if args.n_train is None else int(args.n_train)),
                aug_epsilon=args.aug_epsilon,
                extra_overrides=list(args.override or []),
                outdir=outdir,
            )
            rows.append(row)
            fieldnames = list(row.keys()) if fieldnames is None else fieldnames
            with csv_path.open("w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
            print(
                "[done] "
                f"seed={seed} n_local={n_local} "
                f"dist={row['proj_manifold_dist_fast']:.6f} "
                f"seconds={row['seconds']:.1f}",
                flush=True,
            )

    if rows:
        arr: dict[float, list[float]] = {}
        for r in rows:
            nl = float(r["n_local_neighborhood_mult"])
            val = float(r["proj_manifold_dist_fast"])
            if math.isfinite(val):
                arr.setdefault(nl, []).append(val)
        summary = [
            {"n_local_neighborhood_mult": nl, "mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
            for nl, vals in sorted(arr.items())
        ]
        (outdir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
