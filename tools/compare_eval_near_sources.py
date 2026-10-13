from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from evaluation.eval_runner import run_eval_metrics
from methods.utils.projection import project_points_with_steps_numpy
from planning.checkpoint_loader import _load_mlp_field, load_constraint_model_from_checkpoint
from planning.kinematics import (
    is_arm_dataset,
    is_workspace_pose_dataset,
    workspace_embed_for_eval,
    wrap_np_pi,
    wrap_workspace_pose_rpy_np,
)


def _find_checkpoints(bench: str, method: str) -> list[str]:
    pats = [
        os.path.join(bench, method, "*", f"*_{method}_seed*_model.pt"),
        os.path.join(bench, method, f"*_{method}_model.pt"),
    ]
    out: list[str] = []
    for pat in pats:
        out.extend(glob.glob(pat))
    return sorted(set(out))


def _dataset_from_ckpt_path(path: str, method: str) -> str:
    parent = os.path.basename(os.path.dirname(path))
    if parent and parent != method:
        return parent
    name = os.path.basename(path)
    for marker in (f"_{method}_seed", f"_{method}_model.pt"):
        if marker in name:
            return name.split(marker, 1)[0]
    return name


def _seed_from_ckpt(ckpt: dict[str, Any]) -> str:
    cfg = ckpt.get("cfg", {}) or {}
    seed = cfg.get("seed") if isinstance(cfg, dict) else getattr(cfg, "seed", "")
    return str(seed)


def _load_saved_eval(path: str, dataset: str, method: str, seed: str) -> dict[str, Any]:
    candidates = [
        os.path.join(os.path.dirname(path), f"{dataset}_{method}_seed{seed}_eval.json"),
        os.path.join(os.path.dirname(os.path.dirname(path)), f"{dataset}_{method}_eval.json"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                return {}
    return {}


def _cfg_namespace(ckpt: dict[str, Any], device: str, near_source: str) -> Any:
    cfg = SimpleNamespace(**dict(ckpt.get("cfg", {}) or {}))
    cfg.device = device
    cfg.eval_chamfer_near_source = str(near_source)
    return cfg


def _postprocess_fn(dataset: str):
    if is_arm_dataset(dataset):
        return wrap_np_pi
    if is_workspace_pose_dataset(dataset):
        return wrap_workspace_pose_rpy_np
    return None


def _embed_fn(dataset: str):
    if not (is_arm_dataset(dataset) or is_workspace_pose_dataset(dataset)):
        return None
    use_pybullet_n6 = str(dataset).startswith("6d_spatial_arm_up_n6") and "_py" not in str(dataset)
    return lambda q: workspace_embed_for_eval(dataset, q, ur5_use_pybullet_n6=use_pybullet_n6)


def _generic_project_fn(cfg: Any):
    proj_cfg = getattr(cfg, "projector", {}) or {}

    def project_fn(model: nn.Module, x0: np.ndarray, eps_stop: float) -> tuple[np.ndarray, np.ndarray]:
        return project_points_with_steps_numpy(
            model,
            x0.astype(np.float32),
            device=str(cfg.device),
            proj_steps=int(proj_cfg.get("steps", 80)),
            proj_alpha=float(proj_cfg.get("alpha", 0.3)),
            proj_min_steps=int(proj_cfg.get("min_steps", 0)),
            f_abs_stop=float(eps_stop),
        )

    return project_fn


def _evaluate_ckpt(path: str, method: str, source: str, device: str) -> dict[str, float]:
    ckpt_raw = torch.load(path, map_location="cpu", weights_only=False)
    if "constraint_dim" not in ckpt_raw and str(method) in {"dataaug", "margin"}:
        ckpt = dict(ckpt_raw)
        ckpt["constraint_dim"] = 1
        model_type = str(ckpt.get("model_type") or "mlp").lower()
        model = _load_mlp_field(ckpt, device=device, model_type=model_type)
        model.eval()
    else:
        model, ckpt = load_constraint_model_from_checkpoint(path, device=device)
    dataset = _dataset_from_ckpt_path(path, method)
    cfg = _cfg_namespace(ckpt, device=device, near_source=source)
    x_train = np.asarray(ckpt["x_train"], dtype=np.float32)
    metrics, _eval_cfg, _artifacts = run_eval_metrics(
        cfg=cfg,
        method_key=method,
        dataset_name=dataset,
        model=model,
        x_train=x_train,
        project_fn=_generic_project_fn(cfg),
        embed_fn=_embed_fn(dataset),
        postprocess_fn=_postprocess_fn(dataset),
    )
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description="Compare evaluation near-query source: train demos vs GT manifold.")
    p.add_argument("--bench", default="outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed")
    p.add_argument("--method", default="oncl")
    p.add_argument("--outdir", default="outputs/analysis/eval_near_source_comparison")
    p.add_argument("--device", default="auto")
    args = p.parse_args()

    bench = os.path.abspath(args.bench)
    method = str(args.method)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    device = "cuda" if str(args.device) == "auto" and torch.cuda.is_available() else str(args.device)
    if device == "auto":
        device = "cpu"

    rows: list[dict[str, Any]] = []
    for ckpt_path in _find_checkpoints(bench, method):
        raw_ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        dataset = _dataset_from_ckpt_path(ckpt_path, method)
        seed = _seed_from_ckpt(raw_ckpt)
        saved_eval = _load_saved_eval(ckpt_path, dataset, method, seed)
        print(f"[eval] method={method} dataset={dataset} seed={seed}")
        try:
            train_metrics = _evaluate_ckpt(ckpt_path, method, "train", device)
            gt_metrics = _evaluate_ckpt(ckpt_path, method, "gt", device)
            train_dist = float(train_metrics.get("proj_manifold_dist", np.nan))
            gt_dist = float(gt_metrics.get("proj_manifold_dist", np.nan))
            row = {
                "method": method,
                "dataset": dataset,
                "seed": seed,
                "saved_proj_manifold_dist": saved_eval.get("proj_manifold_dist", ""),
                "train_near_proj_manifold_dist": train_dist,
                "gt_near_proj_manifold_dist": gt_dist,
                "delta_gt_minus_train": gt_dist - train_dist,
                "ratio_gt_over_train": gt_dist / train_dist if np.isfinite(train_dist) and train_dist != 0 else np.nan,
                "train_near_proj_true_dist": float(train_metrics.get("proj_true_dist", np.nan)),
                "gt_near_proj_true_dist": float(gt_metrics.get("proj_true_dist", np.nan)),
                "train_near_proj_steps": float(train_metrics.get("proj_steps", np.nan)),
                "gt_near_proj_steps": float(gt_metrics.get("proj_steps", np.nan)),
                "error": "",
            }
        except Exception as exc:
            row = {
                "method": method,
                "dataset": dataset,
                "seed": seed,
                "saved_proj_manifold_dist": saved_eval.get("proj_manifold_dist", ""),
                "train_near_proj_manifold_dist": "",
                "gt_near_proj_manifold_dist": "",
                "delta_gt_minus_train": "",
                "ratio_gt_over_train": "",
                "train_near_proj_true_dist": "",
                "gt_near_proj_true_dist": "",
                "train_near_proj_steps": "",
                "gt_near_proj_steps": "",
                "error": str(exc),
            }
            print(f"[warn] failed {dataset}: {exc}")
        rows.append(row)

    out_csv = os.path.join(outdir, f"{method}_eval_near_source_comparison.csv")
    fieldnames = [
        "method",
        "dataset",
        "seed",
        "saved_proj_manifold_dist",
        "train_near_proj_manifold_dist",
        "gt_near_proj_manifold_dist",
        "delta_gt_minus_train",
        "ratio_gt_over_train",
        "train_near_proj_true_dist",
        "gt_near_proj_true_dist",
        "train_near_proj_steps",
        "gt_near_proj_steps",
        "error",
    ]
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()
