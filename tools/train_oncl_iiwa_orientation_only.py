#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.constraint_datasets import set_seed
from datasets.real_iiwa.registry import plane_constraint
from methods import oncl
from methods.models.mlp import MLP
from methods.utils.projection import project_points_with_steps_numpy


def _rpy_to_rotmat_batch(rpy: np.ndarray) -> np.ndarray:
    r = np.asarray(rpy, dtype=np.float64)
    cr, sr = np.cos(r[:, 0]), np.sin(r[:, 0])
    cp, sp = np.cos(r[:, 1]), np.sin(r[:, 1])
    cy, sy = np.cos(r[:, 2]), np.sin(r[:, 2])
    rot = np.empty((r.shape[0], 3, 3), dtype=np.float64)
    rot[:, 0, 0] = cy * cp
    rot[:, 0, 1] = cy * sp * sr - sy * cr
    rot[:, 0, 2] = cy * sp * cr + sy * sr
    rot[:, 1, 0] = sy * cp
    rot[:, 1, 1] = sy * sp * sr + cy * cr
    rot[:, 1, 2] = sy * sp * cr - cy * sr
    rot[:, 2, 0] = -sp
    rot[:, 2, 1] = cp * sr
    rot[:, 2, 2] = cp * cr
    return rot


def _orientation_error_deg(rpy: np.ndarray) -> np.ndarray:
    normal, _ = plane_constraint()
    target = -normal
    rot = _rpy_to_rotmat_batch(np.asarray(rpy, dtype=np.float64))
    local_z = rot[:, :, 2]
    dot = np.clip(local_z @ target, -1.0, 1.0)
    return np.degrees(np.arccos(dot)).astype(np.float64)


def _stats(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "p90": float(np.quantile(v, 0.90)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(np.max(v)),
        "min": float(np.min(v)),
    }


def _sample_eval_orientations(x_train: np.ndarray, *, n: int, seed: int, pad_ratio: float) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    lo = x_train.min(axis=0)
    hi = x_train.max(axis=0)
    span = np.maximum(hi - lo, 1e-4)
    lo = lo - float(pad_ratio) * span
    hi = hi + float(pad_ratio) * span
    return rng.uniform(lo, hi, size=(int(n), x_train.shape[1])).astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ONCL on iiwa orientation-only RPY data.")
    parser.add_argument(
        "--train-set",
        type=Path,
        default=Path("datasets/real_iiwa/tasks/plane_surface_fixed_orientation/processed/train_sets/tcp_pos_8mm_manual_crop_v001.npz"),
    )
    parser.add_argument("--x-key", default="tcp_rpy_unwrapped")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/real_iiwa_oncl_fixed_orientation_only_c2"))
    parser.add_argument("--seed", type=int, default=2116)
    parser.add_argument("--constraint-dim", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-points", type=int, default=4096)
    parser.add_argument("--eval-pad-ratio", type=float, default=0.75)
    parser.add_argument("--proj-steps", type=int, default=160)
    parser.add_argument("--proj-min-steps", type=int, default=20)
    parser.add_argument("--proj-alpha", type=float, default=0.3)
    parser.add_argument("--zero-eps-quantile", type=float, default=90.0)
    parser.add_argument("--train-log-every", type=int, default=750)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = oncl._choose_device("auto")

    with np.load(args.train_set, allow_pickle=False) as data:
        x_train = np.asarray(data[str(args.x_key)], dtype=np.float32)
    if x_train.ndim != 2 or x_train.shape[1] != 3:
        raise ValueError(f"expected orientation-only RPY data with shape (N,3), got {x_train.shape}")

    cfg = oncl.DemoCfg(
        seed=int(args.seed),
        device=device,
        n_train=int(x_train.shape[0]),
        hidden=int(args.hidden),
        depth=int(args.depth),
        lr=float(args.lr),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        constraint_dim=int(args.constraint_dim),
        train_log_every=int(args.train_log_every),
        metric_eval_every=max(1, int(args.epochs) + 1),
        projector={
            "alpha": float(args.proj_alpha),
            "steps": int(args.proj_steps),
            "min_steps": int(args.proj_min_steps),
        },
    )
    cfg.train_method_name = "oncl_orientation_only"

    model, train_hist = oncl.train_oncl_only(cfg, x_train, constraint_dim=int(args.constraint_dim))
    model.eval()
    x_eval = _sample_eval_orientations(
        x_train,
        n=int(args.eval_points),
        seed=int(args.seed) + 17,
        pad_ratio=float(args.eval_pad_ratio),
    )
    with torch.no_grad():
        f_train = model(torch.from_numpy(x_train).to(device)).detach().cpu().numpy()
        f_eval_before = model(torch.from_numpy(x_eval).to(device)).detach().cpu().numpy()
    zero_eps_quantile = float(args.zero_eps_quantile)
    f_abs_stop = float(np.quantile(np.linalg.norm(f_train, axis=1), zero_eps_quantile / 100.0))
    proj, steps = project_points_with_steps_numpy(
        model,
        x_eval,
        device=device,
        proj_steps=int(args.proj_steps),
        proj_alpha=float(args.proj_alpha),
        proj_min_steps=int(args.proj_min_steps),
        f_abs_stop=f_abs_stop,
    )
    with torch.no_grad():
        f_eval_after = model(torch.from_numpy(proj.astype(np.float32)).to(device)).detach().cpu().numpy()

    train_err = _orientation_error_deg(x_train)
    before_err = _orientation_error_deg(x_eval)
    after_err = _orientation_error_deg(proj)
    report = {
        "train_set": str(args.train_set),
        "x_key": str(args.x_key),
        "constraint_dim": int(args.constraint_dim),
        "device": str(device),
        "train_shape": list(x_train.shape),
        "train_orientation_error_deg": _stats(train_err),
        "eval_seed_orientation_error_deg": _stats(before_err),
        "projected_orientation_error_deg": _stats(after_err),
        "project_steps": _stats(steps.astype(np.float64)),
        "f_abs_stop": f_abs_stop,
        "zero_eps_quantile": zero_eps_quantile,
        "train_model_norm": _stats(np.linalg.norm(f_train, axis=1)),
        "eval_seed_model_norm": _stats(np.linalg.norm(f_eval_before, axis=1)),
        "projected_model_norm": _stats(np.linalg.norm(f_eval_after, axis=1)),
        "cfg": asdict(cfg),
    }

    args.outdir.mkdir(parents=True, exist_ok=True)
    report_path = args.outdir / "orientation_only_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_dim": 3,
            "constraint_dim": int(args.constraint_dim),
            "hidden": int(args.hidden),
            "depth": int(args.depth),
            "x_train": x_train,
            "train_hist": train_hist,
            "cfg": report["cfg"],
        },
        args.outdir / "orientation_only_model.pt",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    axes[0].hist(before_err, bins=80, alpha=0.75, label="eval seed before", color="#64748B")
    axes[0].hist(after_err, bins=80, alpha=0.75, label="after ONCL projection", color="#B45309")
    axes[0].hist(train_err, bins=60, alpha=0.45, label="train demo", color="#0F766E")
    axes[0].set_title("Orientation error to perpendicular")
    axes[0].set_xlabel("angle error [deg]")
    axes[0].set_ylabel("count")
    axes[0].legend(fontsize=8)

    axes[1].scatter(before_err, after_err, s=3, alpha=0.25)
    lim = float(max(np.max(before_err), np.max(after_err), 1.0))
    axes[1].plot([0, lim], [0, lim], "--", color="black", lw=0.8)
    axes[1].set_xlim(0, lim)
    axes[1].set_ylim(0, lim)
    axes[1].set_title("Per-point projection change")
    axes[1].set_xlabel("before [deg]")
    axes[1].set_ylabel("after [deg]")
    fig.suptitle(f"ONCL orientation-only, codim={int(args.constraint_dim)}")
    fig.tight_layout()
    fig_path = args.outdir / "orientation_only_error_distribution.png"
    fig.savefig(fig_path, dpi=240)
    plt.close(fig)

    print(f"[report] {report_path}")
    print(f"[figure] {fig_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
