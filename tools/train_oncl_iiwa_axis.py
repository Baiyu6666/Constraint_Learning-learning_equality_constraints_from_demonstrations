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
from datasets.ros.transforms import quat_to_matrix_xyzw
from methods import oncl
from methods.models.mlp import MLP
from methods.utils.projection import project_points_with_steps_numpy


def _tool_z_from_quat(quat_xyzw: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64)
    out = np.empty((quat.shape[0], 3), dtype=np.float64)
    for i, q in enumerate(quat):
        out[i] = quat_to_matrix_xyzw(q)[:, 2]
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


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


def _axis_angle_error_deg(axis: np.ndarray) -> np.ndarray:
    normal, _ = plane_constraint()
    target = -normal
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    return np.degrees(np.arccos(np.clip(a @ target, -1.0, 1.0)))


def _position_abs_mm(pos: np.ndarray) -> np.ndarray:
    normal, d = plane_constraint()
    return np.abs(np.asarray(pos, dtype=np.float64) @ normal + float(d)) * 1000.0


def _hemisphere_sign(axis: np.ndarray) -> float:
    median_z = float(np.median(np.asarray(axis, dtype=np.float64)[:, 2]))
    return -1.0 if median_z < 0.0 else 1.0


def _reconstruct_axis_from_xy(axis_xy: np.ndarray, hemisphere_sign: float) -> np.ndarray:
    xy = np.asarray(axis_xy, dtype=np.float64)
    r2 = np.sum(xy[:, :2] ** 2, axis=1)
    z_abs = np.sqrt(np.maximum(1.0 - r2, 0.0))
    axis = np.column_stack([xy[:, 0], xy[:, 1], float(hemisphere_sign) * z_abs])
    return axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), 1e-12)


def _axis_from_features(x: np.ndarray, mode: str, hemisphere_sign: float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if mode == "axis":
        return arr[:, :3]
    if mode == "pos_axis":
        return arr[:, 3:6]
    if mode == "pos_axis_xy":
        return _reconstruct_axis_from_xy(arr[:, 3:5], hemisphere_sign=hemisphere_sign)
    raise ValueError(f"unsupported mode={mode!r}")


def _sample_eval(x_train: np.ndarray, *, n: int, seed: int, pad_ratio: float, mode: str) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    x = np.asarray(x_train, dtype=np.float32)
    lo = x.min(axis=0)
    hi = x.max(axis=0)
    span = np.maximum(hi - lo, 1e-4)
    out = rng.uniform(lo - float(pad_ratio) * span, hi + float(pad_ratio) * span, size=(int(n), x.shape[1]))
    if mode == "axis":
        out = out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)
    elif mode == "pos_axis":
        out[:, 3:6] = out[:, 3:6] / np.maximum(np.linalg.norm(out[:, 3:6], axis=1, keepdims=True), 1e-12)
    return out.astype(np.float32)


def _split_position(x: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    if mode in {"pos_axis", "pos_axis_xy"}:
        return arr[:, :3]
    raise ValueError(f"mode={mode!r} does not contain position features")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ONCL with iiwa tool local-Z axis representation.")
    parser.add_argument(
        "--train-set",
        type=Path,
        default=Path("datasets/real_iiwa/tasks/plane_surface_fixed_orientation/processed/train_sets/tcp_pos_8mm_manual_crop_v001.npz"),
    )
    parser.add_argument("--mode", choices=("axis", "pos_axis", "pos_axis_xy"), default="axis")
    parser.add_argument("--outdir", type=Path, default=Path("outputs/real_iiwa_oncl_fixed_axis"))
    parser.add_argument("--constraint-dim", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=2116)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--lam-oncl", type=float, default=0.25, help="Weight on ONCL gradient norm/orthogonality regularization.")
    parser.add_argument("--lam-oncl-ortho", type=float, default=1.0, help="Relative weight for gradient orthogonality inside ONCL regularization.")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-points", type=int, default=4096)
    parser.add_argument("--eval-pad-ratio", type=float, default=0.75)
    parser.add_argument("--proj-steps", type=int, default=180)
    parser.add_argument("--proj-min-steps", type=int, default=20)
    parser.add_argument("--proj-alpha", type=float, default=0.3)
    parser.add_argument("--zero-eps-quantile", type=float, default=90.0)
    parser.add_argument("--metric-eval-every", type=int, default=1, help="How often to record train h/gradient diagnostics.")
    parser.add_argument("--train-log-every", type=int, default=750)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(int(args.seed))
    device = oncl._choose_device("auto")

    with np.load(args.train_set, allow_pickle=False) as data:
        pos = np.asarray(data["tcp_pos"], dtype=np.float32)
        axis = _tool_z_from_quat(np.asarray(data["tcp_quat"], dtype=np.float64)).astype(np.float32)
    hemisphere_sign = _hemisphere_sign(axis)
    if str(args.mode) == "axis":
        x_train = axis
    elif str(args.mode) == "pos_axis":
        x_train = np.concatenate([pos, axis], axis=1).astype(np.float32)
    elif str(args.mode) == "pos_axis_xy":
        x_train = np.concatenate([pos, axis[:, :2]], axis=1).astype(np.float32)
    else:
        raise ValueError(f"unsupported mode={args.mode!r}")

    cfg = oncl.DemoCfg(
        seed=int(args.seed),
        device=device,
        n_train=int(x_train.shape[0]),
        hidden=int(args.hidden),
        depth=int(args.depth),
        lr=float(args.lr),
        lam_oncl=float(args.lam_oncl),
        lam_oncl_ortho=float(args.lam_oncl_ortho),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        constraint_dim=int(args.constraint_dim),
        train_log_every=int(args.train_log_every),
        metric_eval_every=max(1, int(args.metric_eval_every)),
        projector={
            "alpha": float(args.proj_alpha),
            "steps": int(args.proj_steps),
            "min_steps": int(args.proj_min_steps),
        },
    )
    cfg.train_method_name = f"oncl_{args.mode}"
    model, train_hist = oncl.train_oncl_only(cfg, x_train, constraint_dim=int(args.constraint_dim))
    model.eval()

    x_eval = _sample_eval(
        x_train,
        n=int(args.eval_points),
        seed=int(args.seed) + 29,
        pad_ratio=float(args.eval_pad_ratio),
        mode=str(args.mode),
    )
    with torch.no_grad():
        f_train = model(torch.from_numpy(x_train).to(device)).detach().cpu().numpy()
        f_before = model(torch.from_numpy(x_eval).to(device)).detach().cpu().numpy()
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
        f_after = model(torch.from_numpy(proj.astype(np.float32)).to(device)).detach().cpu().numpy()

    train_axis = _axis_from_features(x_train, str(args.mode), hemisphere_sign=hemisphere_sign)
    before_axis = _axis_from_features(x_eval, str(args.mode), hemisphere_sign=hemisphere_sign)
    after_axis = _axis_from_features(proj, str(args.mode), hemisphere_sign=hemisphere_sign)
    report = {
        "train_set": str(args.train_set),
        "mode": str(args.mode),
        "axis_representation": "world local-z xyz" if str(args.mode) != "pos_axis_xy" else "world local-z x/y with fixed hemisphere z reconstruction",
        "axis_xy_hemisphere_sign": float(hemisphere_sign),
        "constraint_dim": int(args.constraint_dim),
        "device": str(device),
        "train_shape": list(x_train.shape),
        "train_axis_angle_error_deg": _stats(_axis_angle_error_deg(train_axis)),
        "eval_seed_axis_angle_error_deg": _stats(_axis_angle_error_deg(before_axis)),
        "projected_axis_angle_error_deg": _stats(_axis_angle_error_deg(after_axis)),
        "train_axis_norm_abs_error": _stats(np.abs(np.linalg.norm(train_axis, axis=1) - 1.0)),
        "projected_axis_norm_abs_error": _stats(np.abs(np.linalg.norm(after_axis, axis=1) - 1.0)),
        "project_steps": _stats(steps.astype(np.float64)),
        "f_abs_stop": f_abs_stop,
        "zero_eps_quantile": zero_eps_quantile,
        "train_model_norm": _stats(np.linalg.norm(f_train, axis=1)),
        "eval_seed_model_norm": _stats(np.linalg.norm(f_before, axis=1)),
        "projected_model_norm": _stats(np.linalg.norm(f_after, axis=1)),
        "cfg": asdict(cfg),
    }
    if train_hist.get("grad_norm_mean") is not None and len(train_hist["grad_norm_mean"]):
        report["final_train_grad_norm_mean"] = np.asarray(train_hist["grad_norm_mean"][-1], dtype=float).tolist()
    if train_hist.get("ortho_err") is not None and len(train_hist["ortho_err"]):
        report["final_train_ortho_err"] = float(np.asarray(train_hist["ortho_err"])[-1])
    if str(args.mode) == "pos_axis_xy":
        report["train_axis_xy_radius"] = _stats(np.linalg.norm(x_train[:, 3:5], axis=1))
        report["eval_seed_axis_xy_radius"] = _stats(np.linalg.norm(x_eval[:, 3:5], axis=1))
        report["projected_axis_xy_radius"] = _stats(np.linalg.norm(proj[:, 3:5], axis=1))
    if str(args.mode) in {"pos_axis", "pos_axis_xy"}:
        train_pos = _split_position(x_train, str(args.mode))
        eval_pos = _split_position(x_eval, str(args.mode))
        proj_pos = _split_position(proj, str(args.mode))
        report.update(
            {
                "train_position_abs_mm": _stats(_position_abs_mm(train_pos)),
                "eval_seed_position_abs_mm": _stats(_position_abs_mm(eval_pos)),
                "projected_position_abs_mm": _stats(_position_abs_mm(proj_pos)),
            }
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    report_path = args.outdir / "axis_report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    torch.save(
        {
            "model_state": model.state_dict(),
            "in_dim": int(x_train.shape[1]),
            "constraint_dim": int(args.constraint_dim),
            "hidden": int(args.hidden),
            "depth": int(args.depth),
            "x_train": x_train,
            "train_hist": train_hist,
            "cfg": report["cfg"],
            "axis_representation": report["axis_representation"],
            "axis_xy_hemisphere_sign": float(hemisphere_sign),
        },
        args.outdir / "axis_model.pt",
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if str(args.mode) == "axis":
        fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2))
    axes[0].hist(_axis_angle_error_deg(before_axis), bins=80, alpha=0.75, label="eval before", color="#64748B")
    axes[0].hist(_axis_angle_error_deg(after_axis), bins=80, alpha=0.75, label="after projection", color="#B45309")
    axes[0].hist(_axis_angle_error_deg(train_axis), bins=60, alpha=0.45, label="train demo", color="#0F766E")
    axes[0].set_title("Tool-axis error")
    axes[0].set_xlabel("angle to -plane normal [deg]")
    axes[0].legend(fontsize=8)

    axes[1].scatter(_axis_angle_error_deg(before_axis), _axis_angle_error_deg(after_axis), s=3, alpha=0.25)
    lim = float(max(np.max(_axis_angle_error_deg(before_axis)), np.max(_axis_angle_error_deg(after_axis)), 1.0))
    axes[1].plot([0, lim], [0, lim], "--", color="black", lw=0.8)
    axes[1].set_xlim(0, lim)
    axes[1].set_ylim(0, lim)
    axes[1].set_title("Per-point axis projection")
    axes[1].set_xlabel("before [deg]")
    axes[1].set_ylabel("after [deg]")

    if str(args.mode) in {"pos_axis", "pos_axis_xy"}:
        axes[2].hist(_position_abs_mm(_split_position(x_eval, str(args.mode))), bins=80, alpha=0.75, label="eval before", color="#64748B")
        axes[2].hist(_position_abs_mm(_split_position(proj, str(args.mode))), bins=80, alpha=0.75, label="after projection", color="#0F766E")
        axes[2].hist(_position_abs_mm(_split_position(x_train, str(args.mode))), bins=60, alpha=0.45, label="train demo", color="#2563EB")
        axes[2].set_title("Position error")
        axes[2].set_xlabel("abs plane distance [mm]")
        axes[2].legend(fontsize=8)

    fig.suptitle(f"ONCL {args.mode}, codim={int(args.constraint_dim)}")
    fig.tight_layout()
    fig_path = args.outdir / "axis_error_distribution.png"
    fig.savefig(fig_path, dpi=240)
    plt.close(fig)

    print(f"[report] {report_path}")
    print(f"[figure] {fig_path}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
