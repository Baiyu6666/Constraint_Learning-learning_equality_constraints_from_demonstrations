#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.real_iiwa.registry import plane_constraint
from methods.models.mlp import MLP


def _default_report_path(ckpt_path: Path) -> Path | None:
    for name in ("axis_report.json", "orientation_only_report.json", "analytic_pose_error_report.json"):
        p = ckpt_path.with_name(name)
        if p.exists():
            return p
    return None


def _stats(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p90": float(np.quantile(v, 0.90)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(np.max(v)),
        "min": float(np.min(v)),
    }


def _parse_thresholds(text: str) -> list[float]:
    vals: list[float] = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        vals.append(float(part))
    return vals


def _tangent_basis(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(target, dtype=np.float64)
    target = target / np.maximum(np.linalg.norm(target), 1e-12)
    ref = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(ref @ target)) > 0.9:
        ref = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
    e1 = ref - float(ref @ target) * target
    e1 = e1 / np.maximum(np.linalg.norm(e1), 1e-12)
    e2 = np.cross(target, e1)
    e2 = e2 / np.maximum(np.linalg.norm(e2), 1e-12)
    return e1, e2


def _axis_to_tangent_deg(axis: np.ndarray, target: np.ndarray, e1: np.ndarray, e2: np.ndarray) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float64)
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    forward = np.clip(a @ target, 1e-12, 1.0)
    u = np.degrees(np.arctan2(a @ e1, forward))
    v = np.degrees(np.arctan2(a @ e2, forward))
    return np.stack([u, v], axis=1)


def _load_model(ckpt_path: Path) -> tuple[MLP, dict, np.ndarray]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = MLP(
        in_dim=int(ckpt["in_dim"]),
        hidden=int(ckpt["hidden"]),
        depth=int(ckpt["depth"]),
        out_dim=int(ckpt["constraint_dim"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt, np.asarray(ckpt["x_train"], dtype=np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot ONCL h(x) slices over tool local-Z directions.")
    parser.add_argument("ckpt", type=Path, help="ONCL checkpoint containing x_train and model_state.")
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--grid-n", type=int, default=260)
    parser.add_argument("--threshold", type=float, default=None, help="Contour threshold for ||h||. Defaults to train residual p90.")
    parser.add_argument("--zero-eps-quantile", type=float, default=None, help="Train residual quantile used when --threshold is omitted. Defaults to report zero_eps_quantile or 90.")
    parser.add_argument(
        "--threshold-quantiles",
        default="10,25,50,75,90",
        help="Comma-separated train ||h|| quantiles to draw as contour thresholds.",
    )
    parser.add_argument("--position-index", type=int, default=-1, help="For pos+axis models, fixed position index. -1 uses median position.")
    parser.add_argument("--hemisphere-sign", choices=("auto", "positive", "negative"), default="auto")
    parser.add_argument("--slice-mode", choices=("local-xy", "unit-disk", "tangent"), default="local-xy")
    parser.add_argument("--xy-pad-ratio", type=float, default=0.45, help="Padding around train/target local_z_x/local_z_y range.")
    parser.add_argument("--tangent-pad-ratio", type=float, default=0.45, help="Padding around train/target tangent-angle range.")
    parser.add_argument("--tangent-min-half-range-deg", type=float, default=8.0, help="Minimum half-range for tangent-angle slices.")
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ckpt_path = args.ckpt.resolve()
    model, ckpt, x_train = _load_model(ckpt_path)
    in_dim = int(ckpt["in_dim"])
    if in_dim not in (3, 6):
        raise ValueError(f"expected axis-only in_dim=3 or pos+axis in_dim=6, got {in_dim}")
    mode = "axis" if in_dim == 3 else "pos_axis"
    train_axis = x_train[:, :3] if mode == "axis" else x_train[:, 3:6]
    train_axis = train_axis / np.maximum(np.linalg.norm(train_axis, axis=1, keepdims=True), 1e-12)

    if args.hemisphere_sign == "positive":
        sign = 1.0
    elif args.hemisphere_sign == "negative":
        sign = -1.0
    else:
        med_z = float(np.median(train_axis[:, 2]))
        sign = -1.0 if med_z < 0.0 else 1.0

    if mode == "pos_axis":
        if int(args.position_index) >= 0:
            fixed_pos = x_train[int(args.position_index), :3].astype(np.float32)
        else:
            fixed_pos = np.median(x_train[:, :3], axis=0).astype(np.float32)
    else:
        fixed_pos = None

    normal, _ = plane_constraint()
    target = -normal
    target = target / np.linalg.norm(target)

    grid_n = int(args.grid_n)
    plot_xlabel = "local_z_x"
    plot_ylabel = "local_z_y"
    plot_title_suffix = "local_z_x/local_z_y"
    train_plot_xy = train_axis[:, 0:2]
    target_plot_xy = target[None, 0:2]

    if str(args.slice_mode) == "tangent":
        e1, e2 = _tangent_basis(target)
        train_plot_xy = _axis_to_tangent_deg(train_axis, target, e1, e2)
        target_plot_xy = np.zeros((1, 2), dtype=np.float64)
        uv_lo = np.vstack([train_plot_xy, target_plot_xy]).min(axis=0)
        uv_hi = np.vstack([train_plot_xy, target_plot_xy]).max(axis=0)
        uv_span = uv_hi - uv_lo
        center = 0.5 * (uv_lo + uv_hi)
        half = 0.5 * float(np.max(uv_span)) * max(1.0 + 2.0 * float(args.tangent_pad_ratio), 1e-6)
        half = max(half, float(args.tangent_min_half_range_deg))
        x_vals = np.linspace(float(center[0] - half), float(center[0] + half), grid_n, dtype=np.float64)
        y_vals = np.linspace(float(center[1] - half), float(center[1] + half), grid_n, dtype=np.float64)
        plot_xlabel = "tool-z tangent tilt e1 [deg]"
        plot_ylabel = "tool-z tangent tilt e2 [deg]"
        plot_title_suffix = "target-axis tangent plane"
    elif str(args.slice_mode) == "local-xy":
        xy_train_target = np.vstack([train_axis[:, 0:2], target[None, 0:2]])
        xy_lo = xy_train_target.min(axis=0)
        xy_hi = xy_train_target.max(axis=0)
        xy_span = np.maximum(xy_hi - xy_lo, np.asarray([0.08, 0.08], dtype=np.float64))
        center = 0.5 * (xy_lo + xy_hi)
        half = 0.5 * float(np.max(xy_span)) * max(1.0 + 2.0 * float(args.xy_pad_ratio), 1e-6)
        xy_lo = center - half
        xy_hi = center + half
        x_vals = np.linspace(float(xy_lo[0]), float(xy_hi[0]), grid_n, dtype=np.float64)
        y_vals = np.linspace(float(xy_lo[1]), float(xy_hi[1]), grid_n, dtype=np.float64)
    else:
        x_vals = np.linspace(-1.0, 1.0, grid_n, dtype=np.float64)
        y_vals = np.linspace(-1.0, 1.0, grid_n, dtype=np.float64)
    xx_plot, yy_plot = np.meshgrid(x_vals, y_vals, indexing="xy")
    if str(args.slice_mode) == "tangent":
        disk = np.ones_like(xx_plot, dtype=bool)
        u = np.tan(np.deg2rad(xx_plot[disk]))
        v = np.tan(np.deg2rad(yy_plot[disk]))
        axis64 = target.reshape(1, 3) + u[:, None] * e1.reshape(1, 3) + v[:, None] * e2.reshape(1, 3)
        axis64 = axis64 / np.maximum(np.linalg.norm(axis64, axis=1, keepdims=True), 1e-12)
        axis = axis64.astype(np.float32)
    else:
        disk = (xx_plot**2 + yy_plot**2) <= 1.0
        zz = np.full_like(xx_plot, np.nan, dtype=np.float64)
        zz[disk] = sign * np.sqrt(np.maximum(1.0 - xx_plot[disk] ** 2 - yy_plot[disk] ** 2, 0.0))
        axis = np.stack([xx_plot[disk], yy_plot[disk], zz[disk]], axis=1).astype(np.float32)
    axis_grid = np.full((*xx_plot.shape, 3), np.nan, dtype=np.float64)
    axis_grid[disk] = axis.astype(np.float64)
    if mode == "pos_axis":
        pos = np.repeat(fixed_pos.reshape(1, 3), axis.shape[0], axis=0)
        query = np.concatenate([pos, axis], axis=1).astype(np.float32)
    else:
        query = axis.astype(np.float32)

    with torch.no_grad():
        h = model(torch.from_numpy(query)).detach().cpu().numpy()
    if h.ndim == 1:
        h = h[:, None]
    h_norm = np.linalg.norm(h, axis=1)
    h_img = np.full_like(xx_plot, np.nan, dtype=np.float64)
    h_img[disk] = h_norm.astype(np.float64)

    with torch.no_grad():
        h_train = model(torch.from_numpy(x_train.astype(np.float32))).detach().cpu().numpy()
    if h_train.ndim == 1:
        h_train = h_train[:, None]
    train_h_norm = np.linalg.norm(h_train, axis=1)

    report_path = _default_report_path(ckpt_path)
    report = json.loads(report_path.read_text()) if report_path is not None else {}
    threshold = args.threshold
    if threshold is None:
        zero_eps_quantile = float(
            args.zero_eps_quantile
            if args.zero_eps_quantile is not None
            else report.get("zero_eps_quantile", 90.0)
        )
        threshold = float(np.quantile(train_h_norm, zero_eps_quantile / 100.0))
    else:
        zero_eps_quantile = None
    threshold = float(threshold)
    quantiles = _parse_thresholds(str(args.threshold_quantiles))
    contour_levels = [float(np.quantile(train_h_norm, q / 100.0)) for q in quantiles]
    contour_pairs = sorted(
        {
            round(level, 12): (float(level), float(q))
            for level, q in zip(contour_levels, quantiles)
            if np.nanmin(h_img) <= float(level) <= np.nanmax(h_img)
        }.values(),
        key=lambda item: item[0],
    )

    low = disk.copy()
    low[disk] = h_norm <= threshold
    low_axis = axis_grid[low] if np.any(low) else np.empty((0, 3), dtype=np.float64)
    if len(low_axis):
        dot = np.clip(low_axis @ target, -1.0, 1.0)
        low_angle_deg = np.degrees(np.arccos(dot))
    else:
        low_angle_deg = np.asarray([], dtype=np.float64)

    slice_summary = {
        "checkpoint": str(ckpt_path),
        "mode": mode,
        "fixed_position": fixed_pos.astype(float).tolist() if fixed_pos is not None else None,
        "slice_mode": str(args.slice_mode),
        "hemisphere_sign_for_local_z_z": float(sign),
        "threshold": threshold,
        "zero_eps_quantile": zero_eps_quantile,
        "contour_threshold_quantiles": [
            {"quantile": float(q), "threshold": float(level)}
            for level, q in contour_pairs
        ],
        "train_h_norm": _stats(train_h_norm),
        "slice_h_norm": _stats(h_norm),
        "plot_x_range": [float(x_vals[0]), float(x_vals[-1])],
        "plot_y_range": [float(y_vals[0]), float(y_vals[-1])],
        "low_h_fraction_on_slice": float(np.mean(low[disk])),
        "low_h_angle_to_target_deg": _stats(low_angle_deg) if len(low_angle_deg) else None,
        "target_axis": target.astype(float).tolist(),
    }

    out_path = args.out or ckpt_path.with_name("h_slice_local_z_xy.png")
    json_path = out_path.with_suffix(".json")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.3, 6.3))
    finite = np.isfinite(h_img)
    vmax = float(np.nanquantile(h_img, 0.98)) if np.any(finite) else 1.0
    im = ax.imshow(
        h_img,
        extent=[float(x_vals[0]), float(x_vals[-1]), float(y_vals[0]), float(y_vals[-1])],
        origin="lower",
        cmap="viridis",
        vmin=0.0,
        vmax=max(vmax, threshold * 1.5, 1e-6),
        aspect="equal",
    )
    fig.colorbar(im, ax=ax, label="||h(x)||")
    if contour_pairs:
        levels = [p[0] for p in contour_pairs]
        colors = ["#FFFFFF", "#E0F2FE", "#93C5FD", "#3B82F6", "#1E3A8A", "#111827"]
        cs = ax.contour(
            xx_plot,
            yy_plot,
            h_img,
            levels=levels,
            colors=colors[: len(levels)],
            linewidths=[1.6 if q <= 25 else 1.1 for _, q in contour_pairs],
        )
        label_map = {level: f"p{q:g}" for level, q in contour_pairs}
        ax.clabel(cs, fmt=lambda val: label_map.get(float(val), f"{val:.3g}"), fontsize=8, inline=True)
    ax.scatter(train_plot_xy[:, 0], train_plot_xy[:, 1], s=10, c="#F97316", alpha=0.85, label="train local_z")
    ax.scatter(
        target_plot_xy[:, 0],
        target_plot_xy[:, 1],
        s=80,
        c="#EF4444",
        marker="*",
        edgecolors="black",
        linewidths=0.6,
        label="-plane normal",
    )
    ax.set_xlim(float(x_vals[0]), float(x_vals[-1]))
    ax.set_ylim(float(y_vals[0]), float(y_vals[-1]))
    ax.set_xlabel(plot_xlabel)
    ax.set_ylabel(plot_ylabel)
    ax.set_title(f"{mode}: ONCL ||h|| slice over {plot_title_suffix}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)

    json_path.write_text(json.dumps(slice_summary, indent=2) + "\n")
    print(f"[figure] {out_path}")
    print(f"[summary] {json_path}")
    print(json.dumps(slice_summary, indent=2))


if __name__ == "__main__":
    main()
