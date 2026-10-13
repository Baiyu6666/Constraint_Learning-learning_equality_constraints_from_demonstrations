#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.real_iiwa.registry import plane_constraint
from datasets.ros.transforms import quat_to_matrix_xyzw


def _task_root(path: Path) -> Path | None:
    p = path.resolve()
    for parent in p.parents:
        if parent.parent.name == "tasks":
            return parent
    return None


def _default_out(input_path: Path) -> Path:
    task = _task_root(input_path)
    if task is not None:
        return task / "figures" / f"{input_path.stem}_plane_error_distribution.png"
    return input_path.with_name(f"{input_path.stem}_plane_error_distribution.png")


def _axis_world(quat_xyzw: np.ndarray, axis: str) -> np.ndarray:
    axis_vec = {
        "x": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    }[axis]
    dirs = np.empty((quat_xyzw.shape[0], 3), dtype=np.float64)
    for i, q in enumerate(quat_xyzw):
        dirs[i] = quat_to_matrix_xyzw(q) @ axis_vec
    return dirs / np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)


def _stats(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "std": float(np.std(v)),
        "p05": float(np.quantile(v, 0.05)),
        "p90": float(np.quantile(v, 0.90)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(np.max(v)),
        "min": float(np.min(v)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot real iiwa plane position/orientation error distributions.")
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        default=Path("datasets/real_iiwa/tasks/plane_surface_fixed_orientation/processed/crops/demo_000_manual_crop.npz"),
        help="Cropped iiwa demonstration npz. Defaults to fixed-orientation plane demo.",
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--stats-out", type=Path, default=None)
    parser.add_argument("--orientation-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument(
        "--target-direction",
        choices=("normal", "minus-normal", "unsigned-normal"),
        default="minus-normal",
        help="For fixed orientation, tcp local +Z is expected to point toward the plane, i.e. -normal.",
    )
    parser.add_argument("--dpi", type=int, default=240)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input.resolve()
    out_path = args.out or _default_out(input_path)
    stats_out = args.stats_out or out_path.with_suffix(".json")

    with np.load(input_path, allow_pickle=False) as data:
        pos = np.asarray(data["tcp_pos"], dtype=np.float64)
        quat = np.asarray(data["tcp_quat"], dtype=np.float64)
        t = np.asarray(data["t_pose"], dtype=np.float64) if "t_pose" in data else np.arange(pos.shape[0], dtype=np.float64)

    normal, plane_d = plane_constraint()
    signed_m = pos @ normal + float(plane_d)
    abs_mm = np.abs(signed_m) * 1000.0
    signed_mm = signed_m * 1000.0

    dirs = _axis_world(quat, str(args.orientation_axis))
    target = normal.copy()
    if str(args.target_direction) == "minus-normal":
        target = -target
    dot = np.einsum("ij,j->i", dirs, target)
    if str(args.target_direction) == "unsigned-normal":
        dot = np.abs(dot)
    angle_deg = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))

    stats = {
        "input": str(input_path),
        "plane": {
            "normal": normal.astype(float).tolist(),
            "d": float(plane_d),
            "implicit": "normal[0]*x + normal[1]*y + normal[2]*z + d = 0",
        },
        "position_signed_mm": _stats(signed_mm),
        "position_abs_mm": _stats(abs_mm),
        "orientation_error_deg": _stats(angle_deg),
        "orientation_axis": str(args.orientation_axis),
        "target_direction": str(args.target_direction),
        "count": int(pos.shape[0]),
    }

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(11.0, 7.8))
    ax = axes[0, 0]
    ax.hist(signed_mm, bins=80, color="#2563EB", alpha=0.88)
    ax.axvline(0.0, color="black", lw=1.0)
    ax.set_title("Position signed distance to plane")
    ax.set_xlabel("signed distance [mm]")
    ax.set_ylabel("count")

    ax = axes[0, 1]
    ax.hist(abs_mm, bins=80, color="#0F766E", alpha=0.88)
    ax.set_title("Position absolute distance to plane")
    ax.set_xlabel("absolute distance [mm]")
    ax.set_ylabel("count")

    ax = axes[1, 0]
    ax.hist(angle_deg, bins=80, color="#B45309", alpha=0.88)
    ax.set_title(f"Orientation error: local +{str(args.orientation_axis).upper()} vs target")
    ax.set_xlabel("angle error [deg]")
    ax.set_ylabel("count")

    ax = axes[1, 1]
    tt = t - float(t[0]) if t.size else np.arange(pos.shape[0], dtype=np.float64)
    ax.plot(tt, signed_mm, color="#2563EB", lw=0.65, alpha=0.75, label="position signed [mm]")
    ax2 = ax.twinx()
    ax2.plot(tt, angle_deg, color="#B45309", lw=0.65, alpha=0.75, label="orientation [deg]")
    ax.set_title("Errors over time")
    ax.set_xlabel("time [s]" if t.size else "index")
    ax.set_ylabel("signed distance [mm]")
    ax2.set_ylabel("orientation error [deg]")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper right", fontsize=8)

    fig.suptitle(f"{input_path.parent.parent.parent.name}/{input_path.stem} plane constraint errors")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=int(args.dpi))
    plt.close(fig)

    stats_out.parent.mkdir(parents=True, exist_ok=True)
    stats_out.write_text(json.dumps(stats, indent=2) + "\n")
    print(f"[figure] {out_path}")
    print(f"[stats] {stats_out}")
    print(
        "position abs mm: "
        f"median={stats['position_abs_mm']['median']:.3f}, "
        f"mean={stats['position_abs_mm']['mean']:.3f}, "
        f"p95={stats['position_abs_mm']['p95']:.3f}, "
        f"max={stats['position_abs_mm']['max']:.3f}"
    )
    print(
        "orientation deg: "
        f"median={stats['orientation_error_deg']['median']:.3f}, "
        f"mean={stats['orientation_error_deg']['mean']:.3f}, "
        f"p95={stats['orientation_error_deg']['p95']:.3f}, "
        f"max={stats['orientation_error_deg']['max']:.3f}"
    )


if __name__ == "__main__":
    main()

