from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.real_iiwa.registry import (  # noqa: E402
    real_iiwa_neural_surface_z_normal,
    real_iiwa_world_to_normalized_xyz,
)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _axis_angle_error_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = _normalize_rows(a)
    b = _normalize_rows(b)
    dots = np.clip(np.sum(a * b, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dots)).astype(np.float64)


def _surface_signed_distance(
    xyz: np.ndarray,
    z_true: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    grad_norm = np.linalg.norm(normal[:, :2] / np.maximum(normal[:, 2:3], 1e-12), axis=1)
    return ((xyz[:, 2] - z_true) / np.sqrt(1.0 + grad_norm**2)).astype(np.float64)


def _stats(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1) * float(scale)
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build real iiwa curved-surface pose samples with tool-z orientation.")
    parser.add_argument(
        "--in-npz",
        type=Path,
        default=Path("datasets/real_iiwa/tasks/curved_surface_position/processed/train_sets/tcp1845_pose_samples_v001.npz"),
    )
    parser.add_argument("--dataset", default="real_iiwa_curved_surface_position")
    parser.add_argument(
        "--out-world",
        type=Path,
        default=Path("datasets/real_iiwa/tasks/curved_surface_position/processed/train_sets/tcp1845_pose_toolz_samples_v001.npz"),
    )
    parser.add_argument(
        "--out-normalized",
        type=Path,
        default=Path("datasets/real_iiwa/tasks/curved_surface_position/processed/train_sets/tcp1845_pose_toolz_samples_normalized_v001.npz"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("datasets/real_iiwa/tasks/curved_surface_position/figures/tcp1845_pose_toolz_summary.json"),
    )
    args = parser.parse_args()

    with np.load(args.in_npz, allow_pickle=False) as data:
        xyz = np.asarray(data["x"], dtype=np.float64)
        time = np.asarray(data["time"], dtype=np.float64)
        recorded_tool_z = _normalize_rows(np.asarray(data["tool_z"], dtype=np.float64))

    z_true, upward_normal = real_iiwa_neural_surface_z_normal(str(args.dataset), xyz[:, 0], xyz[:, 1])
    upward_normal = _normalize_rows(upward_normal.astype(np.float64))
    target_tool_z = -upward_normal
    xyz_true = np.concatenate([xyz[:, :2], z_true.reshape(-1, 1)], axis=1).astype(np.float64)
    xyz_norm = real_iiwa_world_to_normalized_xyz(str(args.dataset), xyz)
    xyz_true_norm = real_iiwa_world_to_normalized_xyz(str(args.dataset), xyz_true)

    x_world = np.concatenate([xyz, recorded_tool_z], axis=1).astype(np.float32)
    x_normalized = np.concatenate([xyz_norm, recorded_tool_z], axis=1).astype(np.float32)
    gt_pose_toolz = np.concatenate([xyz_true, target_tool_z], axis=1).astype(np.float32)
    gt_pose_toolz_normalized = np.concatenate([xyz_true_norm, target_tool_z], axis=1).astype(np.float32)

    args.out_world.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_world,
        x=x_world,
        time=time.astype(np.float64),
        position=xyz.astype(np.float32),
        position_true=xyz_true.astype(np.float32),
        demo_tool_z=recorded_tool_z.astype(np.float32),
        recorded_tool_z=recorded_tool_z.astype(np.float32),
        surface_normal=upward_normal.astype(np.float32),
        target_tool_z=target_tool_z.astype(np.float32),
        gt_tool_z=target_tool_z.astype(np.float32),
        gt_pose_toolz=gt_pose_toolz.astype(np.float32),
    )
    args.out_normalized.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_normalized,
        x=x_normalized,
        x_world=x_world,
        gt_pose_toolz=gt_pose_toolz.astype(np.float32),
        gt_pose_toolz_normalized=gt_pose_toolz_normalized.astype(np.float32),
        time=time.astype(np.float64),
        position_world=xyz.astype(np.float32),
        position_true_world=xyz_true.astype(np.float32),
        position_normalized=xyz_norm.astype(np.float32),
        position_true_normalized=xyz_true_norm.astype(np.float32),
        demo_tool_z=recorded_tool_z.astype(np.float32),
        recorded_tool_z=recorded_tool_z.astype(np.float32),
        surface_normal=upward_normal.astype(np.float32),
        target_tool_z=target_tool_z.astype(np.float32),
        gt_tool_z=target_tool_z.astype(np.float32),
    )

    pos_signed_m = _surface_signed_distance(xyz, z_true.astype(np.float64), upward_normal)
    ori_error_deg = _axis_angle_error_deg(recorded_tool_z, target_tool_z)
    summary = {
        "input": str(args.in_npz),
        "world_output": str(args.out_world),
        "normalized_output": str(args.out_normalized),
        "representation": "[x, y, z, tool_z_x, tool_z_y, tool_z_z]",
        "normalized_representation": "[u_norm, v_norm, z_norm, tool_z_x, tool_z_y, tool_z_z]",
        "orientation_source": "Demo orientation uses the recorded/reconstructed tool local z-axis from the pose sample file.",
        "target_tool_z": "-upward neural surface normal",
        "position_abs_surface_distance_mm": _stats(np.abs(pos_signed_m), scale=1000.0),
        "position_signed_surface_distance_mm": _stats(pos_signed_m, scale=1000.0),
        "recorded_tool_z_vs_target_tool_z_deg": _stats(ori_error_deg),
        "recorded_tool_z_norm": _stats(np.linalg.norm(recorded_tool_z, axis=1)),
        "target_tool_z_norm": _stats(np.linalg.norm(target_tool_z, axis=1)),
        "point_count": int(len(xyz)),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"saved: {args.out_world}")
    print(f"saved: {args.out_normalized}")
    print(f"saved: {args.summary}")
    print(f"x_world shape: {x_world.shape}")
    print(f"x_normalized shape: {x_normalized.shape}")
    print("position abs surface distance mm:", summary["position_abs_surface_distance_mm"])
    print("recorded tool_z vs target tool_z deg:", summary["recorded_tool_z_vs_target_tool_z_deg"])


if __name__ == "__main__":
    main()
