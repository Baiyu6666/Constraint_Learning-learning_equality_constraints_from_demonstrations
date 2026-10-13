from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.real_iiwa.registry import (  # noqa: E402
    REAL_IIWA_DATASETS,
    canonical_real_iiwa_dataset_name,
    real_iiwa_neural_surface_z_normal,
    real_iiwa_world_to_normalized_xyz,
)


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    return arr / np.maximum(np.linalg.norm(arr, axis=1, keepdims=True), 1e-12)


def _stats(values: np.ndarray, scale: float = 1.0) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1) * float(scale)
    if len(v) == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(v)),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }


def _surface_errors(
    *,
    dataset: str,
    position_world: np.ndarray,
    tool_z: np.ndarray,
) -> dict[str, np.ndarray]:
    pos = np.asarray(position_world, dtype=np.float64)
    tool = _normalize(tool_z)
    z_pred, normal = real_iiwa_neural_surface_z_normal(dataset, pos[:, 0], pos[:, 1])
    normal = _normalize(normal)
    grad_norm = np.linalg.norm(normal[:, :2] / np.maximum(normal[:, 2:3], 1e-12), axis=1)
    signed = (pos[:, 2] - np.asarray(z_pred, dtype=np.float64)) / np.sqrt(1.0 + grad_norm**2)
    target_sign = float(REAL_IIWA_DATASETS[canonical_real_iiwa_dataset_name(dataset)].get("target_tool_axis_sign", -1.0))
    target_tool_z = target_sign * normal
    dot = np.sum(tool * target_tool_z, axis=1)
    angle_deg = np.degrees(np.arccos(np.clip(dot, -1.0, 1.0)))
    true_pos = pos.copy()
    true_pos[:, 2] = np.asarray(z_pred, dtype=np.float64)
    return {
        "position_true_world": true_pos.astype(np.float32),
        "surface_normal": normal.astype(np.float32),
        "target_tool_z": target_tool_z.astype(np.float32),
        "signed_position_error_m": signed.astype(np.float32),
        "position_error_m": np.abs(signed).astype(np.float32),
        "orientation_error_deg": angle_deg.astype(np.float32),
    }


def _arc_length_indices_one_trajectory(position_world: np.ndarray, threshold_m: float) -> np.ndarray:
    pos = np.asarray(position_world, dtype=np.float64)
    if len(pos) == 0:
        return np.zeros((0,), dtype=np.int64)
    if threshold_m <= 0.0:
        return np.arange(len(pos), dtype=np.int64)

    indices = [0]
    accum = 0.0
    for i in range(1, len(pos)):
        accum += float(np.linalg.norm(pos[i] - pos[i - 1]))
        if accum >= threshold_m:
            indices.append(i)
            accum = 0.0
    if indices[-1] != len(pos) - 1:
        indices.append(len(pos) - 1)
    return np.asarray(indices, dtype=np.int64)


def _arc_length_indices_by_trajectory(
    position_world: np.ndarray,
    trajectory_offsets: np.ndarray,
    threshold_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    offsets = np.asarray(trajectory_offsets, dtype=np.int64).reshape(-1)
    if len(offsets) < 2:
        idx = _arc_length_indices_one_trajectory(position_world, threshold_m)
        return idx, np.asarray([0, len(idx)], dtype=np.int64)

    all_idx: list[np.ndarray] = []
    out_offsets = [0]
    for start, end in zip(offsets[:-1], offsets[1:]):
        start_i = int(start)
        end_i = int(end)
        if end_i <= start_i:
            out_offsets.append(out_offsets[-1])
            continue
        local = _arc_length_indices_one_trajectory(position_world[start_i:end_i], threshold_m)
        global_idx = local + start_i
        all_idx.append(global_idx)
        out_offsets.append(out_offsets[-1] + len(global_idx))
    if not all_idx:
        return np.zeros((0,), dtype=np.int64), np.asarray(out_offsets, dtype=np.int64)
    return np.concatenate(all_idx).astype(np.int64), np.asarray(out_offsets, dtype=np.int64)


def _trajectory_ids(offsets: np.ndarray, n_samples: int) -> np.ndarray:
    off = np.asarray(offsets, dtype=np.int64)
    if len(off) < 2:
        return np.zeros((n_samples,), dtype=np.int64)
    return np.searchsorted(off[1:], np.arange(n_samples, dtype=np.int64), side="right").astype(np.int64)


def _aligned(data: dict[str, np.ndarray], key: str, indices: np.ndarray, *, default: np.ndarray | None = None) -> np.ndarray:
    if key in data:
        return np.asarray(data[key])[indices]
    if default is None:
        raise KeyError(key)
    return np.asarray(default)[indices]


def _source_error_stats(source_data: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    if "true_constraint_signed_position_error_m" in source_data:
        signed = np.asarray(source_data["true_constraint_signed_position_error_m"], dtype=np.float64)
        out["source_true_constraint_signed_position_error_mm"] = _stats(signed, scale=1000.0)
        out["source_true_constraint_abs_position_error_mm"] = _stats(np.abs(signed), scale=1000.0)
    if "true_constraint_orientation_error_deg" in source_data:
        out["source_true_constraint_orientation_error_deg"] = _stats(source_data["true_constraint_orientation_error_deg"])
    if "normal_tracking_error_m" in source_data:
        out["source_normal_tracking_error_mm"] = _stats(source_data["normal_tracking_error_m"], scale=1000.0)
    if "orientation_tracking_error_deg" in source_data:
        out["source_orientation_tracking_error_deg"] = _stats(source_data["orientation_tracking_error_deg"])
    return out


def _plot_error_distribution(
    *,
    position_error_m: np.ndarray,
    signed_position_error_m: np.ndarray,
    orientation_error_deg: np.ndarray,
    output: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.8), constrained_layout=True)
    axes[0].hist(np.asarray(position_error_m) * 1000.0, bins=60, color="#2f6f9f", alpha=0.9)
    axes[0].set_xlabel("abs position error (mm)")
    axes[0].set_ylabel("count")
    axes[0].set_title("Position")

    axes[1].hist(np.asarray(signed_position_error_m) * 1000.0, bins=60, color="#8a5a44", alpha=0.9)
    axes[1].axvline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("signed position error (mm)")
    axes[1].set_title("Signed Position")

    axes[2].hist(np.asarray(orientation_error_deg), bins=60, color="#4d8f5b", alpha=0.9)
    axes[2].set_xlabel("tool-z orientation error (deg)")
    axes[2].set_title("Orientation")
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _plot_xy_error_maps(
    *,
    position_world: np.ndarray,
    position_error_m: np.ndarray,
    orientation_error_deg: np.ndarray,
    output: Path,
    title: str,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pos = np.asarray(position_world)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), constrained_layout=True)
    sc0 = axes[0].scatter(pos[:, 0], pos[:, 1], c=np.asarray(position_error_m) * 1000.0, s=5, cmap="magma")
    axes[0].set_aspect("equal", adjustable="box")
    axes[0].set_xlabel("x (m)")
    axes[0].set_ylabel("y (m)")
    axes[0].set_title("Position Error")
    fig.colorbar(sc0, ax=axes[0], label="mm")

    sc1 = axes[1].scatter(pos[:, 0], pos[:, 1], c=np.asarray(orientation_error_deg), s=5, cmap="viridis")
    axes[1].set_aspect("equal", adjustable="box")
    axes[1].set_xlabel("x (m)")
    axes[1].set_ylabel("y (m)")
    axes[1].set_title("Tool-Z Error")
    fig.colorbar(sc1, ax=axes[1], label="deg")
    fig.suptitle(title)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def _write_npz(
    *,
    path: Path,
    source_npz: Path,
    source_data: dict[str, np.ndarray],
    indices: np.ndarray,
    x_full: np.ndarray,
    xyz_norm_full: np.ndarray,
    xyz_true_norm_full: np.ndarray,
    tool_z_full: np.ndarray,
    errors: dict[str, np.ndarray],
    trajectory_offsets_out: np.ndarray,
    arc_threshold_m: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_source = len(x_full)
    source_offsets = np.asarray(source_data["trajectory_offsets"], dtype=np.int64)
    trajectory_id_full = _trajectory_ids(source_offsets, n_source)
    position_world = np.asarray(source_data["actual_position"], dtype=np.float32)
    position_true_world = np.asarray(errors["position_true_world"], dtype=np.float32)
    gt_pose_toolz_world = np.concatenate([position_true_world, np.asarray(errors["target_tool_z"], dtype=np.float32)], axis=1)
    gt_pose_toolz_normalized = np.concatenate([xyz_true_norm_full, np.asarray(errors["target_tool_z"], dtype=np.float32)], axis=1)

    np.savez_compressed(
        path,
        x=x_full[indices].astype(np.float32),
        source_x=x_full.astype(np.float32),
        source_index=indices.astype(np.int64),
        time=_aligned(source_data, "ros_time_s", indices).astype(np.float64),
        segment_time_s=_aligned(source_data, "segment_time_s", indices).astype(np.float64),
        position_world=position_world[indices].astype(np.float32),
        position_normalized=xyz_norm_full[indices].astype(np.float32),
        tool_z=tool_z_full[indices].astype(np.float32),
        position_true_world=position_true_world[indices].astype(np.float32),
        position_true_normalized=xyz_true_norm_full[indices].astype(np.float32),
        surface_normal=np.asarray(errors["surface_normal"], dtype=np.float32)[indices],
        target_tool_z=np.asarray(errors["target_tool_z"], dtype=np.float32)[indices],
        position_error_m=np.asarray(errors["position_error_m"], dtype=np.float32)[indices],
        signed_position_error_m=np.asarray(errors["signed_position_error_m"], dtype=np.float32)[indices],
        orientation_error_deg=np.asarray(errors["orientation_error_deg"], dtype=np.float32)[indices],
        x_world=np.concatenate([position_world, tool_z_full.astype(np.float32)], axis=1)[indices].astype(np.float32),
        gt_pose_toolz=gt_pose_toolz_world[indices].astype(np.float32),
        gt_pose_toolz_normalized=gt_pose_toolz_normalized[indices].astype(np.float32),
        planned_position_world=_aligned(source_data, "planned_position", indices).astype(np.float32),
        nominal_planned_position_world=_aligned(source_data, "nominal_planned_position", indices).astype(np.float32),
        planned_tool_z=_normalize(_aligned(source_data, "planned_tool_z", indices)).astype(np.float32),
        actual_orientation_xyzw=_aligned(source_data, "actual_orientation_xyzw", indices).astype(np.float32),
        joint_position_rad=_aligned(source_data, "joint_position_rad", indices).astype(np.float32),
        joint_velocity_radps=_aligned(source_data, "joint_velocity_radps", indices).astype(np.float32),
        desired_joint_position_rad=_aligned(source_data, "desired_joint_position_rad", indices).astype(np.float32),
        desired_joint_velocity_radps=_aligned(source_data, "desired_joint_velocity_radps", indices).astype(np.float32),
        mode=_aligned(source_data, "mode", indices).astype(np.int64),
        route_index=_aligned(source_data, "route_index", indices).astype(np.int64),
        plan_id=_aligned(source_data, "plan_id", indices).astype(np.int64),
        sample_index=_aligned(source_data, "sample_index", indices).astype(np.int64),
        trajectory_id=trajectory_id_full[indices].astype(np.int64),
        trajectory_offsets=trajectory_offsets_out.astype(np.int64),
        source_trajectory_offsets=source_offsets.astype(np.int64),
        arc_length_threshold_m=np.asarray([arc_threshold_m], dtype=np.float32),
        source_npz=np.asarray(str(source_npz)),
        representation=np.asarray("[surface-model normalized xyz, actual tool_z_x, actual tool_z_y, actual tool_z_z]"),
    )


def build_oracle_dataset(
    *,
    input_npz: Path,
    dataset: str,
    output_npz: Path,
    full_output_npz: Path | None,
    summary_json: Path,
    figure_prefix: Path,
    arc_mm: float,
) -> dict[str, Any]:
    with np.load(input_npz, allow_pickle=False) as loaded:
        source_data = {k: np.asarray(loaded[k]) for k in loaded.files}

    position_world = np.asarray(source_data["actual_position"], dtype=np.float64)
    tool_z = _normalize(source_data["actual_tool_z"]).astype(np.float32)
    errors = _surface_errors(dataset=dataset, position_world=position_world, tool_z=tool_z)
    xyz_norm = real_iiwa_world_to_normalized_xyz(dataset, position_world)
    xyz_true_norm = real_iiwa_world_to_normalized_xyz(dataset, errors["position_true_world"])
    x_full = np.concatenate([xyz_norm.astype(np.float32), tool_z.astype(np.float32)], axis=1)

    threshold_m = float(arc_mm) / 1000.0
    source_offsets = np.asarray(source_data["trajectory_offsets"], dtype=np.int64)
    keep_idx, keep_offsets = _arc_length_indices_by_trajectory(position_world, source_offsets, threshold_m)
    full_idx = np.arange(len(position_world), dtype=np.int64)
    full_offsets = source_offsets

    if full_output_npz is not None:
        _write_npz(
            path=full_output_npz,
            source_npz=input_npz,
            source_data=source_data,
            indices=full_idx,
            x_full=x_full,
            xyz_norm_full=xyz_norm,
            xyz_true_norm_full=xyz_true_norm,
            tool_z_full=tool_z,
            errors=errors,
            trajectory_offsets_out=full_offsets,
            arc_threshold_m=0.0,
        )

    _write_npz(
        path=output_npz,
        source_npz=input_npz,
        source_data=source_data,
        indices=keep_idx,
        x_full=x_full,
        xyz_norm_full=xyz_norm,
        xyz_true_norm_full=xyz_true_norm,
        tool_z_full=tool_z,
        errors=errors,
        trajectory_offsets_out=keep_offsets,
        arc_threshold_m=threshold_m,
    )

    title = input_npz.parent.name
    hist_path = figure_prefix.with_name(f"{figure_prefix.name}_error_hist.png")
    xy_path = figure_prefix.with_name(f"{figure_prefix.name}_xy_error_map.png")
    hist_kept_path = figure_prefix.with_name(f"{figure_prefix.name}_arc{arc_mm:g}mm_error_hist.png")
    xy_kept_path = figure_prefix.with_name(f"{figure_prefix.name}_arc{arc_mm:g}mm_xy_error_map.png")
    _plot_error_distribution(
        position_error_m=errors["position_error_m"],
        signed_position_error_m=errors["signed_position_error_m"],
        orientation_error_deg=errors["orientation_error_deg"],
        output=hist_path,
        title=f"{title}: all samples",
    )
    _plot_xy_error_maps(
        position_world=position_world,
        position_error_m=errors["position_error_m"],
        orientation_error_deg=errors["orientation_error_deg"],
        output=xy_path,
        title=f"{title}: all samples",
    )
    _plot_error_distribution(
        position_error_m=errors["position_error_m"][keep_idx],
        signed_position_error_m=errors["signed_position_error_m"][keep_idx],
        orientation_error_deg=errors["orientation_error_deg"][keep_idx],
        output=hist_kept_path,
        title=f"{title}: arc {arc_mm:g} mm",
    )
    _plot_xy_error_maps(
        position_world=position_world[keep_idx],
        position_error_m=errors["position_error_m"][keep_idx],
        orientation_error_deg=errors["orientation_error_deg"][keep_idx],
        output=xy_kept_path,
        title=f"{title}: arc {arc_mm:g} mm",
    )

    steps = []
    for start, end in zip(keep_offsets[:-1], keep_offsets[1:]):
        idx = keep_idx[int(start) : int(end)]
        if len(idx) > 1:
            steps.append(np.linalg.norm(np.diff(position_world[idx], axis=0), axis=1))
    step = np.concatenate(steps) if steps else np.zeros((0,), dtype=np.float64)
    summary: dict[str, Any] = {
        "input": str(input_npz),
        "dataset": dataset,
        "output": str(output_npz),
        "full_output": str(full_output_npz) if full_output_npz is not None else None,
        "representation": "[surface-model normalized xyz, actual tool_z_x, actual tool_z_y, actual tool_z_z]",
        "source_points": int(len(position_world)),
        "source_trajectories": int(max(0, len(source_offsets) - 1)),
        "output_points": int(len(keep_idx)),
        "arc_length_threshold_mm": float(arc_mm),
        "kept_fraction": float(len(keep_idx) / max(1, len(position_world))),
        "actual_position_bounds_m": {
            "min": [float(v) for v in np.min(position_world, axis=0)],
            "max": [float(v) for v in np.max(position_world, axis=0)],
        },
        "arc_consecutive_xyz_step_mm": _stats(step, scale=1000.0),
        "all_position_abs_error_mm": _stats(errors["position_error_m"], scale=1000.0),
        "all_position_signed_error_mm": _stats(errors["signed_position_error_m"], scale=1000.0),
        "all_orientation_error_deg": _stats(errors["orientation_error_deg"]),
        "kept_position_abs_error_mm": _stats(errors["position_error_m"][keep_idx], scale=1000.0),
        "kept_position_signed_error_mm": _stats(errors["signed_position_error_m"][keep_idx], scale=1000.0),
        "kept_orientation_error_deg": _stats(errors["orientation_error_deg"][keep_idx]),
        "source_logged_error_stats": _source_error_stats(source_data),
        "figures": {
            "all_error_hist": str(hist_path),
            "all_xy_error_map": str(xy_path),
            "kept_error_hist": str(hist_kept_path),
            "kept_xy_error_map": str(xy_kept_path),
        },
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a real-iiwa oracle-controller demonstration dataset.")
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path("/home/baiyu/iiwa_share/oracle_controller_demonstrations_20260720/oracle_demonstrations.npz"),
    )
    parser.add_argument("--dataset", default="RealKuka")
    parser.add_argument("--demo-name", default=None, help="Defaults to the input npz parent directory name.")
    parser.add_argument("--output-npz", type=Path, default=None)
    parser.add_argument("--full-output-npz", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    parser.add_argument("--figure-prefix", type=Path, default=None)
    parser.add_argument("--arc-mm", type=float, default=3.0)
    parser.add_argument("--no-full-output", action="store_true")
    args = parser.parse_args()

    demo_name = str(args.demo_name or args.input_npz.parent.name)
    task_root = Path("datasets/real_iiwa/tasks/RealKuka")
    train_root = task_root / "processed" / "train_sets"
    figure_root = task_root / "figures"
    arc_tag = f"arc{float(args.arc_mm):g}mm"
    output_npz = args.output_npz or train_root / f"{demo_name}_actual_toolz_{arc_tag}_v001.npz"
    full_output_npz = args.full_output_npz or train_root / f"{demo_name}_actual_toolz_full_v001.npz"
    summary_json = args.summary_json or figure_root / f"{demo_name}_summary.json"
    figure_prefix = args.figure_prefix or figure_root / demo_name
    full_output = None if args.no_full_output else full_output_npz
    summary = build_oracle_dataset(
        input_npz=args.input_npz,
        dataset=str(args.dataset),
        output_npz=output_npz,
        full_output_npz=full_output,
        summary_json=summary_json,
        figure_prefix=figure_prefix,
        arc_mm=float(args.arc_mm),
    )
    print(json.dumps(summary, indent=2))
    print(f"saved: {output_npz}")
    if full_output is not None:
        print(f"saved: {full_output}")
    print(f"saved: {summary_json}")


if __name__ == "__main__":
    main()
