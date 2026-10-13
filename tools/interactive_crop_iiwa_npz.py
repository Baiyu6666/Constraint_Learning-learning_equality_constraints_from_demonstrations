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

from datasets.ros.transforms import quat_to_matrix_xyzw


TIME_GROUPS = {
    "t_pose": ("ee_pos", "ee_quat", "tcp_pos", "tcp_quat"),
    "t_joint": ("q", "dq", "effort"),
    "t_ee_vel": ("ee_linear_vel", "ee_angular_vel", "tcp_linear_vel", "tcp_angular_vel"),
    "t_tf_pose": ("tf_ee_pos", "tf_ee_quat", "tf_tcp_pos", "tf_tcp_quat"),
    "t_torque_command": ("torque_command",),
    "t_command_status": ("command_status",),
    "t_measure": ("measure_linear", "measure_angular"),
    "t_plotvar": ("plotvar",),
}


def _task_root(path: Path) -> Path | None:
    p = path.resolve()
    for parent in p.parents:
        if parent.parent.name == "tasks":
            return parent
    return None


def _default_out_path(path: Path) -> Path:
    p = path.resolve()
    for parent in p.parents:
        if parent.parent.name == "tasks":
            task_root = parent
            rel_parts = p.relative_to(task_root).parts
            if "extracted" in rel_parts:
                idx = rel_parts.index("extracted")
                stem = rel_parts[idx + 1] if idx + 1 < len(rel_parts) else path.stem
                return task_root / "processed" / "crops" / f"{stem}_manual_crop.npz"
            return task_root / "processed" / "crops" / f"{path.stem}_manual_crop.npz"
    return path.with_name(path.stem + "_cropped.npz")


def _default_figure_path(out: Path) -> Path:
    task_root = _task_root(out)
    if task_root is not None:
        return task_root / "figures" / f"{out.stem}_crop.png"
    return out.with_name(f"{out.stem}_crop.png")


def _orientation_dirs(quat_xyzw: np.ndarray, axis: str) -> np.ndarray:
    axis_vec = {
        "x": np.asarray([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.asarray([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.asarray([0.0, 0.0, 1.0], dtype=np.float64),
    }[axis]
    dirs = np.empty((quat_xyzw.shape[0], 3), dtype=np.float64)
    for i, q in enumerate(quat_xyzw):
        dirs[i] = quat_to_matrix_xyzw(q) @ axis_vec
    dirs /= np.maximum(np.linalg.norm(dirs, axis=1, keepdims=True), 1e-12)
    return dirs


def _axis_limits(points: np.ndarray) -> list[tuple[float, float]]:
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = 0.5 * (lo + hi)
    span = max(float((hi - lo).max()), 0.05)
    pad = 0.55 * span
    return [(float(c - pad), float(c + pad)) for c in center]


def _crop_time_group(arrays: dict[str, np.ndarray], time_key: str, t0: float, t1: float) -> dict[str, np.ndarray]:
    if time_key not in arrays:
        return {}
    t = np.asarray(arrays[time_key], dtype=np.float64)
    mask = (t >= float(t0)) & (t <= float(t1))
    out = {time_key: arrays[time_key][mask]}
    for key in TIME_GROUPS[time_key]:
        if key in arrays and arrays[key].shape[:1] == t.shape[:1]:
            out[key] = arrays[key][mask]
    return out


def _crop_arrays(path: Path, start: int, end: int) -> tuple[dict[str, np.ndarray], dict[str, object]]:
    with np.load(path, allow_pickle=True) as data:
        arrays = {k: data[k] for k in data.files}

    t_pose = np.asarray(arrays["t_pose"], dtype=np.float64)
    n_pose = int(t_pose.shape[0])
    start = int(np.clip(start, 0, n_pose - 2))
    end = int(np.clip(end, start + 1, n_pose - 1))
    t0 = float(t_pose[start])
    t1 = float(t_pose[end])

    cropped: dict[str, np.ndarray] = {}
    assigned: set[str] = set()
    for time_key in TIME_GROUPS:
        group = _crop_time_group(arrays, time_key, t0, t1)
        cropped.update(group)
        assigned.update(group)

    for key, value in arrays.items():
        if key in assigned or key.startswith("crop_") or key == "metadata_json":
            continue
        if value.shape[:1] == (n_pose,):
            cropped[key] = value[start : end + 1]
            assigned.add(key)
        elif key not in cropped:
            cropped[key] = value

    points = np.asarray(cropped["tcp_pos"], dtype=np.float64)
    path_len = float(np.sum(np.linalg.norm(np.diff(points, axis=0), axis=1))) if points.shape[0] > 1 else 0.0
    meta = {
        "source_npz": str(path),
        "crop_start_pose_index": int(start),
        "crop_end_pose_index_inclusive": int(end),
        "crop_start_time": t0,
        "crop_end_time": t1,
        "original_pose_count": n_pose,
        "cropped_pose_count": int(points.shape[0]),
        "cropped_path_length_m": path_len,
        "time_group_counts": {
            key: int(cropped[key].shape[0])
            for key in TIME_GROUPS
            if key in cropped and hasattr(cropped[key], "shape")
        },
    }
    cropped["crop_metadata_json"] = np.asarray(json.dumps(meta, indent=2))
    return cropped, meta


def _save_crop(path: Path, out: Path, start: int, end: int, *, figure_path: Path | None = None) -> dict[str, object]:
    cropped, meta = _crop_arrays(path, start, end)
    if figure_path is not None:
        meta["figure"] = str(figure_path)
        cropped["crop_metadata_json"] = np.asarray(json.dumps(meta, indent=2))
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **cropped)
    return meta


def _plot_saved_crop(
    cropped_path: Path,
    figure_path: Path,
    *,
    orientation_axis: str,
    arrow_every: int,
    arrow_length: float,
    dpi: int = 220,
) -> None:
    import matplotlib.pyplot as plt

    with np.load(cropped_path, allow_pickle=True) as data:
        points = np.asarray(data["tcp_pos"], dtype=np.float64)
        quat = np.asarray(data["tcp_quat"], dtype=np.float64)
        meta = json.loads(str(data["crop_metadata_json"])) if "crop_metadata_json" in data else {}

    idx = np.arange(points.shape[0], dtype=np.int64)
    dirs = _orientation_dirs(quat, orientation_axis)
    limits = _axis_limits(points)
    fig = plt.figure(figsize=(10.5, 5.8))
    ax3d = fig.add_subplot(1, 2, 1, projection="3d")
    ax_z = fig.add_subplot(1, 2, 2)

    ax3d.plot(points[:, 0], points[:, 1], points[:, 2], color="#1f77b4", lw=0.85, alpha=0.85)
    ax3d.scatter(points[0, 0], points[0, 1], points[0, 2], color="limegreen", edgecolor="black", s=80)
    ax3d.scatter(points[-1, 0], points[-1, 1], points[-1, 2], color="red", edgecolor="black", s=85, marker="X")
    arrow_step = max(1, int(arrow_every))
    arrow_idx = idx[::arrow_step]
    if arrow_idx.size and arrow_idx[-1] != idx[-1]:
        arrow_idx = np.concatenate([arrow_idx, idx[-1:]])
    ax3d.quiver(
        points[arrow_idx, 0],
        points[arrow_idx, 1],
        points[arrow_idx, 2],
        dirs[arrow_idx, 0],
        dirs[arrow_idx, 1],
        dirs[arrow_idx, 2],
        length=float(arrow_length),
        normalize=True,
        color="#1f77b4",
        linewidth=0.85,
        alpha=0.9,
        arrow_length_ratio=0.35,
    )
    ax3d.set_xlim(*limits[0])
    ax3d.set_ylim(*limits[1])
    ax3d.set_zlim(*limits[2])
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.view_init(elev=24, azim=-58)
    ax3d.set_title(f"{cropped_path.stem}: cropped TCP trajectory")

    ax_z.plot(idx, points[:, 2], color="#1f77b4", lw=0.8)
    ax_z.scatter([0, idx[-1]], [points[0, 2], points[-1, 2]], c=["limegreen", "red"], s=35, zorder=3)
    ax_z.set_xlabel("cropped pose index")
    ax_z.set_ylabel("tcp z [m]")
    ax_z.grid(True, alpha=0.25)
    ax_z.set_title(f"{points.shape[0]} poses, {float(meta.get('cropped_path_length_m', 0.0)):.3f} m")

    fig.tight_layout()
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_path, dpi=int(dpi))
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively crop one iiwa demonstration npz without modifying raw data.")
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--initial-start-index", type=int, default=None)
    parser.add_argument("--initial-end-index", type=int, default=None)
    parser.add_argument("--initial-start-frac", type=float, default=0.05)
    parser.add_argument("--initial-end-frac", type=float, default=0.95)
    parser.add_argument("--display-stride", type=int, default=10)
    parser.add_argument("--arrow-every", type=int, default=500)
    parser.add_argument("--arrow-length", type=float, default=0.035)
    parser.add_argument("--orientation-axis", choices=("x", "y", "z"), default="z")
    parser.add_argument("--figure-out", type=Path, default=None)
    parser.add_argument("--no-save-figure", action="store_true", help="Do not save a crop inspection figure.")
    parser.add_argument("--figure-dpi", type=int, default=220)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    path = args.dataset
    out = args.out or _default_out_path(path)
    figure_out = args.figure_out or _default_figure_path(out)

    import matplotlib.pyplot as plt
    from matplotlib.widgets import Button, RangeSlider

    with np.load(path, allow_pickle=True) as data:
        t = np.asarray(data["t_pose"], dtype=np.float64)
        points = np.asarray(data["tcp_pos"], dtype=np.float64)
        quat = np.asarray(data["tcp_quat"], dtype=np.float64)

    n = int(points.shape[0])
    if n < 2:
        raise ValueError(f"Need at least two pose samples, got {n}")

    start0 = int(np.floor(float(args.initial_start_frac) * (n - 1)))
    end0 = int(np.floor(float(args.initial_end_frac) * (n - 1)))
    if args.initial_start_index is not None:
        start0 = int(args.initial_start_index)
    if args.initial_end_index is not None:
        end0 = int(args.initial_end_index)
    start0 = int(np.clip(start0, 0, n - 2))
    end0 = int(np.clip(end0, start0 + 1, n - 1))

    dt = np.diff(t)
    step = np.linalg.norm(np.diff(points, axis=0), axis=1)
    speed = np.divide(step, np.maximum(dt, 1e-9), out=np.zeros_like(step), where=dt > 0)
    speed = np.concatenate([[speed[0] if speed.size else 0.0], speed])
    idx_all = np.arange(n, dtype=np.int64)
    dirs = _orientation_dirs(quat, str(args.orientation_axis))

    fig = plt.figure(figsize=(13.5, 8.0))
    gs = fig.add_gridspec(3, 2, height_ratios=[2.2, 1.0, 0.22], width_ratios=[1.35, 1.0])
    ax3d = fig.add_subplot(gs[:2, 0], projection="3d")
    ax_z = fig.add_subplot(gs[0, 1])
    ax_speed = fig.add_subplot(gs[1, 1], sharex=ax_z)
    ax_slider = fig.add_subplot(gs[2, :])

    stride = max(1, int(args.display_stride))
    shown = idx_all[::stride]
    ax3d.plot(points[shown, 0], points[shown, 1], points[shown, 2], color="0.75", lw=0.8, alpha=0.75)
    selected_line, = ax3d.plot([], [], [], color="#1f77b4", lw=2.0)
    start_scatter = ax3d.scatter([], [], [], color="limegreen", edgecolor="black", s=90)
    end_scatter = ax3d.scatter([], [], [], color="red", edgecolor="black", s=90, marker="X")
    quiver_holder = {"artist": None}

    limits = _axis_limits(points)
    ax3d.set_xlim(*limits[0])
    ax3d.set_ylim(*limits[1])
    ax3d.set_zlim(*limits[2])
    ax3d.set_box_aspect((1, 1, 1))
    ax3d.set_xlabel("x [m]")
    ax3d.set_ylabel("y [m]")
    ax3d.set_zlabel("z [m]")
    ax3d.view_init(elev=24, azim=-58)

    ax_z.plot(idx_all, points[:, 2], color="#1f77b4", lw=0.9)
    ax_z.set_ylabel("tcp z [m]")
    ax_z.grid(True, alpha=0.25)
    ax_speed.plot(idx_all, speed, color="#d62728", lw=0.8)
    ax_speed.set_ylabel("speed [m/s]")
    ax_speed.set_xlabel("pose index")
    ax_speed.grid(True, alpha=0.25)
    z_start = ax_z.axvline(start0, color="limegreen", lw=1.5)
    z_end = ax_z.axvline(end0, color="red", lw=1.5)
    s_start = ax_speed.axvline(start0, color="limegreen", lw=1.5)
    s_end = ax_speed.axvline(end0, color="red", lw=1.5)

    slider = RangeSlider(
        ax=ax_slider,
        label="crop pose index",
        valmin=0,
        valmax=n - 1,
        valinit=(start0, end0),
        valstep=1,
    )

    button_ax = fig.add_axes([0.84, 0.015, 0.12, 0.045])
    save_button = Button(button_ax, "Save crop")
    status = fig.text(0.02, 0.02, "", fontsize=9)

    current = {"start": start0, "end": end0}

    def update(_value=None) -> None:
        a_raw, b_raw = slider.val
        a = int(round(float(a_raw)))
        b = int(round(float(b_raw)))
        a = int(np.clip(a, 0, n - 2))
        b = int(np.clip(b, a + 1, n - 1))
        current["start"] = a
        current["end"] = b

        sel_idx = idx_all[a : b + 1 : stride]
        if sel_idx.size == 0 or sel_idx[-1] != b:
            sel_idx = np.concatenate([sel_idx, np.asarray([b], dtype=np.int64)])
        selected_line.set_data(points[sel_idx, 0], points[sel_idx, 1])
        selected_line.set_3d_properties(points[sel_idx, 2])
        start_scatter._offsets3d = ([points[a, 0]], [points[a, 1]], [points[a, 2]])
        end_scatter._offsets3d = ([points[b, 0]], [points[b, 1]], [points[b, 2]])

        if quiver_holder["artist"] is not None:
            quiver_holder["artist"].remove()
        arrow_step = max(1, int(args.arrow_every))
        arrow_idx = idx_all[a : b + 1 : arrow_step]
        if arrow_idx.size and arrow_idx[-1] != b:
            arrow_idx = np.concatenate([arrow_idx, np.asarray([b], dtype=np.int64)])
        quiver_holder["artist"] = ax3d.quiver(
            points[arrow_idx, 0],
            points[arrow_idx, 1],
            points[arrow_idx, 2],
            dirs[arrow_idx, 0],
            dirs[arrow_idx, 1],
            dirs[arrow_idx, 2],
            length=float(args.arrow_length),
            normalize=True,
            color="#1f77b4",
            linewidth=1.0,
            alpha=0.9,
            arrow_length_ratio=0.35,
        )

        for line in (z_start, s_start):
            line.set_xdata([a, a])
        for line in (z_end, s_end):
            line.set_xdata([b, b])
        path_len = float(np.sum(step[a:b])) if b > a else 0.0
        ax3d.set_title(f"{path.name}: selected {a}:{b} ({b - a + 1} poses, {path_len:.3f} m)")
        status.set_text(f"out: {out}    keys: s=save, q=close")
        fig.canvas.draw_idle()

    def save(_event=None) -> None:
        meta = _save_crop(
            path,
            out,
            current["start"],
            current["end"],
            figure_path=(None if bool(args.no_save_figure) else figure_out),
        )
        if not bool(args.no_save_figure):
            _plot_saved_crop(
                out,
                figure_out,
                orientation_axis=str(args.orientation_axis),
                arrow_every=int(args.arrow_every),
                arrow_length=float(args.arrow_length),
                dpi=int(args.figure_dpi),
            )
        print(json.dumps(meta, indent=2))
        status.set_text(f"saved: {out}    figure: {figure_out if not bool(args.no_save_figure) else 'disabled'}")
        fig.canvas.draw_idle()

    def on_key(event) -> None:
        if event.key == "s":
            save()
        elif event.key in {"q", "escape"}:
            plt.close(fig)

    slider.on_changed(update)
    save_button.on_clicked(save)
    fig.canvas.mpl_connect("key_press_event", on_key)

    print("Use the bottom range slider to choose [start, end]. Press 's' or click 'Save crop'. Press 'q' to close.")
    print(f"Input:  {path}")
    print(f"Output: {out}")
    if not bool(args.no_save_figure):
        print(f"Figure: {figure_out}")
    update()
    plt.show()


if __name__ == "__main__":
    main()
