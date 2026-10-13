from __future__ import annotations

import argparse
import csv
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from evaluation.evaluator import (
    _dual_arm_curve_center_tnb_from_s,
    _dual_arm_pose_params,
    _rpy_zyx_to_rotmat_batch,
    sine_surface_z_and_normal_from_xy,
)
from planning.kinematics import planar_fk, spatial_fk_n3


DATASETS = [
    ("12d_dual_arm_traj", "12DDualArm"),
    ("6d_surf_normal_traj", "6DSurfNormal"),
    ("3d_planar_arm_line_n3_traj", "3DPlanarArmLine"),
]


def _load_ckpt(bench: str, dataset: str) -> tuple[np.ndarray, Any]:
    ckpt = torch.load(os.path.join(bench, "oncl", f"{dataset}_oncl_model.pt"), map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**dict(ckpt.get("cfg", {}) or {}))
    return np.asarray(ckpt["x_train"], dtype=np.float32), cfg


def _summ(vals: np.ndarray) -> tuple[float, float, float, float]:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return float(np.mean(vals)), float(np.median(vals)), float(np.percentile(vals, 95)), float(np.max(vals))


def _nn_idx(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    try:
        from scipy.spatial import cKDTree  # type: ignore

        _, idx = cKDTree(ref.astype(np.float64)).query(x.astype(np.float64), k=1)
        return np.asarray(idx, dtype=np.int64)
    except Exception:
        d2 = np.sum((x[:, None, :] - ref[None, :, :]) ** 2, axis=2)
        return np.argmin(d2, axis=1).astype(np.int64)


def _angle_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return ((a - b + np.pi) % (2.0 * np.pi) - np.pi).astype(np.float32)


def _angular_nn_idx(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    # For small batches in this diagnostic, dense angular distance is clearer than
    # embedding angles in sin/cos and then interpreting branch switches.
    d2 = np.sum(_angle_diff(x[:, None, :], ref[None, :, :]) ** 2, axis=2)
    return np.argmin(d2, axis=1).astype(np.int64)


def _local_z_from_rpy(rpy: np.ndarray) -> np.ndarray:
    return _rpy_zyx_to_rotmat_batch(rpy.astype(np.float32))[:, :, 2].astype(np.float32)


def _surf_normal_metrics(x_train: np.ndarray, x_eval: np.ndarray, proj: np.ndarray, cfg: Any) -> dict[str, np.ndarray]:
    lo = np.min(x_train[:, :3], axis=0)
    hi = np.max(x_train[:, :3], axis=0)
    below = np.maximum(lo.reshape(1, 3) - proj[:, :3], 0.0)
    above = np.maximum(proj[:, :3] - hi.reshape(1, 3), 0.0)
    xyz_bbox_dist = np.linalg.norm(below + above, axis=1).astype(np.float32)
    xy_outside = (
        (proj[:, 0] < lo[0])
        | (proj[:, 0] > hi[0])
        | (proj[:, 1] < lo[1])
        | (proj[:, 1] > hi[1])
    ).astype(np.float32)

    z_true, normal = sine_surface_z_and_normal_from_xy(proj[:, 0], proj[:, 1], cfg=cfg)
    z_axis = _local_z_from_rpy(proj[:, 3:6])
    cosv = np.clip(np.sum(z_axis * normal.astype(np.float32), axis=1), -1.0, 1.0)
    normal_angle_deg = np.degrees(np.arccos(cosv)).astype(np.float32)
    xy_shift = np.linalg.norm(proj[:, :2] - x_eval[:, :2], axis=1).astype(np.float32)
    return {
        "xy_outside_train_bbox": xy_outside,
        "xyz_distance_to_train_bbox": xyz_bbox_dist,
        "xy_shift_from_query": xy_shift,
        "surface_z_error": np.abs(proj[:, 2] - z_true.astype(np.float32)).astype(np.float32),
        "normal_angle_deg": normal_angle_deg,
    }


def _dual_arm_s_from_center_xy(center_xy: np.ndarray, cfg: Any) -> np.ndarray:
    s_grid = np.linspace(-1.0, 1.0, 4096, dtype=np.float32)
    center_grid, _, _, _ = _dual_arm_curve_center_tnb_from_s(s_grid, cfg)
    idx = _nn_idx(center_xy.astype(np.float32), center_grid[:, :2].astype(np.float32))
    return s_grid[idx].astype(np.float32)


def _dual_arm_metrics(x_train: np.ndarray, x_eval: np.ndarray, proj: np.ndarray, cfg: Any) -> dict[str, np.ndarray]:
    p = _dual_arm_pose_params(cfg)
    center = (0.5 * (proj[:, 0:3] + proj[:, 6:9])).astype(np.float32)
    center_eval = (0.5 * (x_eval[:, 0:3] + x_eval[:, 6:9])).astype(np.float32)
    center_train = (0.5 * (x_train[:, 0:3] + x_train[:, 6:9])).astype(np.float32)

    s_proj = _dual_arm_s_from_center_xy(center[:, :2], cfg)
    s_query = _dual_arm_s_from_center_xy(center_eval[:, :2], cfg)
    s_train = _dual_arm_s_from_center_xy(center_train[:, :2], cfg)
    nearest_train = _nn_idx(center[:, :2], center_train[:, :2])
    ds_nearest_train = np.abs(s_proj - s_train[nearest_train]).astype(np.float32)

    u_proj = (center[:, 2] - float(p["z_base"])).astype(np.float32)
    u_train = (center_train[:, 2] - float(p["z_base"])).astype(np.float32)
    du_nearest_train = np.abs(u_proj - u_train[nearest_train]).astype(np.float32)

    R_left = _rpy_zyx_to_rotmat_batch(proj[:, 3:6])
    R_right = _rpy_zyx_to_rotmat_batch(proj[:, 9:12])
    center_curve, tangent, y_axis, z_axis = _dual_arm_curve_center_tnb_from_s(s_proj, cfg)
    del center_curve, y_axis
    left_tangent_angle = np.degrees(
        np.arccos(np.clip(np.sum(R_left[:, :, 0] * tangent, axis=1), -1.0, 1.0))
    ).astype(np.float32)
    right_tangent_angle = np.degrees(
        np.arccos(np.clip(np.sum(R_right[:, :, 0] * tangent, axis=1), -1.0, 1.0))
    ).astype(np.float32)
    left_z_angle = np.degrees(
        np.arccos(np.clip(np.sum(R_left[:, :, 2] * z_axis, axis=1), -1.0, 1.0))
    ).astype(np.float32)
    right_z_angle = np.degrees(
        np.arccos(np.clip(np.sum(R_right[:, :, 2] * z_axis, axis=1), -1.0, 1.0))
    ).astype(np.float32)
    return {
        "s_proj": s_proj,
        "abs_s_change_from_query": np.abs(s_proj - s_query).astype(np.float32),
        "abs_s_to_nearest_demo": ds_nearest_train,
        "abs_u_to_nearest_demo": du_nearest_train,
        "near_curve_endpoint": (np.abs(s_proj) > 0.95).astype(np.float32),
        "left_tangent_angle_deg": left_tangent_angle,
        "right_tangent_angle_deg": right_tangent_angle,
        "left_z_axis_angle_deg": left_z_angle,
        "right_z_axis_angle_deg": right_z_angle,
    }


def _ellipse_metrics(x_train: np.ndarray, x_eval: np.ndarray, proj: np.ndarray) -> dict[str, np.ndarray]:
    proj_branch = np.where(proj[:, 2] >= 0.0, 1.0, -1.0).astype(np.float32)
    train_branch = np.where(x_train[:, 2] >= 0.0, 1.0, -1.0).astype(np.float32)
    nearest_train = _angular_nn_idx(proj[:, :3], x_train[:, :3])
    nearest_query_train = _angular_nn_idx(x_eval[:, :3], x_train[:, :3])
    branch_diff_nearest = (proj_branch != train_branch[nearest_train]).astype(np.float32)
    branch_diff_query_nearest = (proj_branch != train_branch[nearest_query_train]).astype(np.float32)

    ee = spatial_fk_n3(proj.astype(np.float32), [1.0, 0.8])[:, -1, :].astype(np.float32)
    t = np.arctan2(ee[:, 1] / 0.95, ee[:, 0] / 1.35).astype(np.float32)
    ellipse_xy = np.stack([1.35 * np.cos(t), 0.95 * np.sin(t)], axis=1).astype(np.float32)
    ee_xy_dist = np.linalg.norm(ee[:, :2] - ellipse_xy, axis=1).astype(np.float32)
    return {
        "proj_elbow_up_branch": (proj_branch > 0).astype(np.float32),
        "branch_diff_vs_nearest_demo": branch_diff_nearest,
        "branch_diff_vs_query_nearest_demo": branch_diff_query_nearest,
        "ee_xy_dist_to_same_angle_ellipse": ee_xy_dist,
        "ee_z_error": np.abs(ee[:, 2] - 0.35).astype(np.float32),
    }


def _planar_arm_line_branch(q: np.ndarray) -> np.ndarray:
    l1, l2, l3 = 1.0, 0.8, 0.6
    y_line = 0.35
    q = q.astype(np.float32)
    rhs = (y_line - l1 * np.sin(q[:, 0]) - l2 * np.sin(q[:, 0] + q[:, 1])) / l3
    rhs = np.clip(rhs, -1.0, 1.0).astype(np.float32)
    a = np.arcsin(rhs).astype(np.float32)
    q12 = q[:, 0] + q[:, 1]
    q3a = _angle_diff(a - q12, 0.0)
    q3b = _angle_diff((np.pi - a) - q12, 0.0)
    da = np.abs(_angle_diff(q[:, 2], q3a))
    db = np.abs(_angle_diff(q[:, 2], q3b))
    return (db < da).astype(np.float32)


def _planar_arm_line_metrics(x_train: np.ndarray, x_eval: np.ndarray, proj: np.ndarray) -> dict[str, np.ndarray]:
    proj_branch = _planar_arm_line_branch(proj)
    train_branch = _planar_arm_line_branch(x_train)
    eval_branch = _planar_arm_line_branch(x_eval)
    nearest_train = _angular_nn_idx(proj[:, :3], x_train[:, :3])
    ee = planar_fk(proj.astype(np.float32), [1.0, 0.8, 0.6])[:, -1, :].astype(np.float32)
    ee_eval = planar_fk(x_eval.astype(np.float32), [1.0, 0.8, 0.6])[:, -1, :].astype(np.float32)
    return {
        "proj_branch_b": proj_branch,
        "branch_diff_vs_nearest_demo": (proj_branch != train_branch[nearest_train]).astype(np.float32),
        "branch_diff_vs_query": (proj_branch != eval_branch).astype(np.float32),
        "ee_line_y_abs_err": np.abs(ee[:, 1] - 0.35).astype(np.float32),
        "ee_xy_shift_from_query": np.linalg.norm(ee - ee_eval, axis=1).astype(np.float32),
    }


def _metrics(dataset: str, x_train: np.ndarray, x_eval: np.ndarray, proj: np.ndarray, cfg: Any) -> dict[str, np.ndarray]:
    if dataset == "12d_dual_arm_traj":
        return _dual_arm_metrics(x_train, x_eval, proj, cfg)
    if dataset == "6d_surf_normal_traj":
        return _surf_normal_metrics(x_train, x_eval, proj, cfg)
    if dataset == "3d_spatial_arm_ellip_n3_traj":
        return _ellipse_metrics(x_train, x_eval, proj)
    if dataset == "3d_planar_arm_line_n3_traj":
        return _planar_arm_line_metrics(x_train, x_eval, proj)
    raise ValueError(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose whether high-error projections are branch switches or off-manifold sheets.")
    parser.add_argument("--bench", default="outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed")
    parser.add_argument("--diag-dir", default="outputs/analysis/high_projection_error_diagnostics")
    parser.add_argument("--outdir", default="outputs/analysis/high_projection_error_diagnostics")
    args = parser.parse_args()

    bench = os.path.abspath(args.bench)
    diag_dir = os.path.abspath(args.diag_dir)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    rows: list[dict[str, object]] = []
    for dataset, label in DATASETS:
        x_train, cfg = _load_ckpt(bench, dataset)
        data = np.load(os.path.join(diag_dir, f"{dataset}_default_point_diagnostics.npz"))
        x_eval = data["x_eval"].astype(np.float32)
        proj = data["proj"].astype(np.float32)
        top = data["top_default_error"].astype(bool)
        rest = ~top
        metric_map = {
            "proj_error": data["err"].astype(np.float32),
            "learned_residual": data["resid"].astype(np.float32),
            "normalized_demo_distance": data["demo_dist"].astype(np.float32),
            "broad_indicator": (data["query_type"].astype(np.int32) == 0).astype(np.float32),
            **_metrics(dataset, x_train, x_eval, proj, cfg),
        }
        for group, mask in (("top5_high_error", top), ("remaining_95pct", rest)):
            for metric, values in metric_map.items():
                mean, median, p95, max_val = _summ(values[mask])
                rows.append(
                    {
                        "Dataset": label,
                        "group": group,
                        "n": int(np.sum(mask)),
                        "metric": metric,
                        "mean": mean,
                        "median": median,
                        "p95": p95,
                        "max": max_val,
                    }
                )
        print(f"[done] {label}")

    out_csv = os.path.join(outdir, "projection_branch_or_sheet_diagnostics.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["Dataset", "group", "n", "metric", "mean", "median", "p95", "max"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()
