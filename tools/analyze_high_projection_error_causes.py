from __future__ import annotations

import argparse
import csv
import os
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from evaluation.evaluator import (
    _dual_arm_pose_analytic_target_raw,
    _rotation_geodesic_deg_rpy,
    dual_arm_pose_true_constraint_error_arrays,
    sine_surface_z_and_normal_from_xy,
)
from planning.kinematics import planar_fk, spatial_fk_n3


DATASETS = [
    ("12d_dual_arm_traj", "12DDualArm"),
    ("6d_surf_normal_traj", "6DSurfNormal"),
    ("3d_planar_arm_line_n3_traj", "3DPlanarArmLine"),
]


def _load_cfg(bench: str, dataset: str) -> Any:
    ckpt = torch.load(os.path.join(bench, "oncl", f"{dataset}_oncl_model.pt"), map_location="cpu", weights_only=False)
    return SimpleNamespace(**dict(ckpt.get("cfg", {}) or {}))


def _summ(vals: np.ndarray) -> dict[str, float]:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(np.max(vals)),
    }


def _rpy_zyx_to_local_z_np(rpy: np.ndarray) -> np.ndarray:
    roll = rpy[:, 0]
    pitch = rpy[:, 1]
    yaw = rpy[:, 2]
    cr = np.cos(roll)
    sr = np.sin(roll)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    z = np.stack(
        [
            cy * sp * cr + sy * sr,
            sy * sp * cr - cy * sr,
            cp * cr,
        ],
        axis=1,
    ).astype(np.float32)
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    return z


def _surf_normal_components(x: np.ndarray, cfg: Any) -> dict[str, np.ndarray]:
    z_true, n_true = sine_surface_z_and_normal_from_xy(x[:, 0], x[:, 1], cfg=cfg)
    z_axis = _rpy_zyx_to_local_z_np(x[:, 3:6].astype(np.float32))
    cosv = np.clip(np.sum(z_axis * n_true.astype(np.float32), axis=1), -1.0, 1.0)
    axis_angle_deg = np.degrees(np.arccos(cosv)).astype(np.float32)
    axis_chord = np.linalg.norm(z_axis - n_true.astype(np.float32), axis=1).astype(np.float32)
    z_err = np.abs(x[:, 2].astype(np.float32) - z_true.astype(np.float32))
    return {
        "surface_z_abs_err": z_err.astype(np.float32),
        "tool_axis_chord_err": axis_chord.astype(np.float32),
        "tool_axis_angle_deg": axis_angle_deg.astype(np.float32),
    }


def _ellipse_components(q: np.ndarray) -> dict[str, np.ndarray]:
    ee = spatial_fk_n3(q.astype(np.float32), [1.0, 0.8])[:, -1, :].astype(np.float32)
    a, b, z0 = 1.35, 0.95, 0.35
    t = np.linspace(-np.pi, np.pi, 4096, endpoint=False, dtype=np.float32)
    ell = np.stack([a * np.cos(t), b * np.sin(t), np.full_like(t, z0)], axis=1).astype(np.float32)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        d, idx = cKDTree(ell.astype(np.float64)).query(ee.astype(np.float64), k=1)
        target = ell[idx]
        total = np.asarray(d, dtype=np.float32)
    except Exception:
        d2 = np.sum((ee[:, None, :] - ell[None, :, :]) ** 2, axis=2)
        idx = np.argmin(d2, axis=1)
        target = ell[idx]
        total = np.sqrt(np.maximum(d2[np.arange(len(ee)), idx], 0.0)).astype(np.float32)
    xy_err = np.linalg.norm(ee[:, :2] - target[:, :2], axis=1).astype(np.float32)
    z_err = np.abs(ee[:, 2] - z0).astype(np.float32)
    rho = np.sqrt(np.maximum(ee[:, 0] ** 2 + ee[:, 1] ** 2, 1e-12)).astype(np.float32)
    return {
        "ee_to_ellipse_dist": total,
        "ee_xy_ellipse_err": xy_err,
        "ee_z_abs_err": z_err,
        "ee_radius": rho,
    }


def _planar_arm_line_components(q: np.ndarray) -> dict[str, np.ndarray]:
    ee = planar_fk(q.astype(np.float32), [1.0, 0.8, 0.6])[:, -1, :].astype(np.float32)
    y_line = 0.35
    return {
        "ee_line_y_abs_err": np.abs(ee[:, 1] - y_line).astype(np.float32),
        "ee_x_coord": ee[:, 0].astype(np.float32),
        "ee_y_coord": ee[:, 1].astype(np.float32),
    }


def _dual_arm_extra_components(x: np.ndarray, cfg: Any) -> dict[str, np.ndarray]:
    out = dual_arm_pose_true_constraint_error_arrays(x[:, :12].astype(np.float32), cfg)
    target = _dual_arm_pose_analytic_target_raw(x[:, :12].astype(np.float32), cfg)
    center = 0.5 * (x[:, 0:3] + x[:, 6:9])
    center_t = 0.5 * (target[:, 0:3] + target[:, 6:9])
    out["center_xy_err"] = np.linalg.norm(center[:, :2] - center_t[:, :2], axis=1).astype(np.float32)
    out["center_z_err"] = np.abs(center[:, 2] - center_t[:, 2]).astype(np.float32)
    out["left_right_ori_disagreement_deg"] = _rotation_geodesic_deg_rpy(x[:, 3:6], x[:, 9:12]).astype(np.float32)
    return out


def _component_dict(dataset: str, x: np.ndarray, cfg: Any) -> dict[str, np.ndarray]:
    if dataset == "12d_dual_arm_traj":
        return _dual_arm_extra_components(x, cfg)
    if dataset == "6d_surf_normal_traj":
        return _surf_normal_components(x, cfg)
    if dataset == "3d_spatial_arm_ellip_n3_traj":
        return _ellipse_components(x)
    if dataset == "3d_planar_arm_line_n3_traj":
        return _planar_arm_line_components(x)
    raise ValueError(dataset)


def _add_group_rows(rows: list[dict[str, object]], label: str, group: str, n: int, values: dict[str, np.ndarray], mask: np.ndarray) -> None:
    for metric, arr in values.items():
        s = _summ(np.asarray(arr)[mask])
        rows.append(
            {
                "Dataset": label,
                "group": group,
                "n": int(n),
                "metric": metric,
                "mean": s["mean"],
                "median": s["median"],
                "p95": s["p95"],
                "max": s["max"],
            }
        )


def main() -> None:
    p = argparse.ArgumentParser(description="Analyze why high projection-error points occur.")
    p.add_argument("--bench", default="outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed")
    p.add_argument("--diag-dir", default="outputs/analysis/high_projection_error_diagnostics")
    p.add_argument("--outdir", default="outputs/analysis/high_projection_error_diagnostics")
    args = p.parse_args()

    bench = os.path.abspath(args.bench)
    diag_dir = os.path.abspath(args.diag_dir)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    rows: list[dict[str, object]] = []
    overview: list[dict[str, object]] = []
    for dataset, label in DATASETS:
        cfg = _load_cfg(bench, dataset)
        z = np.load(os.path.join(diag_dir, f"{dataset}_default_point_diagnostics.npz"))
        proj = z["proj"].astype(np.float32)
        top = z["top_default_error"].astype(bool)
        rest = ~top
        query_type = z["query_type"].astype(np.int32)
        base_values = {
            "proj_error": z["err"].astype(np.float32),
            "learned_residual": z["resid"].astype(np.float32),
            "raw_learned_residual": z["raw_resid"].astype(np.float32),
            "normalized_demo_distance": z["demo_dist"].astype(np.float32),
            "clip_delta": z["clip_delta"].astype(np.float32),
            "broad_indicator": (query_type == 0).astype(np.float32),
        }
        values = {**base_values, **_component_dict(dataset, proj, cfg)}
        _add_group_rows(rows, label, "top5_high_error", int(np.sum(top)), values, top)
        _add_group_rows(rows, label, "remaining_95pct", int(np.sum(rest)), values, rest)

        # Correlations help distinguish OOD distance from residual-driven failures.
        err = z["err"].astype(np.float64)
        demo = z["demo_dist"].astype(np.float64)
        resid = z["resid"].astype(np.float64)
        corr_demo = float(np.corrcoef(err, demo)[0, 1]) if np.std(err) > 0 and np.std(demo) > 0 else float("nan")
        corr_resid = float(np.corrcoef(err, resid)[0, 1]) if np.std(err) > 0 and np.std(resid) > 0 else float("nan")
        overview.append(
            {
                "Dataset": label,
                "top5_n": int(np.sum(top)),
                "top5_proj_error_median": float(np.median(err[top])),
                "rest_proj_error_median": float(np.median(err[rest])),
                "top5_broad_frac": float(np.mean(query_type[top] == 0)),
                "rest_broad_frac": float(np.mean(query_type[rest] == 0)),
                "top5_demo_dist_median": float(np.median(demo[top])),
                "rest_demo_dist_median": float(np.median(demo[rest])),
                "corr_error_demo_dist": corr_demo,
                "corr_error_learned_residual": corr_resid,
            }
        )
        print(f"[done] {label}")

    out_csv = os.path.join(outdir, "high_projection_error_cause_components.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["Dataset", "group", "n", "metric", "mean", "median", "p95", "max"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    overview_csv = os.path.join(outdir, "high_projection_error_cause_overview.csv")
    with open(overview_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(overview[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(overview)

    print(f"[saved] {overview_csv}")
    print(f"[saved] {out_csv}")

    try:
        import matplotlib.pyplot as plt

        wanted = {
            "12DDualArm": ["mean_pos_err", "mean_ori_err_deg", "center_err", "span_err"],
            "6DSurfNormal": ["surface_z_abs_err", "tool_axis_chord_err", "tool_axis_angle_deg"],
            "3DPlanarArmLine": ["ee_line_y_abs_err"],
        }
        label_map = {
            "mean_pos_err": "mean pos",
            "mean_ori_err_deg": "mean ori deg",
            "center_err": "center",
            "span_err": "span",
            "surface_z_abs_err": "surface z",
            "tool_axis_chord_err": "axis chord",
            "tool_axis_angle_deg": "axis deg",
            "ee_xy_ellipse_err": "EE xy",
            "ee_z_abs_err": "EE z",
            "ee_line_y_abs_err": "EE y-line",
        }
        fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.3), constrained_layout=True)
        for ax, (dataset_label, metrics) in zip(axes, wanted.items()):
            x = np.arange(len(metrics))
            for offset, group, color in (
                (-0.18, "top5_high_error", "#D55E00"),
                (0.18, "remaining_95pct", "#0072B2"),
            ):
                vals = [
                    next(
                        r["median"]
                        for r in rows
                        if r["Dataset"] == dataset_label and r["group"] == group and r["metric"] == metric
                    )
                    for metric in metrics
                ]
                ax.bar(x + offset, vals, width=0.34, color=color, label="top 5%" if group == "top5_high_error" else "remaining")
            ax.set_title(dataset_label)
            ax.set_xticks(x)
            ax.set_xticklabels([label_map[m] for m in metrics], rotation=30, ha="right")
            ax.grid(True, axis="y", alpha=0.25)
        axes[0].set_ylabel("Median component error")
        axes[-1].legend(frameon=False, loc="upper right")
        out_png = os.path.join(outdir, "high_projection_error_cause_components.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"[saved] {out_png}")
    except Exception as exc:
        print(f"[warn] failed to plot component summary: {exc}")


if __name__ == "__main__":
    main()
