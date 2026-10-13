from __future__ import annotations

import argparse
import csv
import json
import os
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.evaluator import (
    _clip_points_to_eval_bounds,
    _dual_arm_pose_analytic_dist_embed,
    _nn_dist,
    _workspace_pose_analytic_dist_embed,
    compute_eps_stop,
    resolve_eval_cfg,
    resolve_gt_grid,
    sample_eval_seed_points,
)
from methods.models.mlp import MLP
from methods.utils.projection import project_points_with_steps_numpy
from planning.kinematics import (
    is_arm_dataset,
    is_workspace_pose_dataset,
    workspace_embed_for_eval,
    wrap_np_pi,
    wrap_workspace_pose_rpy_np,
)


DATASETS = [
    ("12d_dual_arm_traj", "12DDualArm"),
    ("6d_surf_normal_traj", "6DSurfNormal"),
    ("3d_vz_2d_ellipse_traj", "3DTwistEllipse"),
]


def _load_oncl_model(ckpt_path: str) -> tuple[torch.nn.Module, np.ndarray, Any]:
    from types import SimpleNamespace

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**dict(ckpt.get("cfg", {}) or {}))
    cfg.device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(
        in_dim=int(ckpt["in_dim"]),
        hidden=int(ckpt["hidden"]),
        depth=int(ckpt["depth"]),
        out_dim=int(ckpt["constraint_dim"]),
    ).to(str(cfg.device))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, np.asarray(ckpt["x_train"], dtype=np.float32), cfg


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


def _project(model: torch.nn.Module, cfg: Any, x_eval: np.ndarray, eps_stop: float) -> tuple[np.ndarray, np.ndarray]:
    proj_cfg = getattr(cfg, "projector", {}) or {}
    return project_points_with_steps_numpy(
        model,
        x_eval.astype(np.float32),
        device=str(cfg.device),
        proj_steps=int(proj_cfg.get("steps", 80)),
        proj_alpha=float(proj_cfg.get("alpha", 0.3)),
        proj_min_steps=int(proj_cfg.get("min_steps", 0)),
        f_abs_stop=float(eps_stop),
    )


def _compute_points(
    *,
    dataset: str,
    model: torch.nn.Module,
    x_train: np.ndarray,
    cfg: Any,
    n_queries: int,
    gt_n_grid: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    eval_cfg = resolve_eval_cfg(cfg, method_key="oncl", dataset_name=dataset)
    eval_cfg.eval_proj_n_points = int(n_queries)
    eval_cfg.eval_gt_n_grid = int(gt_n_grid)
    eval_cfg.device = str(cfg.device)

    post_fn = _postprocess_fn(dataset)
    embed_fn = _embed_fn(dataset)
    x_eval = sample_eval_seed_points(x_train, eval_cfg, dataset_name=dataset)
    if post_fn is not None:
        x_eval = post_fn(x_eval).astype(np.float32)

    near_ratio = float(np.clip(getattr(eval_cfg, "eval_chamfer_near_ratio", 0.75), 0.0, 1.0))
    n_near = int(round(len(x_eval) * near_ratio))
    query_type = np.zeros((len(x_eval),), dtype=np.int32)
    query_type[:n_near] = 1

    if embed_fn is not None:
        query_metric = embed_fn(x_eval)
        train_metric = embed_fn(post_fn(x_train) if post_fn is not None else x_train)
    else:
        query_metric = x_eval
        train_metric = x_train
    demo_span = (np.max(train_metric, axis=0) - np.min(train_metric, axis=0)).astype(np.float32)
    span_floor = max(0.10 * float(np.max(demo_span)), 1e-6)
    demo_span = np.maximum(demo_span, span_floor).astype(np.float32)
    query_metric_norm = query_metric.astype(np.float32) / demo_span.reshape(1, -1)
    train_metric_norm = train_metric.astype(np.float32) / demo_span.reshape(1, -1)
    d_demo = _nn_dist(query_metric_norm, train_metric_norm)

    eps_stop = float(compute_eps_stop(model, x_train, eval_cfg))
    proj, _steps = _project(model, eval_cfg, x_eval, eps_stop)
    proj = _clip_points_to_eval_bounds(
        proj.astype(np.float32),
        x_train,
        eval_cfg,
        dataset_name=dataset,
    )
    if post_fn is not None:
        proj = post_fn(proj).astype(np.float32)

    mask = np.isfinite(proj).all(axis=1)
    proj = proj[mask]
    d_demo = d_demo[mask]
    query_type = query_type[mask]

    if dataset in (
        "6d_surf_normal",
        "6d_surf_normal_traj",
    ):
        if embed_fn is None:
            raise RuntimeError(f"missing embedding for {dataset}")
        proj_metric = embed_fn(proj)
        proj_error = _workspace_pose_analytic_dist_embed(proj_metric[:, :6].astype(np.float32), eval_cfg)
    elif dataset in ("12d_dual_arm", "12d_dual_arm_traj"):
        proj_error = _dual_arm_pose_analytic_dist_embed(proj[:, :12].astype(np.float32), eval_cfg)
    else:
        grid = resolve_gt_grid(dataset, eval_cfg, x_train=x_train)
        if post_fn is not None:
            grid = post_fn(grid).astype(np.float32)
        if embed_fn is not None:
            proj_error = _nn_dist(embed_fn(proj), embed_fn(grid))
        else:
            proj_error = _nn_dist(proj, grid)
    return d_demo.astype(np.float32), proj_error.astype(np.float32), query_type


def _plot(out_png: str, data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], *, max_scatter_points: int = 800) -> None:
    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.15), constrained_layout=True)
        rng = np.random.default_rng(20260713)
        for ax, (label, (d_demo, proj_error, query_type)) in zip(axes, data.items()):
            if max_scatter_points > 0 and len(d_demo) > max_scatter_points:
                idx = rng.choice(len(d_demo), size=int(max_scatter_points), replace=False)
                d_plot = d_demo[idx]
                e_plot = proj_error[idx]
                q_plot = query_type[idx]
            else:
                d_plot = d_demo
                e_plot = proj_error
                q_plot = query_type
            near = q_plot == 1
            broad = ~near
            ax.scatter(d_plot[near], e_plot[near], s=6, alpha=0.35, color="#4C78A8", edgecolors="none", label="near")
            ax.scatter(d_plot[broad], e_plot[broad], s=6, alpha=0.40, color="#E45756", edgecolors="none", label="broad")
            ax.set_title(label)
            ax.set_xlabel("Span-normalized demo distance")
            ax.grid(True, alpha=0.25, linewidth=0.6)
        axes[0].set_ylabel("Distance error after projection")
        axes[-1].legend(frameon=False, loc="upper right")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def _plot_binned_quantiles(out_png: str, data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]) -> None:
    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(6.8, 2.15), constrained_layout=True)
        for ax, (label, (d_demo, proj_error, _query_type)) in zip(axes, data.items()):
            d_demo = np.asarray(d_demo, dtype=np.float32)
            proj_error = np.asarray(proj_error, dtype=np.float32)
            edges = np.linspace(float(np.min(d_demo)), float(np.max(d_demo)), 13, dtype=np.float32)
            centers: list[float] = []
            p50: list[float] = []
            p90: list[float] = []
            p95: list[float] = []
            counts: list[int] = []
            for lo, hi in zip(edges[:-1], edges[1:]):
                mask = (d_demo >= lo) & (d_demo < hi if hi < edges[-1] else d_demo <= hi)
                if int(np.sum(mask)) < 8:
                    continue
                vals = proj_error[mask]
                centers.append(float(0.5 * (lo + hi)))
                p50.append(float(np.percentile(vals, 50)))
                p90.append(float(np.percentile(vals, 90)))
                p95.append(float(np.percentile(vals, 95)))
                counts.append(int(np.sum(mask)))
            ax.plot(centers, p50, color="#0072B2", linewidth=1.4, label="median")
            ax.plot(centers, p90, color="#E69F00", linewidth=1.2, label="p90")
            ax.plot(centers, p95, color="#D55E00", linewidth=1.2, label="p95")
            ymax = max(max(p95) if p95 else 1.0, 1e-8)
            if counts:
                count_scale = 0.12 * ymax / max(counts)
                ax.bar(centers, np.asarray(counts) * count_scale, width=(edges[1] - edges[0]) * 0.75, color="#999999", alpha=0.25, label="count")
            ax.set_title(label)
            ax.set_xlabel("Span-normalized demo distance")
            ax.grid(True, alpha=0.25, linewidth=0.6)
        axes[0].set_ylabel("Distance error after projection")
        axes[-1].legend(frameon=False, loc="upper right")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Scatter query distance to demos vs projection error.")
    p.add_argument("--bench", default="outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed")
    p.add_argument("--outdir", default="outputs/analysis/query_distance_vs_projection_error")
    p.add_argument(
        "--ckpt-map",
        default="",
        help="Optional JSON object mapping dataset id to an explicit ONCL checkpoint path.",
    )
    p.add_argument("--n-queries", type=int, default=4096)
    p.add_argument("--gt-n-grid", type=int, default=8128)
    args = p.parse_args()

    bench = os.path.abspath(args.bench)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    ckpt_map = json.loads(args.ckpt_map) if str(args.ckpt_map).strip() else {}

    data: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    rows: list[dict[str, object]] = []
    for dataset, label in DATASETS:
        ckpt_path = str(ckpt_map.get(dataset) or os.path.join(bench, "oncl", f"{dataset}_oncl_model.pt"))
        if not os.path.isabs(ckpt_path):
            ckpt_path = os.path.abspath(ckpt_path)
        model, x_train, cfg = _load_oncl_model(ckpt_path)
        d_demo, proj_error, query_type = _compute_points(
            dataset=dataset,
            model=model,
            x_train=x_train,
            cfg=cfg,
            n_queries=int(args.n_queries),
            gt_n_grid=int(args.gt_n_grid),
        )
        data[label] = (d_demo, proj_error, query_type)
        out_npz = os.path.join(outdir, f"{dataset}_query_distance_vs_projection_error.npz")
        np.savez_compressed(out_npz, d_demo=d_demo, proj_error=proj_error, query_type=query_type)
        for qt, name in ((1, "near"), (0, "broad"), (-1, "all")):
            mask = np.ones_like(query_type, dtype=bool) if qt == -1 else query_type == qt
            rows.append(
                {
                    "dataset": dataset,
                    "Dataset": label,
                    "query_type": name,
                    "n": int(np.sum(mask)),
                    "demo_dist_mean": float(np.mean(d_demo[mask])),
                    "demo_dist_median": float(np.median(d_demo[mask])),
                    "proj_error_mean": float(np.mean(proj_error[mask])),
                    "proj_error_median": float(np.median(proj_error[mask])),
                    "proj_error_p75": float(np.percentile(proj_error[mask], 75)),
                    "proj_error_p90": float(np.percentile(proj_error[mask], 90)),
                    "proj_error_p95": float(np.percentile(proj_error[mask], 95)),
                    "proj_error_p99": float(np.percentile(proj_error[mask], 99)),
                    "frac_proj_error_gt_0p1": float(np.mean(proj_error[mask] > 0.1)),
                    "frac_proj_error_gt_0p2": float(np.mean(proj_error[mask] > 0.2)),
                    "frac_proj_error_gt_0p5": float(np.mean(proj_error[mask] > 0.5)),
                }
            )
        print(f"[done] {label}: n={len(d_demo)}")

    out_png = os.path.join(outdir, "query_distance_vs_projection_error_scatter.png")
    _plot(out_png, data)
    out_quantile_png = os.path.join(outdir, "query_distance_vs_projection_error_binned_quantiles.png")
    _plot_binned_quantiles(out_quantile_png, data)
    out_csv = os.path.join(outdir, "query_distance_vs_projection_error_summary.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "dataset",
            "Dataset",
            "query_type",
            "n",
            "demo_dist_mean",
            "demo_dist_median",
            "proj_error_mean",
            "proj_error_median",
            "proj_error_p75",
            "proj_error_p90",
            "proj_error_p95",
            "proj_error_p99",
            "frac_proj_error_gt_0p1",
            "frac_proj_error_gt_0p2",
            "frac_proj_error_gt_0p5",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[saved] {out_png}")
    print(f"[saved] {out_quantile_png}")
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()
