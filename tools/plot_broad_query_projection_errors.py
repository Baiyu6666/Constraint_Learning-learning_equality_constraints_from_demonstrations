from __future__ import annotations

import argparse
import csv
import os
from types import SimpleNamespace
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
    ("3d_torus_surface_traj", "3DTorus"),
]


def _cfg_from_checkpoint(ckpt: dict[str, Any]) -> SimpleNamespace:
    cfg = dict(ckpt.get("cfg", {}) or {})
    cfg["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return SimpleNamespace(**cfg)


def _load_oncl_model(ckpt_path: str) -> tuple[torch.nn.Module, np.ndarray, SimpleNamespace]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = _cfg_from_checkpoint(ckpt)
    model = MLP(
        in_dim=int(ckpt["in_dim"]),
        hidden=int(ckpt["hidden"]),
        depth=int(ckpt["depth"]),
        out_dim=int(ckpt["constraint_dim"]),
    ).to(str(cfg.device))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    x_train = np.asarray(ckpt["x_train"], dtype=np.float32)
    return model, x_train, cfg


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


def _projection_distances(
    *,
    dataset: str,
    model: torch.nn.Module,
    x_train: np.ndarray,
    cfg: Any,
    n_queries: int,
    gt_n_grid: int,
    query_mode: str,
) -> np.ndarray:
    eval_cfg = resolve_eval_cfg(cfg, method_key="oncl", dataset_name=dataset)
    if query_mode == "broad":
        eval_cfg.eval_chamfer_near_ratio = 0.0
    elif query_mode == "near":
        eval_cfg.eval_chamfer_near_ratio = 1.0
    else:
        raise ValueError(f"unknown query_mode: {query_mode}")
    eval_cfg.eval_proj_n_points = int(n_queries)
    eval_cfg.eval_gt_n_grid = int(gt_n_grid)
    eval_cfg.device = str(cfg.device)

    post_fn = _postprocess_fn(dataset)
    embed_fn = _embed_fn(dataset)

    x_eval = sample_eval_seed_points(x_train, eval_cfg, dataset_name=dataset)
    if post_fn is not None:
        x_eval = post_fn(x_eval).astype(np.float32)

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
    if dataset in (
        "6d_surf_normal",
        "6d_surf_normal_traj",
    ):
        if embed_fn is None:
            raise RuntimeError(f"missing embedding function for {dataset}")
        proj_metric = embed_fn(proj)
        return _workspace_pose_analytic_dist_embed(proj_metric[:, :6].astype(np.float32), eval_cfg)
    if dataset in ("12d_dual_arm", "12d_dual_arm_traj"):
        return _dual_arm_pose_analytic_dist_embed(proj[:, :12].astype(np.float32), eval_cfg)

    grid = resolve_gt_grid(dataset, eval_cfg, x_train=x_train)
    if post_fn is not None:
        grid = post_fn(grid).astype(np.float32)
    if embed_fn is not None:
        return _nn_dist(embed_fn(proj), embed_fn(grid))
    return _nn_dist(proj, grid)


def _summary(vals: np.ndarray) -> dict[str, float]:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return {
        "n": float(vals.size),
        "mean": float(np.mean(vals)),
        "median": float(np.median(vals)),
        "p75": float(np.percentile(vals, 75)),
        "p90": float(np.percentile(vals, 90)),
        "p95": float(np.percentile(vals, 95)),
        "max": float(np.max(vals)),
    }


def _plot_distributions(out_png: str, values: dict[str, np.ndarray]) -> None:
    colors = {
        "12DDualArm": "#4C78A8",
        "6DSurfNormal": "#E45756",
        "3DTorus": "#54A24B",
    }
    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    ):
        fig, axes = plt.subplots(1, 3, figsize=(6.7, 1.9), constrained_layout=True)
        for ax, label in zip(axes, values):
            vals = np.asarray(values[label], dtype=np.float64)
            vals = vals[np.isfinite(vals)]
            ax.hist(vals, bins=40, color=colors.get(label, "#4C78A8"), alpha=0.82, edgecolor="white", linewidth=0.35)
            ax.set_title(label)
            ax.set_xlabel(r"Distance to true manifold $\varepsilon_l$")
            ax.grid(True, axis="y", alpha=0.25, linewidth=0.6)
        axes[0].set_ylabel("Count")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot ONCL broad-query projection error distributions.")
    p.add_argument("--bench", default="outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed")
    p.add_argument("--outdir", default="outputs/analysis/broad_query_projection_errors")
    p.add_argument("--n-queries", type=int, default=4096)
    p.add_argument("--gt-n-grid", type=int, default=8128)
    p.add_argument("--query-mode", choices=["broad", "near"], default="broad")
    args = p.parse_args()

    bench = os.path.abspath(args.bench)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    all_values: dict[str, np.ndarray] = {}
    summary_rows: list[dict[str, object]] = []
    for dataset, label in DATASETS:
        ckpt_path = os.path.join(bench, "oncl", f"{dataset}_oncl_model.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(ckpt_path)
        model, x_train, cfg = _load_oncl_model(ckpt_path)
        vals = _projection_distances(
            dataset=dataset,
            model=model,
            x_train=x_train,
            cfg=cfg,
            n_queries=int(args.n_queries),
            gt_n_grid=int(args.gt_n_grid),
            query_mode=str(args.query_mode),
        )
        all_values[label] = vals
        s = _summary(vals)
        s_row: dict[str, object] = {"dataset": dataset, "Dataset": label}
        s_row.update(s)
        summary_rows.append(s_row)
        np.save(os.path.join(outdir, f"{dataset}_{args.query_mode}_query_projection_distances.npy"), vals.astype(np.float32))
        print(f"[done] {label}: mean={s['mean']:.6g}, median={s['median']:.6g}, p95={s['p95']:.6g}, n={int(s['n'])}")

    summary_csv = os.path.join(outdir, f"{args.query_mode}_query_projection_error_summary.csv")
    with open(summary_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = ["dataset", "Dataset", "n", "mean", "median", "p75", "p90", "p95", "max"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in summary_rows:
            w.writerow(row)

    out_png = os.path.join(outdir, f"{args.query_mode}_query_projection_error_distributions.png")
    _plot_distributions(out_png, all_values)
    print(f"[saved] {summary_csv}")
    print(f"[saved] {out_png}")


if __name__ == "__main__":
    main()
