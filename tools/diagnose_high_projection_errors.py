from __future__ import annotations

import argparse
import csv
import os
from typing import Any

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
    ("3d_planar_arm_line_n3_traj", "3DPlanarArmLine"),
]


def _load_model(ckpt_path: str) -> tuple[torch.nn.Module, np.ndarray, Any]:
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


def _project(
    model: torch.nn.Module,
    cfg: Any,
    x_eval: np.ndarray,
    *,
    steps: int,
    alpha: float,
    min_steps: int,
    f_abs_stop: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    return project_points_with_steps_numpy(
        model,
        x_eval.astype(np.float32),
        device=str(cfg.device),
        proj_steps=int(steps),
        proj_alpha=float(alpha),
        proj_min_steps=int(min_steps),
        f_abs_stop=f_abs_stop,
    )


def _projection_residual(model: torch.nn.Module, cfg: Any, x: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        f = model(torch.from_numpy(x.astype(np.float32)).to(str(cfg.device)))
        if f.dim() == 1:
            f = f.unsqueeze(1)
        h = torch.linalg.norm(f, dim=1)
    return h.detach().cpu().numpy().astype(np.float32)


def _true_error(dataset: str, cfg: Any, x_train: np.ndarray, proj: np.ndarray) -> np.ndarray:
    post_fn = _postprocess_fn(dataset)
    embed_fn = _embed_fn(dataset)
    if post_fn is not None:
        proj = post_fn(proj).astype(np.float32)
    if dataset in (
        "6d_surf_normal",
        "6d_surf_normal_traj",
    ):
        if embed_fn is None:
            raise RuntimeError(dataset)
        proj_metric = embed_fn(proj)
        return _workspace_pose_analytic_dist_embed(proj_metric[:, :6].astype(np.float32), cfg)
    if dataset in ("12d_dual_arm", "12d_dual_arm_traj"):
        return _dual_arm_pose_analytic_dist_embed(proj[:, :12].astype(np.float32), cfg)
    grid = resolve_gt_grid(dataset, cfg, x_train=x_train)
    if post_fn is not None:
        grid = post_fn(grid).astype(np.float32)
    if embed_fn is not None:
        return _nn_dist(embed_fn(proj), embed_fn(grid))
    return _nn_dist(proj, grid)


def _normalized_demo_distance(dataset: str, x_train: np.ndarray, x_eval: np.ndarray) -> np.ndarray:
    post_fn = _postprocess_fn(dataset)
    embed_fn = _embed_fn(dataset)
    train = post_fn(x_train) if post_fn is not None else x_train
    query = post_fn(x_eval) if post_fn is not None else x_eval
    if embed_fn is not None:
        train_metric = embed_fn(train)
        query_metric = embed_fn(query)
    else:
        train_metric = train
        query_metric = query
    demo_span = (np.max(train_metric, axis=0) - np.min(train_metric, axis=0)).astype(np.float32)
    span_floor = max(0.10 * float(np.max(demo_span)), 1e-6)
    demo_span = np.maximum(demo_span, span_floor).astype(np.float32)
    return _nn_dist(query_metric.astype(np.float32) / demo_span, train_metric.astype(np.float32) / demo_span)


def _summ(vals: np.ndarray) -> tuple[float, float, float]:
    vals = np.asarray(vals, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    return float(np.mean(vals)), float(np.median(vals)), float(np.percentile(vals, 95))


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose high projection-error points.")
    p.add_argument("--bench", default="outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed")
    p.add_argument("--outdir", default="outputs/analysis/high_projection_error_diagnostics")
    p.add_argument("--n-queries", type=int, default=4096)
    p.add_argument("--gt-n-grid", type=int, default=8128)
    p.add_argument("--top-frac", type=float, default=0.05)
    args = p.parse_args()

    bench = os.path.abspath(args.bench)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)
    rows: list[dict[str, object]] = []

    for dataset, label in DATASETS:
        model, x_train, cfg = _load_model(os.path.join(bench, "oncl", f"{dataset}_oncl_model.pt"))
        eval_cfg = resolve_eval_cfg(cfg, method_key="oncl", dataset_name=dataset)
        eval_cfg.eval_proj_n_points = int(args.n_queries)
        eval_cfg.eval_gt_n_grid = int(args.gt_n_grid)
        eval_cfg.device = str(cfg.device)
        post_fn = _postprocess_fn(dataset)
        x_eval = sample_eval_seed_points(x_train, eval_cfg, dataset_name=dataset)
        if post_fn is not None:
            x_eval = post_fn(x_eval).astype(np.float32)
        near_ratio = float(np.clip(getattr(eval_cfg, "eval_chamfer_near_ratio", 0.75), 0.0, 1.0))
        n_near = int(round(len(x_eval) * near_ratio))
        query_type = np.zeros((len(x_eval),), dtype=np.int32)
        query_type[:n_near] = 1
        demo_dist = _normalized_demo_distance(dataset, x_train, x_eval)

        eps_stop = float(compute_eps_stop(model, x_train, eval_cfg))
        proj_cfg = getattr(eval_cfg, "projector", {}) or {}
        default_steps = int(proj_cfg.get("steps", 80))
        default_alpha = float(proj_cfg.get("alpha", 0.3))
        default_min_steps = int(proj_cfg.get("min_steps", 0))
        variants = [
            ("default", default_steps, default_alpha, default_min_steps, eps_stop),
            ("more_steps_stop", default_steps * 5, default_alpha, default_min_steps, eps_stop),
            ("more_steps_no_stop", default_steps * 5, default_alpha, default_min_steps, None),
            ("lower_alpha_no_stop", default_steps * 5, default_alpha / 3.0, default_min_steps, None),
        ]

        default_proj, default_step_arr = _project(
            model,
            eval_cfg,
            x_eval,
            steps=default_steps,
            alpha=default_alpha,
            min_steps=default_min_steps,
            f_abs_stop=eps_stop,
        )
        default_proj = _clip_points_to_eval_bounds(default_proj, x_train, eval_cfg, dataset_name=dataset)
        default_err = _true_error(dataset, eval_cfg, x_train, default_proj)
        cutoff = float(np.percentile(default_err, 100.0 * (1.0 - float(args.top_frac))))
        top_mask = default_err >= cutoff

        for variant, steps, alpha, min_steps, stop in variants:
            proj_raw, step_arr = _project(
                model,
                eval_cfg,
                x_eval,
                steps=steps,
                alpha=alpha,
                min_steps=min_steps,
                f_abs_stop=stop,
            )
            proj = _clip_points_to_eval_bounds(proj_raw, x_train, eval_cfg, dataset_name=dataset)
            clip_delta = np.linalg.norm((proj - proj_raw).astype(np.float32), axis=1)
            raw_err = _true_error(dataset, eval_cfg, x_train, proj_raw)
            err = _true_error(dataset, eval_cfg, x_train, proj)
            resid = _projection_residual(model, eval_cfg, proj)
            resid_raw = _projection_residual(model, eval_cfg, proj_raw)

            for subset, mask in (("all", np.ones_like(top_mask, dtype=bool)), ("top5_default_error", top_mask)):
                e_mean, e_med, e_p95 = _summ(err[mask])
                re_mean, re_med, re_p95 = _summ(raw_err[mask])
                r_mean, r_med, r_p95 = _summ(resid[mask])
                rr_mean, rr_med, rr_p95 = _summ(resid_raw[mask])
                s_mean, s_med, s_p95 = _summ(step_arr[mask])
                c_mean, c_med, c_p95 = _summ(clip_delta[mask])
                d_mean, d_med, d_p95 = _summ(demo_dist[mask])
                rows.append(
                    {
                        "dataset": dataset,
                        "Dataset": label,
                        "subset": subset,
                        "variant": variant,
                        "steps": steps,
                        "alpha": alpha,
                        "stop_enabled": stop is not None,
                        "n": int(np.sum(mask)),
                        "err_mean": e_mean,
                        "err_median": e_med,
                        "err_p95": e_p95,
                        "raw_err_mean": re_mean,
                        "raw_err_median": re_med,
                        "raw_err_p95": re_p95,
                        "clip_err_delta_mean": e_mean - re_mean,
                        "clip_err_delta_median": e_med - re_med,
                        "clip_err_delta_p95": e_p95 - re_p95,
                        "resid_mean": r_mean,
                        "resid_median": r_med,
                        "resid_p95": r_p95,
                        "raw_resid_mean": rr_mean,
                        "raw_resid_median": rr_med,
                        "raw_resid_p95": rr_p95,
                        "proj_steps_mean": s_mean,
                        "proj_steps_median": s_med,
                        "proj_steps_p95": s_p95,
                        "frac_at_step_limit": float(np.mean(step_arr[mask] >= steps)),
                        "clip_delta_mean": c_mean,
                        "clip_delta_median": c_med,
                        "clip_delta_p95": c_p95,
                        "frac_clipped": float(np.mean(clip_delta[mask] > 1e-6)),
                        "demo_dist_mean": d_mean,
                        "demo_dist_median": d_med,
                        "demo_dist_p95": d_p95,
                        "broad_frac": float(np.mean(query_type[mask] == 0)),
                        "default_top5_cutoff": cutoff,
                    }
                )
            if variant == "default":
                point_path = os.path.join(outdir, f"{dataset}_default_point_diagnostics.npz")
                np.savez_compressed(
                    point_path,
                    x_eval=x_eval.astype(np.float32),
                    raw_err=raw_err.astype(np.float32),
                    err=err.astype(np.float32),
                    clip_delta=clip_delta.astype(np.float32),
                    raw_resid=resid_raw.astype(np.float32),
                    resid=resid.astype(np.float32),
                    proj_raw=proj_raw.astype(np.float32),
                    proj=proj.astype(np.float32),
                    proj_steps=step_arr.astype(np.float32),
                    demo_dist=demo_dist.astype(np.float32),
                    query_type=query_type.astype(np.int32),
                    top_default_error=top_mask.astype(bool),
                    default_top5_cutoff=np.asarray([cutoff], dtype=np.float32),
                )
        print(f"[done] {label}: top5 cutoff={cutoff:.6g}, eps_stop={eps_stop:.6g}")

    out_csv = os.path.join(outdir, "high_projection_error_diagnostics.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print(f"[saved] {out_csv}")


if __name__ == "__main__":
    main()
