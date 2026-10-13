from __future__ import annotations

import argparse
import csv
import os
import time
from dataclasses import asdict
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from datasets.constraint_datasets import set_seed
from evaluation.eval_runner import run_eval_metrics
from experiments.config_loader import apply_overrides, load_layered_config
from experiments.dataset_resolve import resolve_dataset
from experiments.unified_experiment import (
    _apply_planner_subcfg,
    _apply_projector_subcfg,
    _attach_extra_cfg_fields,
    _build_cfg_from_mapping_strict,
    _is_arm_dataset,
    _is_workspace_pose_dataset,
    _ur5_use_pybullet_n6,
    _workspace_embed_for_eval,
    _wrap_np_pi,
    _wrap_workspace_pose_rpy_np,
)
from methods import oncl as ve
from methods.models.mlp import MLP
from methods.utils.projection import project_points_with_steps_numpy


def _projector_value(cfg: Any, key: str, default: float | int) -> float | int:
    proj = getattr(cfg, "projector", {}) or {}
    if isinstance(proj, dict) and key in proj:
        return proj[key]
    return default


def _training_residual(model: torch.nn.Module, x_train: np.ndarray, device: str) -> float:
    model.eval()
    with torch.no_grad():
        f = model(torch.from_numpy(x_train.astype(np.float32)).to(device))
        if f.dim() == 1:
            f = f.unsqueeze(1)
        h = torch.linalg.norm(f, dim=1)
        return float(torch.mean(0.5 * h * h).detach().cpu())


def _resolve_codim(cfg: Any, dataset: str, x_train: np.ndarray, data_dim: int, true_codim: int) -> int:
    if str(cfg.constraint_dim).lower() != "auto":
        codim = max(1, int(cfg.constraint_dim))
        if data_dim == 2 and codim != 1:
            codim = 1
        return codim
    try:
        est = ve.estimate_codim_local_pca(
            x_train,
            periodic_joint=_is_arm_dataset(dataset),
            sample_ratio=float(cfg.codim_auto_sample_ratio),
            k_neighbors=int(cfg.codim_auto_k_neighbors),
            const_axis_std_ratio=float(cfg.codim_auto_const_axis_std_ratio),
            seed=int(cfg.seed) + 1000,
        )
        codim = int(est["estimated_codim"])
        if codim != true_codim:
            codim = int(true_codim)
        return codim
    except Exception:
        return int(true_codim)


def _make_eval_functions(cfg: Any, dataset: str):
    def project_fn(model: torch.nn.Module, x0: np.ndarray, eps_stop: float) -> tuple[np.ndarray, np.ndarray]:
        return project_points_with_steps_numpy(
            model,
            x0.astype(np.float32),
            device=str(cfg.device),
            proj_steps=int(_projector_value(cfg, "steps", 80)),
            proj_alpha=float(_projector_value(cfg, "alpha", 0.3)),
            proj_min_steps=int(_projector_value(cfg, "min_steps", 0)),
            f_abs_stop=float(eps_stop),
        )

    post_fn = None
    if _is_arm_dataset(dataset):
        post_fn = _wrap_np_pi
    elif _is_workspace_pose_dataset(dataset):
        post_fn = _wrap_workspace_pose_rpy_np

    use_pybullet_n6 = _ur5_use_pybullet_n6(str(dataset))
    embed_fn = (
        lambda q, _name=dataset: _workspace_embed_for_eval(  # noqa: E731
            _name,
            q,
            ur5_use_pybullet_n6=use_pybullet_n6,
        )
    ) if (_is_arm_dataset(dataset) or _is_workspace_pose_dataset(dataset)) else None
    return project_fn, embed_fn, post_fn


def _evaluate(
    *,
    cfg: Any,
    dataset: str,
    model: torch.nn.Module,
    x_train: np.ndarray,
) -> dict[str, float]:
    project_fn, embed_fn, post_fn = _make_eval_functions(cfg, dataset)
    metrics, _, _ = run_eval_metrics(
        cfg=cfg,
        method_key="oncl",
        dataset_name=dataset,
        model=model,
        x_train=x_train,
        project_fn=project_fn,
        embed_fn=embed_fn,
        postprocess_fn=post_fn,
    )
    return metrics


def _plot(rows: list[dict[str, float]], out_png: str) -> None:
    epochs = np.asarray([r["epoch"] for r in rows], dtype=np.float64)
    train_resid = np.asarray([r["training_residual"] for r in rows], dtype=np.float64)
    eval_dist = np.asarray([r["eval_distance_error"] for r in rows], dtype=np.float64)

    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    ):
        fig, ax1 = plt.subplots(figsize=(4.3, 2.35))
        ax2 = ax1.twinx()

        l1 = ax1.plot(epochs, train_resid, color="#4C78A8", marker="o", ms=3, lw=1.5, label="training residual")
        l2 = ax2.plot(epochs, eval_dist, color="#E45756", marker="s", ms=3, lw=1.5, label="eval distance error")

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel(r"Training residual $\frac{1}{2}\|L_\theta(x_{\mathrm{train}})\|^2$", color="#4C78A8")
        ax2.set_ylabel(r"Eval distance error $\varepsilon_l$", color="#E45756")
        ax1.tick_params(axis="y", labelcolor="#4C78A8")
        ax2.tick_params(axis="y", labelcolor="#E45756")
        ax1.set_yscale("log")
        ax2.set_yscale("log")
        ax1.grid(True, axis="both", alpha=0.25, linewidth=0.6)

        lines = l1 + l2
        labels = [line.get_label() for line in lines]
        ax1.legend(lines, labels, frameon=False, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(description="Plot ONCL training residual and evaluation distance over training.")
    p.add_argument("--dataset", default="12d_dual_arm_traj")
    p.add_argument("--config-root", default="configs")
    p.add_argument("--outdir", default="outputs/analysis/oncl_train_eval_curve")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-proj-n-points", type=int, default=1024)
    p.add_argument("--eval-gt-n-grid", type=int, default=4096)
    p.add_argument("--override", action="append", default=[])
    args = p.parse_args()

    cfg_map, _ = load_layered_config(args.config_root, "oncl", args.dataset)
    cfg_map = apply_overrides(cfg_map, list(args.override))
    cfg = _build_cfg_from_mapping_strict(ve.DemoCfg, cfg_map)
    cfg = _attach_extra_cfg_fields(cfg, cfg_map)
    _apply_projector_subcfg(cfg)
    _apply_planner_subcfg(cfg)
    if args.seed is not None:
        cfg.seed = int(args.seed)
    if args.epochs is not None:
        cfg.epochs = int(args.epochs)
    cfg.eval_proj_n_points = int(args.eval_proj_n_points)
    cfg.eval_gt_n_grid = int(args.eval_gt_n_grid)
    cfg.train_method_name = "oncl_train_eval_curve"
    cfg.device = ve._choose_device(str(cfg.device))
    if cfg.device == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    set_seed(int(cfg.seed))

    ds = resolve_dataset(
        args.dataset,
        cfg,
        optimize_ur5_train_only=True,
        ur5_backend=("pybullet" if _ur5_use_pybullet_n6(str(args.dataset)) else "analytic"),
    )
    x_train = np.asarray(ds["x_train"], dtype=np.float32)
    data_dim = int(ds.get("data_dim", int(x_train.shape[1])))
    true_codim = int(ds.get("true_codim", 1))
    codim = _resolve_codim(cfg, args.dataset, x_train, data_dim, true_codim)

    model = MLP(
        in_dim=x_train.shape[1],
        hidden=int(cfg.hidden),
        depth=int(cfg.depth),
        out_dim=max(1, int(codim)),
    ).to(str(cfg.device))
    opt = torch.optim.Adam(model.parameters(), lr=float(cfg.lr))
    scheduler = None
    if int(cfg.lr_decay_step) > 0 and float(cfg.lr_decay_gamma) < 1.0:
        scheduler = torch.optim.lr_scheduler.StepLR(
            opt,
            step_size=int(cfg.lr_decay_step),
            gamma=float(cfg.lr_decay_gamma),
        )

    x_t = torch.from_numpy(x_train)
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_t),
        batch_size=int(cfg.batch_size),
        shuffle=True,
        drop_last=False,
        pin_memory=(str(cfg.device) == "cuda"),
        num_workers=0,
    )

    mins = x_train.min(axis=0)
    maxs = x_train.max(axis=0)
    span_raw = maxs - mins
    ref_span = max(float(np.max(span_raw)), 1e-6)
    min_axis_span = max(0.0, float(cfg.train_min_axis_span_ratio)) * ref_span
    span = np.maximum(span_raw, min_axis_span)
    scale = max(1.0 + float(cfg.train_sample_pad_ratio), 1e-6)
    center = 0.5 * (mins + maxs)
    half = 0.5 * span * scale
    mins_t = torch.from_numpy((center - half).astype(np.float32)).to(str(cfg.device))
    maxs_t = torch.from_numpy((center + half).astype(np.float32)).to(str(cfg.device))
    x_train_dev = x_t.to(str(cfg.device))
    span_t = (maxs_t - mins_t).clamp_min(1e-6)
    near_ratio = float(min(max(cfg.oncl_near_ratio, 0.0), 1.0))
    near_std_ratio = float(max(cfg.oncl_near_std_ratio, 1e-5))

    rows: list[dict[str, float]] = []
    os.makedirs(args.outdir, exist_ok=True)
    t0 = time.perf_counter()
    eval_epochs = set(range(0, int(cfg.epochs) + 1, max(1, int(args.eval_every))))
    eval_epochs.add(int(cfg.epochs))

    for ep in range(int(cfg.epochs) + 1):
        if ep in eval_epochs:
            train_resid = _training_residual(model, x_train, str(cfg.device))
            metrics = _evaluate(cfg=cfg, dataset=args.dataset, model=model, x_train=x_train)
            row = {
                "epoch": float(ep),
                "training_residual": float(train_resid),
                "eval_distance_error": float(metrics["proj_manifold_dist"]),
                "eval_proj_true_dist": float(metrics["proj_true_dist"]),
                "eval_proj_residual": float(metrics["proj_v_residual"]),
                "eval_steps": float(metrics["proj_steps"]),
            }
            rows.append(row)
            print(
                f"[eval] epoch={ep:4d} "
                f"train_resid={row['training_residual']:.6g} "
                f"eval_dist={row['eval_distance_error']:.6g} "
                f"elapsed={time.perf_counter() - t0:.1f}s"
            )
        if ep >= int(cfg.epochs):
            break

        model.train()
        for (xb,) in dl:
            xb = xb.to(str(cfg.device))
            opt.zero_grad(set_to_none=True)
            f_on = model(xb)
            loss_on = (f_on ** 2).mean()
            xr = ve._sample_xr_batch(
                xb,
                x_train_dev,
                mins_t,
                maxs_t,
                span_t,
                near_ratio,
                near_std_ratio,
            )
            loss_eik = ve._oncl_multi_constraint(model, xr, lam_ortho=float(cfg.lam_oncl_ortho))
            loss = loss_on + float(cfg.lam_oncl) * loss_eik
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        if scheduler is not None:
            scheduler.step()

    csv_path = os.path.join(args.outdir, f"{args.dataset}_oncl_train_eval_curve.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "epoch",
            "training_residual",
            "eval_distance_error",
            "eval_proj_true_dist",
            "eval_proj_residual",
            "eval_steps",
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    png_path = os.path.join(args.outdir, f"{args.dataset}_oncl_train_eval_curve.png")
    _plot(rows, png_path)

    cfg_path = os.path.join(args.outdir, f"{args.dataset}_oncl_train_eval_curve_config.json")
    with open(cfg_path, "w", encoding="utf-8") as f:
        import json

        json.dump(
            {
                "dataset": args.dataset,
                "codim": int(codim),
                "cfg": asdict(cfg),
                "eval_every": int(args.eval_every),
                "eval_proj_n_points": int(args.eval_proj_n_points),
                "eval_gt_n_grid": int(args.eval_gt_n_grid),
            },
            f,
            indent=2,
        )
    print(f"[saved] {csv_path}")
    print(f"[saved] {png_path}")
    print(f"[saved] {cfg_path}")


if __name__ == "__main__":
    main()
