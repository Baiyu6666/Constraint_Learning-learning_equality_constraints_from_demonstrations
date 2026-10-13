from __future__ import annotations

import argparse
import glob
import os
from types import SimpleNamespace

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluation.evaluator import (
    eval_bounds_from_train,
    resolve_eval_cfg,
    resolve_gt_grid,
    sample_eval_seed_points,
)


def _find_ckpt(bench: str, method: str, dataset: str, seed: str | None) -> str:
    method_dir = os.path.join(bench, method)
    if seed:
        pats = [
            os.path.join(method_dir, dataset, f"{dataset}_{method}_seed{seed}_model.pt"),
            os.path.join(method_dir, f"{dataset}_{method}_seed{seed}_model.pt"),
        ]
    else:
        pats = [
            os.path.join(method_dir, dataset, f"{dataset}_{method}_seed*_model.pt"),
            os.path.join(method_dir, f"{dataset}_{method}_model.pt"),
        ]
    matches: list[str] = []
    for pat in pats:
        matches.extend(glob.glob(pat))
    matches = sorted(set(matches))
    if not matches:
        raise FileNotFoundError(f"checkpoint not found for {method}/{dataset}, seed={seed or '*'} under {bench}")
    return matches[-1]


def main() -> None:
    p = argparse.ArgumentParser(description="Plot 2D ellipse eval query sampling distribution.")
    p.add_argument("--bench", default="outputs/bench/paper_mix_2d_3d6d_traj_vs_nontraj_7seed")
    p.add_argument("--method", default="oncl")
    p.add_argument("--dataset", default="2d_ellipse")
    p.add_argument("--seed", default="")
    p.add_argument("--outdir", default="outputs/analysis/query_distribution")
    p.add_argument("--n-queries", type=int, default=4096)
    p.add_argument("--max-near-plot", type=int, default=1200)
    p.add_argument("--max-broad-plot", type=int, default=800)
    p.add_argument("--max-demo-plot", type=int, default=512)
    args = p.parse_args()

    bench = os.path.abspath(args.bench)
    outdir = os.path.abspath(args.outdir)
    os.makedirs(outdir, exist_ok=True)

    ckpt_path = _find_ckpt(bench, str(args.method), str(args.dataset), str(args.seed).strip() or None)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = SimpleNamespace(**dict(ckpt.get("cfg", {}) or {}))
    x_train = np.asarray(ckpt["x_train"], dtype=np.float32)
    eval_cfg = resolve_eval_cfg(cfg, method_key=str(args.method), dataset_name=str(args.dataset))
    eval_cfg.eval_proj_n_points = int(args.n_queries)
    grid = resolve_gt_grid(str(args.dataset), eval_cfg, x_train=x_train).astype(np.float32)
    x_eval = sample_eval_seed_points(x_train, eval_cfg, dataset_name=str(args.dataset))

    near_ratio = float(np.clip(getattr(eval_cfg, "eval_chamfer_near_ratio", 0.75), 0.0, 1.0))
    n_near = int(round(len(x_eval) * near_ratio))
    near = x_eval[:n_near]
    broad = x_eval[n_near:]
    mins, maxs = eval_bounds_from_train(x_train, eval_cfg, dataset_name=str(args.dataset))

    rng = np.random.default_rng(20260713)

    def _sample(arr: np.ndarray, n: int) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if n > 0 and len(arr) > n:
            return arr[rng.choice(len(arr), size=n, replace=False)]
        return arr

    near_plot = _sample(near, int(args.max_near_plot))
    broad_plot = _sample(broad, int(args.max_broad_plot))
    demo_plot = _sample(x_train, int(args.max_demo_plot))

    with plt.rc_context(
        {
            "font.size": 8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
        }
    ):
        fig, ax = plt.subplots(figsize=(4.2, 3.0), constrained_layout=True)
        ax.plot(grid[:, 0], grid[:, 1], color="#111111", linewidth=1.4, label="true constraint", zorder=4)
        ax.scatter(broad_plot[:, 0], broad_plot[:, 1], s=8, alpha=0.30, color="#E45756", edgecolors="none", label=f"broad query ({len(broad)})")
        ax.scatter(near_plot[:, 0], near_plot[:, 1], s=8, alpha=0.28, color="#4C78A8", edgecolors="none", label=f"near query ({len(near)})")
        ax.scatter(demo_plot[:, 0], demo_plot[:, 1], s=10, alpha=0.75, color="#111111", marker="x", linewidths=0.6, label=f"demo ({len(x_train)})")
        ax.set_xlim(float(mins[0]), float(maxs[0]))
        ax.set_ylim(float(mins[1]), float(maxs[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("$x_1$")
        ax.set_ylabel("$x_2$")
        ax.set_title("2DEllipse query distribution")
        ax.grid(True, alpha=0.25, linewidth=0.6)
        ax.legend(frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5), borderaxespad=0.0)
        out_png = os.path.join(outdir, f"{args.dataset}_{args.method}_query_distribution.png")
        fig.savefig(out_png, dpi=300, bbox_inches="tight")
        plt.close(fig)

    out_npz = os.path.join(outdir, f"{args.dataset}_{args.method}_query_distribution_points.npz")
    np.savez_compressed(
        out_npz,
        ckpt_path=ckpt_path,
        x_train=x_train.astype(np.float32),
        grid=grid.astype(np.float32),
        near=near.astype(np.float32),
        broad=broad.astype(np.float32),
        mins=mins.astype(np.float32),
        maxs=maxs.astype(np.float32),
        near_ratio=np.asarray([near_ratio], dtype=np.float32),
    )
    print(f"[info] ckpt={ckpt_path}")
    print(f"[info] n_train={len(x_train)} n_grid={len(grid)} near={len(near)} broad={len(broad)} total={len(x_eval)}")
    print(f"[saved] {out_png}")
    print(f"[saved] {out_npz}")


if __name__ == "__main__":
    main()
