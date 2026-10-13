from __future__ import annotations

import argparse
import csv
import os
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


METHODS = ["oncl", "dataaug", "ecomann", "vae"]
METHOD_LABELS = {
    "oncl": "ONCL",
    "dataaug": "DataAug",
    "ecomann": "ECoMaNN",
    "vae": "VAE",
}
METHOD_COLORS = {
    "oncl": "#4C78A8",
    "dataaug": "#F58518",
    "ecomann": "#54A24B",
    "vae": "#B279A2",
}

DATASETS = [
    "2d_ellipse",
    "2d_planar_arm_line_n2",
    "2d_sparse_sine",
    "3d_spatial_arm_ellip_n3_traj",
    "3d_vz_2d_ellipse_traj",
    "3d_twosphere_traj",
    "3d_torus_surface_traj",
    "3d_planar_arm_line_n3_traj",
    "6d_spatial_arm_up_n6_py_traj",
    "6d_surf_normal_traj",
    "12d_dual_arm_traj",
]
DATASET_LABELS = {
    "2d_ellipse": "2DEllipse",
    "2d_planar_arm_line_n2": "2DPlanarArmLine",
    "2d_sparse_sine": "2DSineSparse",
    "3d_spatial_arm_ellip_n3_traj": "3DArmEllipse",
    "3d_vz_2d_ellipse_traj": "3DTwistEllipse",
    "3d_twosphere_traj": "3DTwoSphere",
    "3d_torus_surface_traj": "3DTorus",
    "3d_planar_arm_line_n3_traj": "3DPlanarArmLine",
    "6d_spatial_arm_up_n6_py_traj": "6DArmUp",
    "6d_surf_normal_traj": "6DSurfNormal",
    "12d_dual_arm_traj": "12DDualArm",
}


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


def _read_metric(path: Path, metric: str) -> dict[tuple[str, str], list[float]]:
    if path.is_dir():
        path = path / "per_case_metrics.csv"
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    with path.open("r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            dataset = str(row.get("dataset", "")).strip()
            method = str(row.get("method", "")).strip()
            if dataset not in DATASETS or method not in METHODS:
                continue
            try:
                value = float(row.get(metric, ""))
            except (TypeError, ValueError):
                continue
            if np.isfinite(value):
                out[(dataset, method)].append(value)
    return out


def _box_style(box, method: str, is_old: bool) -> None:
    color = METHOD_COLORS[method]
    box.set_facecolor(color)
    box.set_alpha(0.28 if is_old else 0.72)
    box.set_edgecolor(color)
    box.set_linewidth(1.0)
    if is_old:
        box.set_hatch("//")


def _plot_dataset(ax, dataset: str, old, new, *, show_ylabel: bool) -> None:
    base_x = np.arange(len(METHODS), dtype=float)
    offset = 0.18
    width = 0.28
    rng = np.random.default_rng(0)

    for mi, method in enumerate(METHODS):
        for is_old, samples, dx in [
            (True, old.get((dataset, method), []), -offset),
            (False, new.get((dataset, method), []), offset),
        ]:
            if not samples:
                continue
            arr = np.asarray(samples, dtype=float)
            pos = base_x[mi] + dx
            bp = ax.boxplot(
                [arr],
                positions=[pos],
                widths=width,
                patch_artist=True,
                showfliers=False,
                medianprops={"color": "#111827", "linewidth": 1.0},
                whiskerprops={"color": "#4B5563", "linewidth": 0.9},
                capprops={"color": "#4B5563", "linewidth": 0.9},
            )
            _box_style(bp["boxes"][0], method, is_old)
            jitter = rng.normal(loc=0.0, scale=0.018, size=arr.size)
            ax.scatter(
                np.full(arr.size, pos) + jitter,
                arr,
                s=10,
                color=METHOD_COLORS[method],
                alpha=0.40 if is_old else 0.70,
                linewidths=0,
                zorder=3,
            )
            ax.scatter(
                [pos],
                [float(np.mean(arr))],
                marker="D",
                s=24,
                color="#111827",
                edgecolors="white",
                linewidths=0.45,
                zorder=4,
            )

    ax.set_title(DATASET_LABELS.get(dataset, dataset), fontsize=10, pad=5)
    ax.set_xticks(base_x)
    ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], rotation=25, ha="right", fontsize=8)
    ax.set_xlim(-0.6, len(METHODS) - 0.4)
    ax.grid(axis="y", alpha=0.24, linewidth=0.7)
    ax.set_ylim(bottom=0.0)
    if show_ylabel:
        ax.set_ylabel("proj manifold dist", fontsize=9)


def _write_summary_csv(path: Path, old, new) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "dataset",
                "Dataset",
                "method",
                "old_n",
                "old_mean",
                "old_median",
                "new_n",
                "new_mean",
                "new_median",
                "new_minus_old_mean",
                "relative_change",
            ]
        )
        for dataset in DATASETS:
            for method in METHODS:
                o = np.asarray(old.get((dataset, method), []), dtype=float)
                n = np.asarray(new.get((dataset, method), []), dtype=float)
                old_mean = float(np.mean(o)) if o.size else np.nan
                new_mean = float(np.mean(n)) if n.size else np.nan
                delta = new_mean - old_mean if np.isfinite(old_mean) and np.isfinite(new_mean) else np.nan
                rel = delta / old_mean if np.isfinite(delta) and abs(old_mean) > 1e-12 else np.nan
                writer.writerow(
                    [
                        dataset,
                        DATASET_LABELS.get(dataset, dataset),
                        method,
                        int(o.size),
                        old_mean,
                        float(np.median(o)) if o.size else np.nan,
                        int(n.size),
                        new_mean,
                        float(np.median(n)) if n.size else np.nan,
                        delta,
                        rel,
                    ]
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot old-vs-new benchmark comparison for proj_manifold_dist.")
    parser.add_argument("--old", required=True, help="Old benchmark dir or per_case_metrics.csv.")
    parser.add_argument("--new", required=True, help="New benchmark dir or per_case_metrics.csv.")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--metric", default="proj_manifold_dist")
    args = parser.parse_args()

    old = _read_metric(_resolve(args.old), args.metric)
    new = _read_metric(_resolve(args.new), args.metric)
    outdir = _resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(3, 4, figsize=(17.0, 10.0), constrained_layout=True)
    axes_flat = axes.ravel()
    for i, dataset in enumerate(DATASETS):
        _plot_dataset(axes_flat[i], dataset, old, new, show_ylabel=(i % 4 == 0))
    axes_flat[-1].axis("off")
    old_patch = plt.Rectangle((0, 0), 1, 1, facecolor="#9CA3AF", alpha=0.28, hatch="//", edgecolor="#4B5563")
    new_patch = plt.Rectangle((0, 0), 1, 1, facecolor="#9CA3AF", alpha=0.72, edgecolor="#4B5563")
    mean_marker = plt.Line2D([0], [0], marker="D", color="none", markerfacecolor="#111827", markeredgecolor="white", markersize=6)
    fig.legend(
        [old_patch, new_patch, mean_marker],
        ["old bench", "new 15-seed bench", "mean"],
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(0.985, 0.055),
        ncol=1,
    )
    fig.suptitle("Old vs New Benchmark Comparison: proj_manifold_dist", fontsize=13)
    combined_png = outdir / "old_vs_new_proj_manifold_dist_all_envs.png"
    combined_pdf = outdir / "old_vs_new_proj_manifold_dist_all_envs.pdf"
    fig.savefig(combined_png, dpi=220, bbox_inches="tight")
    fig.savefig(combined_pdf, bbox_inches="tight")
    plt.close(fig)

    per_env_dir = outdir / "per_environment"
    per_env_dir.mkdir(exist_ok=True)
    for dataset in DATASETS:
        fig, ax = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
        _plot_dataset(ax, dataset, old, new, show_ylabel=True)
        ax.set_title(f"{DATASET_LABELS.get(dataset, dataset)}: old vs new", fontsize=11)
        handles = [
            plt.Rectangle((0, 0), 1, 1, facecolor="#9CA3AF", alpha=0.28, hatch="//", edgecolor="#4B5563"),
            plt.Rectangle((0, 0), 1, 1, facecolor="#9CA3AF", alpha=0.72, edgecolor="#4B5563"),
            plt.Line2D([0], [0], marker="D", color="none", markerfacecolor="#111827", markeredgecolor="white", markersize=6),
        ]
        ax.legend(handles, ["old bench", "new 15-seed bench", "mean"], frameon=False, loc="upper right", fontsize=8)
        fig.savefig(per_env_dir / f"{dataset}_old_vs_new_proj_manifold_dist.png", dpi=220, bbox_inches="tight")
        plt.close(fig)

    summary_csv = outdir / "old_vs_new_proj_manifold_dist_summary.csv"
    _write_summary_csv(summary_csv, old, new)
    print(f"[saved] {combined_png}")
    print(f"[saved] {combined_pdf}")
    print(f"[saved] {summary_csv}")
    print(f"[saved] {per_env_dir}")


if __name__ == "__main__":
    main()
