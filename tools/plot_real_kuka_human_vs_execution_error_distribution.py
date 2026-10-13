#!/usr/bin/env python3
"""Compare human demonstration and real execution error distributions."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HUMAN_NPZ = (
    PROJECT_ROOT
    / "datasets/real_iiwa/tasks/RealKuka/processed/train_sets/"
    / "human_demo_paper_toolz_arc3mm_cropped_v001.npz"
)
DEFAULT_EXEC_ROOT = PROJECT_ROOT / "kuka_experiment_data/learned_from_human"
DEFAULT_OUTDIR = PROJECT_ROOT / "datasets/real_iiwa/tasks/RealKuka/figures"


def _stats(values: np.ndarray) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    return {
        "n": int(len(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
        "median": float(np.median(v)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }


def _load_human(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    position_mm = 1000.0 * np.asarray(data["position_error_m"], dtype=np.float64)
    orientation_deg = np.asarray(data["orientation_error_deg"], dtype=np.float64)
    return position_mm, orientation_deg


def _load_execution(root: Path, segment: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    exec_pos: list[float] = []
    exec_ori: list[float] = []
    per_traj: list[dict[str, object]] = []
    for path in sorted(root.glob("trajectory_*/constraint_plan_*.json")):
        with path.open() as handle:
            data = json.load(handle)
        local_exec_pos: list[float] = []
        local_exec_ori: list[float] = []
        for sample in data.get("samples", []):
            if segment and sample.get("segment") != segment:
                continue

            exec_pos_m = sample.get("reference_true_surface_position_error_m")
            exec_ori_deg = sample.get("reference_true_surface_orientation_error_deg")
            if exec_pos_m is not None and exec_ori_deg is not None:
                value_pos = 1000.0 * float(exec_pos_m)
                value_ori = float(exec_ori_deg)
                local_exec_pos.append(value_pos)
                local_exec_ori.append(value_ori)
                exec_pos.append(value_pos)
                exec_ori.append(value_ori)

        if local_exec_pos:
            per_traj.append(
                {
                    "trajectory": path.parent.name,
                    "samples": len(local_exec_pos),
                    "position_error_mm_mean": float(np.mean(local_exec_pos)),
                    "orientation_error_deg_mean": float(np.mean(local_exec_ori)),
                }
            )
    return np.asarray(exec_pos, dtype=np.float64), np.asarray(exec_ori, dtype=np.float64), per_traj


def _plot_panel(
    ax: plt.Axes,
    human: np.ndarray,
    execution: np.ndarray,
    *,
    bins: np.ndarray,
    xlabel: str,
    title: str,
) -> None:
    h_weights = np.full(len(human), 100.0 / max(1, len(human)), dtype=np.float64)
    e_weights = np.full(len(execution), 100.0 / max(1, len(execution)), dtype=np.float64)
    ax.hist(
        human,
        bins=bins,
        weights=h_weights,
        color="#4B5563",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.55,
        label="Human demonstration",
    )
    ax.hist(
        execution,
        bins=bins,
        weights=e_weights,
        color="#F59E0B",
        alpha=0.48,
        edgecolor="white",
        linewidth=0.55,
        label="Robot execution",
    )
    h = _stats(human)
    e = _stats(execution)
    human_mean = h["mean"]
    human_std = h["std"]
    exec_mean = e["mean"]
    exec_std = e["std"]
    ax.axvline(human_mean, color="#EF1D27", linewidth=1.5, label="Human mean")
    ax.axvline(exec_mean, color="#C2410C", linewidth=1.5, linestyle="--", label="Execution mean")
    text = (
        f"human: {human_mean:.3g} +/- {human_std:.3g}\n"
        f"execution: {exec_mean:.3g} +/- {exec_std:.3g}"
    )
    ax.text(
        0.97,
        0.80,
        text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.2,
        bbox={
            "boxstyle": "round,pad=0.22",
            "facecolor": "white",
            "edgecolor": "#D1D5DB",
            "linewidth": 1.0,
            "alpha": 0.94,
        },
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Share of samples")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:.0f}%"))
    ax.grid(True, axis="y", color="#D1D5DB", linewidth=0.65, alpha=0.78)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-npz", type=Path, default=DEFAULT_HUMAN_NPZ)
    parser.add_argument("--execution-root", type=Path, default=DEFAULT_EXEC_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--segment", default="following_manifold")
    args = parser.parse_args()

    human_pos, human_ori = _load_human(args.human_npz)
    exec_pos, exec_ori, per_traj = _load_execution(args.execution_root, args.segment)
    if len(exec_pos) == 0:
        raise RuntimeError(f"no execution samples found under {args.execution_root}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "human_npz": str(args.human_npz),
        "execution_root": str(args.execution_root),
        "execution_segment": str(args.segment),
        "human_position_error_mm": _stats(human_pos),
        "human_orientation_error_deg": _stats(human_ori),
        "execution_position_error_mm": _stats(exec_pos),
        "execution_orientation_error_deg": _stats(exec_ori),
        "execution_per_trajectory": per_traj,
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "axes.labelsize": 10.5,
            "axes.titlesize": 12.0,
            "legend.fontsize": 8.4,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.linewidth": 0.85,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.1), constrained_layout=False)
    pos_hi = max(5.0, float(np.percentile(np.concatenate([human_pos, exec_pos]), 99.5)) * 1.08)
    ori_hi = max(13.0, float(np.percentile(np.concatenate([human_ori, exec_ori]), 99.5)) * 1.08)
    pos_bins = np.linspace(0.0, pos_hi, 28)
    ori_bins = np.linspace(0.0, ori_hi, 28)
    _plot_panel(
        axes[0],
        human_pos,
        exec_pos,
        bins=pos_bins,
        xlabel="Position error to true constraint (mm)",
        title="Position error",
    )
    _plot_panel(
        axes[1],
        human_ori,
        exec_ori,
        bins=ori_bins,
        xlabel="Orientation error to true constraint (deg)",
        title="Orientation error",
    )
    handles, labels = axes[0].get_legend_handles_labels()
    keep = ["Human demonstration", "Robot execution", "Human mean", "Execution mean"]
    by_label = {label: handle for handle, label in zip(handles, labels)}
    fig.legend(
        [by_label[label] for label in keep],
        keep,
        loc="upper center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 1.00),
        handlelength=1.4,
        columnspacing=1.0,
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.27, top=0.83, wspace=0.28)

    stem = "human_vs_execution_true_constraint_error_distribution"
    png_path = args.outdir / f"{stem}.png"
    pdf_path = args.outdir / f"{stem}.pdf"
    summary_path = args.outdir / f"{stem}_summary.json"
    bins_path = args.outdir / f"{stem}_bins.csv"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    with bins_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "bin_edges"])
        writer.writerow(["position_error_mm", json.dumps(pos_bins.tolist())])
        writer.writerow(["orientation_error_deg", json.dumps(ori_bins.tolist())])

    print(json.dumps(summary, indent=2))
    print(png_path)
    print(pdf_path)
    print(summary_path)
    print(bins_path)


if __name__ == "__main__":
    main()
