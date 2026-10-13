#!/usr/bin/env python3
"""Boxplots for human, learned, and execution errors on the real KUKA task."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
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
        "q25": float(np.percentile(v, 25)),
        "q75": float(np.percentile(v, 75)),
        "p95": float(np.percentile(v, 95)),
        "max": float(np.max(v)),
    }


def _angle_deg(a: list[float], b: list[float]) -> float:
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom <= 1e-12:
        return float("nan")
    cosine = float(np.dot(va, vb) / denom)
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _load_human(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    position_mm = 1000.0 * np.asarray(data["position_error_m"], dtype=np.float64)
    orientation_deg = np.asarray(data["orientation_error_deg"], dtype=np.float64)
    return position_mm, orientation_deg


def _load_learned_and_execution(
    root: Path,
    segment: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    learned_pos: list[float] = []
    learned_ori: list[float] = []
    exec_pos: list[float] = []
    exec_ori: list[float] = []
    for path in sorted(root.glob("trajectory_*/constraint_plan_*.json")):
        with path.open() as handle:
            data = json.load(handle)
        for sample in data.get("samples", []):
            if segment and sample.get("segment") != segment:
                continue

            exec_pos_m = sample.get("reference_true_surface_position_error_m")
            exec_ori_deg = sample.get("reference_true_surface_orientation_error_deg")
            if exec_pos_m is not None and exec_ori_deg is not None:
                exec_pos.append(1000.0 * float(exec_pos_m))
                exec_ori.append(float(exec_ori_deg))

            learned_signed = sample.get("reference_learned_surface_signed_position_error_m")
            true_signed = sample.get("reference_true_surface_signed_position_error_m")
            if learned_signed is not None and true_signed is not None:
                learned_pos.append(1000.0 * abs(float(true_signed) - float(learned_signed)))

            learned_tool_z = sample.get("reference_learned_surface_expected_tool_z")
            true_tool_z = sample.get("reference_true_surface_expected_tool_z")
            if learned_tool_z is not None and true_tool_z is not None:
                angle = _angle_deg(learned_tool_z, true_tool_z)
                if np.isfinite(angle):
                    learned_ori.append(angle)

    arrays = tuple(
        np.asarray(values, dtype=np.float64)
        for values in (learned_pos, learned_ori, exec_pos, exec_ori)
    )
    if any(len(values) == 0 for values in arrays):
        raise RuntimeError(f"missing learned/execution samples under {root}")
    return arrays


def _draw_boxplot(ax: plt.Axes, values: list[np.ndarray], ylabel: str) -> None:
    colors = ["#4B5563", "#2563EB", "#F59E0B"]
    bp = ax.boxplot(
        values,
        tick_labels=["Human", "Learned", "Executed"],
        widths=0.55,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        showfliers=False,
        medianprops={"color": "black", "linewidth": 1.2},
        meanprops={"color": "#EF1D27", "linewidth": 1.3},
        whiskerprops={"color": "#4B5563", "linewidth": 1.0},
        capprops={"color": "#4B5563", "linewidth": 1.0},
    )
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.62)
        patch.set_edgecolor("#374151")
        patch.set_linewidth(1.0)
    ax.set_ylabel(ylabel, labelpad=2)
    ax.grid(True, axis="y", color="#D1D5DB", linewidth=0.65, alpha=0.78)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="x", pad=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--human-npz", type=Path, default=DEFAULT_HUMAN_NPZ)
    parser.add_argument("--execution-root", type=Path, default=DEFAULT_EXEC_ROOT)
    parser.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--segment", default="following_manifold")
    args = parser.parse_args()

    human_pos, human_ori = _load_human(args.human_npz)
    learned_pos, learned_ori, exec_pos, exec_ori = _load_learned_and_execution(
        args.execution_root,
        args.segment,
    )

    args.outdir.mkdir(parents=True, exist_ok=True)
    summary = {
        "human_npz": str(args.human_npz),
        "execution_root": str(args.execution_root),
        "execution_segment": str(args.segment),
        "definition": {
            "human": "recorded human demonstration error relative to true constraint",
            "learned": "learned constraint error relative to true constraint at execution-path samples",
            "execution": "robot execution error relative to true constraint",
        },
        "position_error_mm": {
            "human": _stats(human_pos),
            "learned": _stats(learned_pos),
            "execution": _stats(exec_pos),
        },
        "orientation_error_deg": {
            "human": _stats(human_ori),
            "learned": _stats(learned_ori),
            "execution": _stats(exec_ori),
        },
    }

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.2,
            "axes.labelsize": 7.8,
            "axes.titlesize": 8.6,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "axes.linewidth": 0.65,
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(4.0, 1.15), constrained_layout=False)
    _draw_boxplot(axes[0], [human_pos, learned_pos, exec_pos], "Error (mm)")
    _draw_boxplot(axes[1], [human_ori, learned_ori, exec_ori], "Error (deg)")
    fig.subplots_adjust(left=0.10, right=0.995, bottom=0.31, top=0.98, wspace=0.36)

    stem = "human_learned_execution_true_constraint_error_boxplot"
    png_path = args.outdir / f"{stem}.png"
    pdf_path = args.outdir / f"{stem}.pdf"
    summary_path = args.outdir / f"{stem}_summary.json"
    csv_path = args.outdir / f"{stem}_samples.csv"
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")

    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "source", "value"])
        for metric, groups in (
            ("position_error_mm", [("human", human_pos), ("learned", learned_pos), ("execution", exec_pos)]),
            ("orientation_error_deg", [("human", human_ori), ("learned", learned_ori), ("execution", exec_ori)]),
        ):
            for source, arr in groups:
                for value in arr:
                    writer.writerow([metric, source, float(value)])

    print(json.dumps(summary, indent=2))
    print(png_path)
    print(pdf_path)
    print(summary_path)
    print(csv_path)


if __name__ == "__main__":
    main()
