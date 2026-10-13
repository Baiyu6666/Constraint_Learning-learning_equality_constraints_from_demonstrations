from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/analysis/noisy_data_boxplots"

ENV_LABELS = {
    "12d_dual_arm_traj": "12D dual arm",
    "3d_vz_2d_ellipse_traj": "3D twist ellipse",
    "6d_surf_normal_traj": "6DSurfNormal",
}
PERT_LABELS = {
    "noise": "Additive noise",
    "discontinuous": "Discontinuous missing",
    "subsample": "Trajectory subsampling",
}
COLORS = {
    "oncl": "#4C78A8",
    "vae": "#F58518",
}
METHOD_LABELS = {
    "oncl": "ONCL",
    "vae": "VAE",
}


def _sources() -> list[dict[str, object]]:
    sources: list[dict[str, object]] = [
        {
            "environment": "12d_dual_arm_traj",
            "method": "oncl",
            "perturbation": "noise",
            "path": ROOT / "outputs/bench/12d_dual_arm_traj_oncl_rawspan_noise_7seed/noise_ratio_per_seed.csv",
            "delta_col": "delta_proj_manifold_dist",
            "value_col": "proj_manifold_dist",
        },
        {
            "environment": "12d_dual_arm_traj",
            "method": "oncl",
            "perturbation": "discontinuous",
            "path": ROOT / "outputs/bench/12d_dual_arm_traj_oncl_missing_segments_7seed/missing_ratio_per_seed.csv",
            "delta_col": "delta_proj_manifold_dist",
            "value_col": "proj_manifold_dist",
        },
        {
            "environment": "12d_dual_arm_traj",
            "method": "oncl",
            "perturbation": "subsample",
            "path": ROOT / "outputs/bench/12d_dual_arm_traj_oncl_subsample_sparse_7seed/subsample_ratio_per_seed_with_0p8.csv",
            "delta_col": "delta_proj_manifold_dist",
            "value_col": "proj_manifold_dist",
        },
    ]
    for perturbation, path in [
        ("noise", ROOT / "outputs/bench/oncl_3dvzellipse_surfnormal_noise_7seed/noise_ratio_per_seed.csv"),
        (
            "discontinuous",
            ROOT / "outputs/bench/oncl_3dvzellipse_surfnormal_discontinuous_norefill_7seed/discontinuous_ratio_per_seed.csv",
        ),
        (
            "subsample",
            ROOT / "outputs/bench/oncl_3dvzellipse_surfnormal_subsample_sparse_7seed/subsample_ratio_per_seed.csv",
        ),
    ]:
        for environment in [
            "3d_vz_2d_ellipse_traj",
            "6d_surf_normal_traj",
        ]:
            sources.append(
                {
                    "environment": environment,
                    "method": "oncl",
                    "perturbation": perturbation,
                    "path": path,
                    "delta_col": "proj_manifold_dist_delta_vs_seed_baseline",
                    "value_col": "proj_manifold_dist",
                }
            )
    for perturbation, path in [
        (
            "discontinuous",
            ROOT / "outputs/bench/vae_3env_discontinuous_norefill_7seed/discontinuous_ratio_per_seed.csv",
        ),
        (
            "subsample",
            ROOT / "outputs/bench/vae_3env_subsample_sparse_7seed/subsample_ratio_per_seed.csv",
        ),
    ]:
        for environment in [
            "12d_dual_arm_traj",
            "3d_vz_2d_ellipse_traj",
            "6d_surf_normal_traj",
        ]:
            sources.append(
                {
                    "environment": environment,
                    "method": "vae",
                    "perturbation": perturbation,
                    "path": path,
                    "delta_col": "proj_manifold_dist_delta_vs_seed_baseline",
                    "value_col": "proj_manifold_dist",
                }
            )
    return sources


def _read_source(source: dict[str, object]) -> list[dict[str, object]]:
    environment = str(source["environment"])
    method = str(source["method"])
    perturbation = str(source["perturbation"])
    path = Path(source["path"])
    rows: list[dict[str, object]] = []
    with path.open() as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            dataset = row.get("dataset") or environment
            if dataset != environment:
                continue
            try:
                ratio = float(row["ratio"])
                seed = int(float(row["seed"]))
                delta = float(row[str(source["delta_col"])])
                value = float(row[str(source["value_col"])])
            except (KeyError, TypeError, ValueError):
                continue
            rows.append(
                {
                    "environment": environment,
                    "environment_label": ENV_LABELS[environment],
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "perturbation": perturbation,
                    "perturbation_label": PERT_LABELS[perturbation],
                    "ratio": ratio,
                    "seed": seed,
                    "proj_manifold_dist": value,
                    "delta_proj_manifold_dist": delta,
                    "source": str(path.relative_to(ROOT)),
                }
            )
    return rows


def _write_plot_data(rows: list[dict[str, object]]) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "oncl_vae_three_env_noisy_proj_raw_boxplot_data.csv"
    cols = [
        "environment",
        "environment_label",
        "method",
        "method_label",
        "perturbation",
        "perturbation_label",
        "ratio",
        "seed",
        "proj_manifold_dist",
        "delta_proj_manifold_dist",
        "source",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, cols)
        writer.writeheader()
        writer.writerows(rows)
    return path


def main() -> None:
    rows: list[dict[str, object]] = []
    for source in _sources():
        rows.extend(_read_source(source))
    data_path = _write_plot_data(rows)

    perturbations = ["noise", "discontinuous", "subsample"]
    environments = [
        "12d_dual_arm_traj",
        "3d_vz_2d_ellipse_traj",
        "6d_surf_normal_traj",
    ]
    fig, axes = plt.subplots(
        len(perturbations),
        len(environments),
        figsize=(15.8, 11.2),
        constrained_layout=False,
    )
    fig.suptitle(
        "ONCL and VAE performance under noisy training data",
        fontsize=17,
        fontweight="bold",
        y=0.985,
    )
    fig.subplots_adjust(left=0.065, right=0.985, top=0.90, bottom=0.09, hspace=0.30, wspace=0.17)

    for row_i, perturbation in enumerate(perturbations):
        for col_i, environment in enumerate(environments):
            ax = axes[row_i][col_i]
            panel = [
                row
                for row in rows
                if row["perturbation"] == perturbation
                and row["environment"] == environment
            ]
            ratios = sorted(
                {float(row["ratio"]) for row in panel},
                reverse=perturbation == "subsample",
            )
            methods = [
                method
                for method in ["oncl", "vae"]
                if any(row["method"] == method for row in panel)
            ]
            offset = {"oncl": 0.0, "vae": 0.0}
            width = 0.55
            if len(methods) == 2:
                offset = {"oncl": -0.18, "vae": 0.18}
                width = 0.30
            data = []
            positions = []
            box_colors = []
            for ratio_i, ratio in enumerate(ratios, start=1):
                for method in methods:
                    values = [
                        float(row["proj_manifold_dist"])
                        for row in panel
                        if float(row["ratio"]) == ratio and row["method"] == method
                    ]
                    if values:
                        data.append(values)
                        positions.append(float(ratio_i) + float(offset[method]))
                        box_colors.append(COLORS[method])
            if data:
                box = ax.boxplot(
                    data,
                    positions=positions,
                    patch_artist=True,
                    widths=width,
                    medianprops={"color": "#111111", "linewidth": 1.2},
                    boxprops={"linewidth": 1.0},
                    whiskerprops={"linewidth": 1.0},
                    capprops={"linewidth": 1.0},
                    flierprops={
                        "marker": "o",
                        "markersize": 3,
                        "markerfacecolor": "#555555",
                        "markeredgecolor": "none",
                        "alpha": 0.55,
                    },
                )
                for patch, color in zip(box["boxes"], box_colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.55)
                ax.set_xticks(range(1, len(ratios) + 1))
                ax.set_xticklabels([f"{ratio:g}" for ratio in ratios], rotation=0)
            ax.axhline(0.0, color="#333333", linewidth=0.8, linestyle="--", alpha=0.7)
            ax.grid(axis="y", color="#D9D9D9", linewidth=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            if row_i == 0:
                ax.set_title(ENV_LABELS[environment], fontsize=12.5, fontweight="bold")
            if col_i == 0:
                ax.set_ylabel(f"{PERT_LABELS[perturbation]}\nproj. dist", fontsize=11)
            else:
                ax.set_ylabel("")
            ax.set_xlabel("ratio" if row_i == len(perturbations) - 1 else "")
            ax.tick_params(axis="both", labelsize=9)
            for spine in ["top", "right"]:
                ax.spines[spine].set_visible(False)

    fig.text(
        0.5,
        0.03,
        "Boxplots show 7 seeds. Metric is proj_manifold_dist, lower is better. "
        "Additive-noise row currently has ONCL only. 12D ONCL discontinuous uses the earlier keep-n-train missing-segment run; VAE discontinuous uses no-refill.",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#333333",
    )
    legend = [
        Patch(facecolor=COLORS[key], edgecolor="#333333", alpha=0.55, label=METHOD_LABELS[key])
        for key in ["oncl", "vae"]
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=3,
        frameon=False,
        fontsize=10,
    )

    png_path = OUT / "oncl_vae_three_env_noisy_proj_raw_boxplot.png"
    pdf_path = OUT / "oncl_vae_three_env_noisy_proj_raw_boxplot.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    print(png_path)
    print(pdf_path)
    print(data_path)


if __name__ == "__main__":
    main()
