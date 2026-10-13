from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


ROOT = Path(__file__).resolve().parents[1]
DATASET = "6d_surf_normal_traj"
METRIC = "proj_manifold_dist"
METRIC_LABEL = r"Distance Error $\epsilon_l$"

OUTDIR = ROOT / "outputs" / "analysis" / "noisy_data_lines"

METHOD_STYLE = {
    "oncl": {"label": "ONCL", "color": "#4C78A8", "marker": "o"},
    "vae": {"label": "VAE", "color": "#F58518", "marker": "s"},
    "dataaug": {"label": "DataAug", "color": "#54A24B", "marker": "^"},
}

COMBINED_STEM = "surfnormal_noisydata_three_panel_line"


@dataclass(frozen=True)
class Source:
    method: str | None
    path: Path


@dataclass(frozen=True)
class PlotSpec:
    stem: str
    title: str
    xlabel: str
    x_col: str
    sources: tuple[Source, ...]


SPECS = (
    PlotSpec(
        stem="surfnormal_median_nn_noise_alpha_line",
        title="Gaussian noise",
        xlabel=r"Relative noise level $\alpha$",
        x_col="ratio",
        sources=(
            Source(None, ROOT / "outputs/bench/oncl_vae_surfnormal_median_nn_noise_7seed/median_nn_noise_alpha_summary.csv"),
            Source(None, ROOT / "outputs/bench/dataaug_surfnormal_median_nn_noise_7seed/median_nn_noise_alpha_summary.csv"),
        ),
    ),
    PlotSpec(
        stem="surfnormal_discontinuous_ratio_line",
        title="Discontinuous trajectories",
        xlabel="Discontinuity ratio",
        x_col="ratio",
        sources=(
            Source("oncl", ROOT / "outputs/bench/oncl_3dvzellipse_surfnormal_discontinuous_norefill_7seed/discontinuous_ratio_summary.csv"),
            Source("vae", ROOT / "outputs/bench/vae_3env_discontinuous_norefill_7seed/discontinuous_ratio_summary.csv"),
            Source(None, ROOT / "outputs/bench/dataaug_surfnormal_discontinuous_norefill_7seed/discontinuous_ratio_summary.csv"),
        ),
    ),
    PlotSpec(
        stem="surfnormal_subsample_ratio_line",
        title="Sparse trajectories",
        xlabel="Within-trajectory sparsity",
        x_col="ratio",
        sources=(
            Source("oncl", ROOT / "outputs/bench/oncl_3dvzellipse_surfnormal_subsample_sparse_7seed/subsample_ratio_summary.csv"),
            Source("vae", ROOT / "outputs/bench/vae_3env_subsample_sparse_7seed/subsample_ratio_summary.csv"),
            Source(None, ROOT / "outputs/bench/dataaug_surfnormal_subsample_sparse_7seed/subsample_ratio_summary.csv"),
        ),
    ),
)


def _float(value: str) -> float:
    return float(value.strip())


def _read_spec_rows(spec: PlotSpec) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for source in spec.sources:
        with source.path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                if row.get("dataset") != DATASET:
                    continue
                method = row.get("method") or source.method
                if method not in METHOD_STYLE:
                    continue
                out = dict(row)
                out["method"] = method
                out["noise_type"] = spec.stem
                out["plot_ratio"] = _format_x(_plot_x_value(spec, _float(row[spec.x_col])))
                rows.append(out)
    return rows


def _write_tidy_csv(all_rows: Iterable[dict[str, str]]) -> Path:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTDIR / "surfnormal_noisydata_line_points.csv"
    fields = [
        "noise_type",
        "method",
        "dataset",
        "ratio",
        "plot_ratio",
        "n_seeds",
        f"{METRIC}_mean",
        f"{METRIC}_std",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in all_rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return out_path


def _format_x(value: float) -> str:
    return f"{value:g}"


def _format_tick(spec: PlotSpec, value: float) -> str:
    if spec.stem == "surfnormal_subsample_ratio_line":
        return f"{value:g}x"
    return _format_x(value)


def _plot_x_value(spec: PlotSpec, raw_value: float) -> float:
    if spec.stem == "surfnormal_subsample_ratio_line":
        return 1.0 / raw_value
    return raw_value


def _draw_spec_on_axis(
    ax: Axes,
    spec: PlotSpec,
    rows: list[dict[str, str]],
    *,
    show_ylabel: bool,
    show_y_ticklabels: bool = True,
    label_line_ends: bool,
    compact: bool,
    tight_x: bool = False,
) -> None:
    x_values = sorted({_plot_x_value(spec, _float(row[spec.x_col])) for row in rows})
    x_positions = {value: i for i, value in enumerate(x_values)}
    line_ends: list[tuple[float, float, str, str]] = []
    fontsize_axis = 10.2 if compact else 9.0
    fontsize_tick = 9.2 if compact else 8.0
    fontsize_label = 8.0 if compact else 7.2
    linewidth = 1.5 if compact else 1.45
    markersize = 3.8 if compact else 3.4
    capsize = 2.0 if compact else 2.0

    for method, style in METHOD_STYLE.items():
        method_rows = [row for row in rows if row["method"] == method]
        method_rows.sort(key=lambda row: _plot_x_value(spec, _float(row[spec.x_col])))
        if not method_rows:
            continue

        xs = [
            x_positions[_plot_x_value(spec, _float(row[spec.x_col]))]
            for row in method_rows
        ]
        ys = [_float(row[f"{METRIC}_mean"]) for row in method_rows]
        yerrs = [_float(row[f"{METRIC}_std"]) for row in method_rows]

        ax.errorbar(
            xs,
            ys,
            yerr=yerrs,
            color=style["color"],
            marker=style["marker"],
            linestyle="-",
            linewidth=linewidth,
            markersize=markersize,
            markeredgewidth=0.8,
            capsize=capsize,
            elinewidth=0.8,
            label=style["label"],
            zorder=3,
        )
        line_ends.append((xs[-1], ys[-1], style["label"], style["color"]))

    if tight_x:
        ax.set_xlim(-0.25, len(x_values) - 0.72)
    else:
        ax.set_xlim(-0.25, len(x_values) + (0.45 if label_line_ends else 0.1))
    ax.set_xticks(list(range(len(x_values))))
    ax.set_xticklabels([_format_tick(spec, x) for x in x_values])
    ax.set_xlabel(spec.xlabel, fontsize=fontsize_axis, labelpad=1.5)
    if show_ylabel:
        ax.set_ylabel(METRIC_LABEL, fontsize=fontsize_axis, labelpad=1.5)
    ax.tick_params(axis="y", labelleft=show_y_ticklabels)
    if not show_y_ticklabels:
        ax.tick_params(axis="y", labelleft=False)
    ax.tick_params(axis="both", labelsize=fontsize_tick, pad=1.2)
    ax.grid(axis="y", alpha=0.24, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    if not show_y_ticklabels:
        ax.tick_params(axis="y", length=0)
    if not label_line_ends:
        return
    for x, y, label, color in line_ends:
        ax.text(
            x + 0.08,
            y,
            label,
            color=color,
            fontsize=fontsize_label,
            va="center",
            ha="left",
            clip_on=False,
        )


def _plot_spec(spec: PlotSpec, rows: list[dict[str, str]]) -> tuple[Path, Path]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(2.45, 1.5))
    _draw_spec_on_axis(
        ax,
        spec,
        rows,
        show_ylabel=True,
        show_y_ticklabels=True,
        label_line_ends=True,
        compact=False,
    )
    fig.subplots_adjust(left=0.24, right=0.98, top=0.96, bottom=0.2)
    png = OUTDIR / f"{spec.stem}.png"
    pdf = OUTDIR / f"{spec.stem}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def _shared_y_limits(rows_by_spec: dict[PlotSpec, list[dict[str, str]]]) -> tuple[float, float]:
    low = float("inf")
    high = float("-inf")
    for rows in rows_by_spec.values():
        for row in rows:
            mean = _float(row[f"{METRIC}_mean"])
            std = _float(row[f"{METRIC}_std"])
            low = min(low, mean - std)
            high = max(high, mean + std)
    pad = max(0.01, (high - low) * 0.08)
    return max(0.0, low - pad), high + pad


def _plot_combined(rows_by_spec: dict[PlotSpec, list[dict[str, str]]]) -> tuple[Path, Path]:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 1.65), sharey=True)
    y_limits = _shared_y_limits(rows_by_spec)
    for i, (ax, spec) in enumerate(zip(axes, SPECS)):
        _draw_spec_on_axis(
            ax,
            spec,
            rows_by_spec[spec],
            show_ylabel=i == 0,
            show_y_ticklabels=True,
            label_line_ends=False,
            compact=True,
            tight_x=True,
        )
        ax.set_ylim(y_limits)
        if i == 0:
            ax.legend(
                frameon=False,
                fontsize=9.0,
                loc="upper left",
                handlelength=1.7,
                borderaxespad=0.2,
                labelspacing=0.25,
            )
    fig.subplots_adjust(left=0.085, right=0.995, top=0.97, bottom=0.31, wspace=0.22)
    png = OUTDIR / f"{COMBINED_STEM}.png"
    pdf = OUTDIR / f"{COMBINED_STEM}.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    all_rows: list[dict[str, str]] = []
    rows_by_spec: dict[PlotSpec, list[dict[str, str]]] = {}
    for spec in SPECS:
        rows = _read_spec_rows(spec)
        if not rows:
            raise RuntimeError(f"No rows found for {spec.stem}")
        rows_by_spec[spec] = rows
        all_rows.extend(rows)
        png, pdf = _plot_spec(spec, rows)
        print(png)
        print(pdf)
    png, pdf = _plot_combined(rows_by_spec)
    print(png)
    print(pdf)
    print(_write_tidy_csv(all_rows))


if __name__ == "__main__":
    main()
