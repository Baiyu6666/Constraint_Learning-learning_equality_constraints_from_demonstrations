from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Any

import matplotlib.pyplot as plt


METHOD_LABELS = {
    "oncl": "ONCL",
    "dataaug": "DataAug",
    "vae": "VAE",
    "ecomann": "ECoMaNN",
}

METHOD_COLORS = {
    "oncl": "#4C78A8",
    "dataaug": "#54A24B",
    "vae": "#F58518",
    "ecomann": "#B279A2",
}


def _resolve(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


def _input_file(path: Path) -> Path:
    if path.is_file():
        return path
    csv_path = path / "per_case_metrics.csv"
    if csv_path.exists():
        return csv_path
    json_path = path / "parallel_summary.json"
    if json_path.exists():
        return json_path
    raise FileNotFoundError(f"missing per_case_metrics.csv or parallel_summary.json under {path}")


def _num(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _read_rows(path: Path, dataset: str) -> list[dict[str, Any]]:
    path = _input_file(path)
    rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text())
        for row in data:
            if row.get("dataset") != dataset or row.get("status") != "success":
                continue
            rec = {
                "dataset": row.get("dataset"),
                "method": row.get("method"),
                "seed": int(row.get("seed")),
            }
            metrics = row.get("metrics") or {}
            if isinstance(metrics, dict):
                rec.update(metrics)
            rows.append(rec)
        return rows

    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row.get("dataset") != dataset or row.get("status", "success") not in ("", "success"):
                continue
            rec = dict(row)
            rec["seed"] = int(float(rec["seed"]))
            rows.append(rec)
    return rows


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": mean(values),
        "std": stdev(values) if len(values) > 1 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    return f"{float(value):.10g}"


def _plot_metric(
    outdir: Path,
    summary_rows: list[dict[str, Any]],
    *,
    metric: str,
    old_label: str,
    new_label: str,
) -> tuple[Path, Path]:
    methods = [m for m in ("oncl", "dataaug", "vae", "ecomann") if any(r["method"] == m for r in summary_rows)]
    fig, ax = plt.subplots(figsize=(5.0, 2.2))
    width = 0.34
    xs = list(range(len(methods)))
    old_means = []
    new_means = []
    old_stds = []
    new_stds = []
    for method in methods:
        row = next(r for r in summary_rows if r["method"] == method and r["metric"] == metric)
        old_means.append(row["old_mean"])
        new_means.append(row["new_mean"])
        old_stds.append(row["old_std"])
        new_stds.append(row["new_std"])
    ax.bar([x - width / 2 for x in xs], old_means, width, yerr=old_stds, label=old_label, color="#A0AEC0", capsize=2)
    ax.bar([x + width / 2 for x in xs], new_means, width, yerr=new_stds, label=new_label, color="#4C78A8", capsize=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax.set_ylabel(metric)
    ax.tick_params(axis="both", labelsize=8)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.25)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout(pad=0.5)
    png = outdir / f"{metric}_dataset_pair_compare.png"
    pdf = outdir / f"{metric}_dataset_pair_compare.pdf"
    fig.savefig(png, dpi=300)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old", required=True)
    parser.add_argument("--new", required=True)
    parser.add_argument("--old-dataset", required=True)
    parser.add_argument("--new-dataset", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--metrics", default="proj_manifold_dist,bidirectional_chamfer,proj_true_dist,pred_recall,train_seconds")
    args = parser.parse_args()

    outdir = _resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    metrics = [m.strip() for m in args.metrics.split(",") if m.strip()]
    old_rows = _read_rows(_resolve(args.old), args.old_dataset)
    new_rows = _read_rows(_resolve(args.new), args.new_dataset)

    old_by_key = {(r["method"], int(r["seed"])): r for r in old_rows}
    new_by_key = {(r["method"], int(r["seed"])): r for r in new_rows}
    methods = sorted({k[0] for k in old_by_key} & {k[0] for k in new_by_key})

    paired_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    result: dict[str, Any] = {
        "old_dataset": args.old_dataset,
        "new_dataset": args.new_dataset,
        "old_source": str(_input_file(_resolve(args.old))),
        "new_source": str(_input_file(_resolve(args.new))),
        "metrics": {},
    }

    for metric in metrics:
        result["metrics"][metric] = {}
        for method in methods:
            seeds = sorted(seed for meth, seed in old_by_key if meth == method and (method, seed) in new_by_key)
            old_values: list[float] = []
            new_values: list[float] = []
            diffs: list[float] = []
            for seed in seeds:
                old_v = _num(old_by_key[(method, seed)].get(metric))
                new_v = _num(new_by_key[(method, seed)].get(metric))
                if old_v is None or new_v is None:
                    continue
                old_values.append(old_v)
                new_values.append(new_v)
                diffs.append(new_v - old_v)
                paired_rows.append({
                    "metric": metric,
                    "method": method,
                    "seed": seed,
                    "old_value": old_v,
                    "new_value": new_v,
                    "new_minus_old": new_v - old_v,
                    "relative_change": (new_v - old_v) / old_v if old_v != 0 else None,
                })
            old_s = _summary(old_values)
            new_s = _summary(new_values)
            diff_s = _summary(diffs)
            summary = {
                "metric": metric,
                "method": method,
                "n": len(diffs),
                "old_mean": old_s["mean"],
                "old_std": old_s["std"],
                "new_mean": new_s["mean"],
                "new_std": new_s["std"],
                "new_minus_old_mean": diff_s["mean"],
                "new_minus_old_std": diff_s["std"],
                "relative_change_mean": (
                    (new_s["mean"] - old_s["mean"]) / old_s["mean"]
                    if old_s["mean"] not in (None, 0)
                    else None
                ),
            }
            summary_rows.append(summary)
            result["metrics"][metric][method] = summary

    summary_csv = outdir / "dataset_pair_summary.csv"
    paired_csv = outdir / "dataset_pair_paired.csv"
    with summary_csv.open("w", newline="") as handle:
        fields = ["metric", "method", "n", "old_mean", "old_std", "new_mean", "new_std", "new_minus_old_mean", "new_minus_old_std", "relative_change_mean"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({k: _fmt(row.get(k)) for k in fields})
    with paired_csv.open("w", newline="") as handle:
        fields = ["metric", "method", "seed", "old_value", "new_value", "new_minus_old", "relative_change"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in paired_rows:
            writer.writerow({k: _fmt(row.get(k)) for k in fields})

    json_path = outdir / "dataset_pair_summary.json"
    json_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    png, pdf = _plot_metric(outdir, summary_rows, metric="proj_manifold_dist", old_label="old", new_label="new")
    print(summary_csv)
    print(paired_csv)
    print(json_path)
    print(png)
    print(pdf)


if __name__ == "__main__":
    main()
