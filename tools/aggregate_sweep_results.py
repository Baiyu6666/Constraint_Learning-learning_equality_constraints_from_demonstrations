from __future__ import annotations

import argparse
import csv
import json
import math
import statistics as stats
from pathlib import Path
from typing import Any


METRICS = [
    "proj_manifold_dist",
    "bidirectional_chamfer",
    "pred_recall",
    "pred_FPrate",
    "pred_precision",
    "proj_true_dist",
    "train_seconds",
]


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _mean(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return sum(clean) / len(clean) if clean else None


def _std(values: list[float | int | None]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    if not clean:
        return None
    if len(clean) == 1:
        return 0.0
    return stats.stdev(clean)


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, int):
        return str(value)
    return f"{float(value):.10g}"


def _infer_traj_len(cfg: dict[str, Any]) -> int:
    n_train = int(cfg.get("n_train") or 0)
    traj_count = int(cfg.get("traj_count") or 1)
    return max(1, math.ceil(n_train / max(1, traj_count)))


def _actual_n_train_est(experiment: str, cfg: dict[str, Any], ratio: float) -> int:
    base = int(cfg.get("n_train") or 0)
    if experiment == "noise":
        return base

    if "traj_len" not in cfg:
        if experiment == "subsample":
            return max(1, min(base, int(round(float(ratio) * base))))
        if experiment == "discontinuous":
            removed = min(base - 1, int(round(float(ratio) * base)))
            return max(1, base - removed)

    traj_count = int(cfg.get("traj_count") or 1)
    traj_len = _infer_traj_len(cfg)
    if experiment == "subsample":
        keep = max(1, min(traj_len, int(round(float(ratio) * traj_len))))
        return traj_count * keep
    if experiment == "discontinuous":
        missing = min(traj_len - 1, int(round(float(ratio) * traj_len)))
        keep = max(1, traj_len - missing)
        return traj_count * keep
    return base


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def aggregate(
    *,
    outdir: Path,
    experiment: str,
    ratio_key: str,
    baseline_ratio: float,
    summary_name: str,
    per_seed_name: str,
    sort_desc: bool,
) -> None:
    rows = _read_jsonl(outdir / "per_run_metrics.jsonl")
    per_seed_rows: list[dict[str, Any]] = []
    by_method_dataset_seed_ratio: dict[tuple[str, str, int, float], dict[str, Any]] = {}

    for row in rows:
        if row.get("status") != "success":
            continue
        cfg = row.get("config") or {}
        metrics = row.get("metrics") or {}
        ratio = _num(cfg.get(ratio_key))
        if ratio is None:
            raise ValueError(f"missing ratio key {ratio_key!r} in {row.get('job')}")
        record: dict[str, Any] = {
            "experiment": experiment,
            "method": row.get("method") or "",
            "dataset": row["dataset"],
            "ratio": ratio,
            "seed": int(row["seed"]),
            "actual_n_train_est": _actual_n_train_est(experiment, cfg, ratio),
            "status": row.get("status"),
        }
        for metric in METRICS:
            record[metric] = _num(metrics.get(metric))
        per_seed_rows.append(record)
        by_method_dataset_seed_ratio[
            (record["method"], record["dataset"], record["seed"], ratio)
        ] = record

    for record in per_seed_rows:
        baseline = by_method_dataset_seed_ratio.get(
            (record["method"], record["dataset"], record["seed"], baseline_ratio)
        )
        for metric in METRICS:
            base_value = baseline.get(metric) if baseline else None
            value = record.get(metric)
            delta_key = f"{metric}_delta_vs_seed_baseline"
            record[delta_key] = (
                value - base_value
                if value is not None and base_value is not None
                else None
            )

    per_seed_rows.sort(
        key=lambda item: (
            item["method"],
            item["dataset"],
            -item["ratio"] if sort_desc else item["ratio"],
            item["seed"],
        )
    )
    per_seed_cols = [
        "experiment",
        "method",
        "dataset",
        "ratio",
        "seed",
        "actual_n_train_est",
        "status",
    ]
    per_seed_cols += METRICS
    per_seed_cols += [f"{metric}_delta_vs_seed_baseline" for metric in METRICS]
    with (outdir / per_seed_name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, per_seed_cols)
        writer.writeheader()
        for record in per_seed_rows:
            writer.writerow({key: _fmt(record.get(key)) for key in per_seed_cols})

    groups: dict[tuple[str, str, float], list[dict[str, Any]]] = {}
    for record in per_seed_rows:
        groups.setdefault(
            (record["method"], record["dataset"], record["ratio"]), []
        ).append(record)

    summary_rows: list[dict[str, Any]] = []
    for (method, dataset, ratio), group in groups.items():
        summary: dict[str, Any] = {
            "experiment": experiment,
            "method": method,
            "dataset": dataset,
            "ratio": ratio,
            "n_seeds": len(group),
            "actual_n_train_est_mean": _mean(
                [record["actual_n_train_est"] for record in group]
            ),
        }
        for metric in METRICS:
            values = [record[metric] for record in group]
            deltas = [record[f"{metric}_delta_vs_seed_baseline"] for record in group]
            summary[f"{metric}_mean"] = _mean(values)
            summary[f"{metric}_std"] = _std(values)
            summary[f"{metric}_delta_mean"] = _mean(deltas)
            summary[f"{metric}_delta_std"] = _std(deltas)
        summary_rows.append(summary)

    summary_rows.sort(
        key=lambda item: (
            item["method"],
            item["dataset"],
            -item["ratio"] if sort_desc else item["ratio"],
        )
    )
    summary_cols = [
        "experiment",
        "method",
        "dataset",
        "ratio",
        "n_seeds",
        "actual_n_train_est_mean",
    ]
    for metric in METRICS:
        summary_cols += [
            f"{metric}_mean",
            f"{metric}_std",
            f"{metric}_delta_mean",
            f"{metric}_delta_std",
        ]
    with (outdir / summary_name).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, summary_cols)
        writer.writeheader()
        for record in summary_rows:
            writer.writerow({key: _fmt(record.get(key)) for key in summary_cols})

    print(outdir / summary_name)
    print(outdir / per_seed_name)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=Path, required=True)
    parser.add_argument("--experiment", required=True)
    parser.add_argument("--ratio-key", required=True)
    parser.add_argument("--baseline-ratio", type=float, required=True)
    parser.add_argument("--summary-name", required=True)
    parser.add_argument("--per-seed-name", required=True)
    parser.add_argument("--sort-desc", action="store_true")
    args = parser.parse_args()

    aggregate(
        outdir=args.outdir,
        experiment=args.experiment,
        ratio_key=args.ratio_key,
        baseline_ratio=args.baseline_ratio,
        summary_name=args.summary_name,
        per_seed_name=args.per_seed_name,
        sort_desc=args.sort_desc,
    )


if __name__ == "__main__":
    main()
