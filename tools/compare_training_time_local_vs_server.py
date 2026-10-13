#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable


KEYS = ("method", "dataset", "seed")


def _split_csv_arg(text: str | None) -> set[str] | None:
    if text is None or not str(text).strip():
        return None
    return {x.strip() for x in str(text).split(",") if x.strip()}


def _as_float(value: object) -> float | None:
    try:
        out = float(value)  # type: ignore[arg-type]
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(sum(vals) / len(vals)) if vals else float("nan")


def _median(values: Iterable[float]) -> float:
    vals = list(values)
    return float(statistics.median(vals)) if vals else float("nan")


def _read_per_case(
    path: Path,
    *,
    methods: set[str] | None,
    datasets: set[str] | None,
) -> dict[tuple[str, str, str], dict[str, str]]:
    rows: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        missing = [k for k in (*KEYS, "train_seconds") if k not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path} is missing required columns: {missing}")
        for row in reader:
            method = str(row.get("method", "")).strip()
            dataset = str(row.get("dataset", "")).strip()
            seed = str(row.get("seed", "")).strip()
            if methods is not None and method not in methods:
                continue
            if datasets is not None and dataset not in datasets:
                continue
            seconds = _as_float(row.get("train_seconds"))
            if seconds is None:
                continue
            rows[(method, dataset, seed)] = row
    return rows


def _write_csv(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def _summarize(rows: list[dict[str, object]], group_cols: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[c] for c in group_cols)].append(row)

    out: list[dict[str, object]] = []
    for key, items in grouped.items():
        local = [_as_float(r.get("local_train_seconds")) for r in items]
        server = [_as_float(r.get("server_train_seconds")) for r in items]
        ratios = [_as_float(r.get("local_over_server_ratio")) for r in items]
        deltas = [_as_float(r.get("server_minus_local_seconds")) for r in items]
        local_vals = [x for x in local if x is not None]
        server_vals = [x for x in server if x is not None]
        ratio_vals = [x for x in ratios if x is not None]
        delta_vals = [x for x in deltas if x is not None]

        row = {col: val for col, val in zip(group_cols, key)}
        row.update(
            {
                "n_matched_runs": len(items),
                "local_mean_seconds": _mean(local_vals),
                "server_mean_seconds": _mean(server_vals),
                "local_median_seconds": _median(local_vals),
                "server_median_seconds": _median(server_vals),
                "mean_local_over_server_ratio": _mean(ratio_vals),
                "median_local_over_server_ratio": _median(ratio_vals),
                "mean_server_minus_local_seconds": _mean(delta_vals),
            }
        )
        out.append(row)
    return sorted(out, key=lambda r: tuple(str(r[c]) for c in group_cols))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare train_seconds between local and server benchmark CSVs.")
    parser.add_argument("--local", required=True, help="Local benchmark directory or per_case_metrics.csv")
    parser.add_argument("--server", required=True, help="Server benchmark directory or per_case_metrics.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--methods", default="", help="Optional comma-separated method filter")
    parser.add_argument("--datasets", default="", help="Optional comma-separated dataset filter")
    args = parser.parse_args()

    local_path = Path(args.local)
    server_path = Path(args.server)
    if local_path.is_dir():
        local_path = local_path / "per_case_metrics.csv"
    if server_path.is_dir():
        server_path = server_path / "per_case_metrics.csv"

    methods = _split_csv_arg(args.methods)
    datasets = _split_csv_arg(args.datasets)
    local = _read_per_case(local_path, methods=methods, datasets=datasets)
    server = _read_per_case(server_path, methods=methods, datasets=datasets)

    matched_keys = sorted(set(local) & set(server))
    matched_rows: list[dict[str, object]] = []
    for key in matched_keys:
        method, dataset, seed = key
        local_seconds = _as_float(local[key].get("train_seconds"))
        server_seconds = _as_float(server[key].get("train_seconds"))
        if local_seconds is None or server_seconds is None or server_seconds == 0.0:
            continue
        matched_rows.append(
            {
                "method": method,
                "dataset": dataset,
                "seed": seed,
                "local_train_seconds": local_seconds,
                "server_train_seconds": server_seconds,
                "server_minus_local_seconds": server_seconds - local_seconds,
                "local_over_server_ratio": local_seconds / server_seconds,
            }
        )

    outdir = Path(args.outdir)
    run_cols = [
        "method",
        "dataset",
        "seed",
        "local_train_seconds",
        "server_train_seconds",
        "server_minus_local_seconds",
        "local_over_server_ratio",
    ]
    agg_cols_by_method = [
        "dataset",
        "method",
        "n_matched_runs",
        "local_mean_seconds",
        "server_mean_seconds",
        "local_median_seconds",
        "server_median_seconds",
        "mean_local_over_server_ratio",
        "median_local_over_server_ratio",
        "mean_server_minus_local_seconds",
    ]
    agg_cols_by_dataset = [
        "dataset",
        "n_matched_runs",
        "local_mean_seconds",
        "server_mean_seconds",
        "local_median_seconds",
        "server_median_seconds",
        "mean_local_over_server_ratio",
        "median_local_over_server_ratio",
        "mean_server_minus_local_seconds",
    ]

    _write_csv(outdir / "matched_runs.csv", matched_rows, run_cols)
    _write_csv(outdir / "by_dataset_method.csv", _summarize(matched_rows, ("dataset", "method")), agg_cols_by_method)
    _write_csv(outdir / "by_dataset.csv", _summarize(matched_rows, ("dataset",)), agg_cols_by_dataset)

    print(f"local rows: {len(local)}")
    print(f"server rows: {len(server)}")
    print(f"matched runs: {len(matched_rows)}")
    print(f"missing on server: {len(set(local) - set(server))}")
    print(f"missing locally: {len(set(server) - set(local))}")
    print(f"wrote: {outdir}")


if __name__ == "__main__":
    main()
