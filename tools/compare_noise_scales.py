from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.config_loader import apply_overrides, load_layered_config
from experiments.dataset_resolve import resolve_dataset
from experiments.unified_experiment import _attach_extra_cfg_fields, _build_cfg_from_mapping_strict
from methods import oncl as ve


def _nearest_neighbor_distances(x: np.ndarray) -> np.ndarray:
    if len(x) <= 1:
        return np.zeros((len(x),), dtype=np.float64)
    try:
        from scipy.spatial import cKDTree  # type: ignore

        tree = cKDTree(x.astype(np.float64))
        dists, _ = tree.query(x.astype(np.float64), k=2)
        return np.asarray(dists[:, 1], dtype=np.float64)
    except Exception:
        xx = x.astype(np.float64)
        out = np.empty((len(xx),), dtype=np.float64)
        chunk = 512
        for start in range(0, len(xx), chunk):
            end = min(len(xx), start + chunk)
            d2 = np.sum((xx[start:end, None, :] - xx[None, :, :]) ** 2, axis=2)
            rows = np.arange(end - start)
            d2[rows, start + rows] = np.inf
            out[start:end] = np.sqrt(np.min(d2, axis=1))
        return out


def _load_clean_train(dataset: str, seed: int, config_root: Path) -> tuple[np.ndarray, Any]:
    cfg_dict, _ = load_layered_config(str(config_root), "oncl", dataset)
    cfg_dict = apply_overrides(
        cfg_dict,
        [
            f"seed={seed}",
            "train_noise_enable=false",
            "train_missing_segment_enable=false",
            "train_subsample_enable=false",
        ],
    )
    cfg = _build_cfg_from_mapping_strict(ve.DemoCfg, cfg_dict)
    cfg = _attach_extra_cfg_fields(cfg, cfg_dict)
    cfg.seed = int(seed)
    ds = resolve_dataset(dataset, cfg, optimize_ur5_train_only=True, ur5_backend="analytic")
    return np.asarray(ds["x_train"], dtype=np.float64), cfg


def compare(dataset: str, seed: int, ratios: list[float], config_root: Path) -> list[dict[str, Any]]:
    x_train, _ = _load_clean_train(dataset, seed, config_root)
    d = int(x_train.shape[1])
    nn_dists = _nearest_neighbor_distances(x_train)
    median_nn = float(np.median(nn_dists))
    mean_nn = float(np.mean(nn_dists))
    per_dim_sigma_knn_alpha1 = median_nn / math.sqrt(float(d))

    span = np.max(x_train, axis=0) - np.min(x_train, axis=0)
    span_l2 = float(np.linalg.norm(span))
    span_mean = float(np.mean(span))
    span_min = float(np.min(span))
    span_max = float(np.max(span))

    rows = []
    for ratio in ratios:
        span_sigma = float(ratio) * span
        span_noise_rms_l2 = float(np.linalg.norm(span_sigma))
        span_equiv_per_dim_sigma = span_noise_rms_l2 / math.sqrt(float(d))
        rows.append(
            {
                "dataset": dataset,
                "seed": seed,
                "n_train": int(len(x_train)),
                "dim": d,
                "median_nn_l2": median_nn,
                "mean_nn_l2": mean_nn,
                "knn_alpha1_target_rms_l2": median_nn,
                "knn_alpha1_per_dim_sigma": per_dim_sigma_knn_alpha1,
                "raw_span_ratio": float(ratio),
                "raw_span_noise_rms_l2": span_noise_rms_l2,
                "raw_span_equiv_per_dim_sigma": span_equiv_per_dim_sigma,
                "raw_span_noise_over_median_nn": span_noise_rms_l2 / median_nn if median_nn > 0 else float("nan"),
                "raw_span_l2": span_l2,
                "raw_span_mean": span_mean,
                "raw_span_min": span_min,
                "raw_span_max": span_max,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--datasets",
        default="12d_dual_arm_traj,3d_vz_2d_ellipse_traj,6d_surf_normal_traj",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", default="", help="comma-separated seeds; overrides --seed when set")
    parser.add_argument("--ratios", default="0.01,0.02,0.05,0.1")
    parser.add_argument("--config-root", type=Path, default=ROOT / "configs")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "outputs/analysis/noisy_data_boxplots/noise_scale_comparison_alpha1.csv",
    )
    args = parser.parse_args()

    datasets = [item.strip() for item in str(args.datasets).split(",") if item.strip()]
    ratios = [float(item.strip()) for item in str(args.ratios).split(",") if item.strip()]
    rows: list[dict[str, Any]] = []
    seeds = (
        [int(item.strip()) for item in str(args.seeds).split(",") if item.strip()]
        if str(args.seeds).strip()
        else [int(args.seed)]
    )
    for dataset in datasets:
        for seed in seeds:
            rows.extend(compare(dataset, seed, ratios, args.config_root))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else []
    with args.out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, cols)
        writer.writeheader()
        writer.writerows(rows)

    print(args.out)
    for row in rows:
        if abs(float(row["raw_span_ratio"]) - 0.05) < 1e-12:
            print(
                f"{row['dataset']} seed={row['seed']}: median_nn={row['median_nn_l2']:.6g}, "
                f"knn_alpha1_per_dim_sigma={row['knn_alpha1_per_dim_sigma']:.6g}, "
                f"span0.05_rms_l2={row['raw_span_noise_rms_l2']:.6g}, "
                f"span0.05/median_nn={row['raw_span_noise_over_median_nn']:.3g}"
            )


if __name__ == "__main__":
    main()
