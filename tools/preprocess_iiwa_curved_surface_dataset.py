from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.real_iiwa.registry import real_iiwa_neural_surface_z_normal


def _points_in_polygon(points: np.ndarray, polygon: np.ndarray) -> np.ndarray:
    x = points[:, 0]
    y = points[:, 1]
    poly_x = polygon[:, 0]
    poly_y = polygon[:, 1]
    inside = np.zeros((len(points),), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        yi = poly_y[i]
        yj = poly_y[j]
        xi = poly_x[i]
        xj = poly_x[j]
        crosses = ((yi > y) != (yj > y)) & (
            x < (xj - xi) * (y - yi) / max(float(yj - yi), 1e-12) + xi
        )
        inside ^= crosses
        j = i
    return inside


def _uniform_xy_in_polygon(polygon: np.ndarray, n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    lo = polygon.min(axis=0)
    hi = polygon.max(axis=0)
    out: list[np.ndarray] = []
    while sum(len(x) for x in out) < n:
        need = n - sum(len(x) for x in out)
        cand = rng.uniform(lo, hi, size=(max(need * 3, 256), 2)).astype(np.float32)
        keep = cand[_points_in_polygon(cand, polygon)]
        if len(keep):
            out.append(keep[:need])
    return np.concatenate(out, axis=0)[:n].astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a uniformly resampled real iiwa curved-surface train set.")
    parser.add_argument("--n", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260719)
    parser.add_argument("--dataset", default="real_iiwa_curved_surface_position")
    parser.add_argument(
        "--model-json",
        type=Path,
        default=Path("datasets/real_iiwa/surface_models/calibrated_tcp1845_neural_1d_surface_model/calibrated_tcp1845_neural_1d_surface.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/real_iiwa/tasks/curved_surface_position/processed/train_sets/tcp1845_uniform_surface_1200_v001.npz"),
    )
    args = parser.parse_args()

    meta = json.loads(args.model_json.read_text())
    polygon = np.asarray(meta["data_support_polygon"], dtype=np.float32)
    xy = _uniform_xy_in_polygon(polygon, int(args.n), int(args.seed))
    z, normal = real_iiwa_neural_surface_z_normal(str(args.dataset), xy[:, 0], xy[:, 1])
    x = np.concatenate([xy, z.reshape(-1, 1).astype(np.float32)], axis=1).astype(np.float32)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        x=x,
        surface_normal=normal.astype(np.float32),
        source_polygon=polygon.astype(np.float32),
        seed=np.asarray([int(args.seed)], dtype=np.int64),
    )
    print(f"saved: {args.out}")
    print(f"x shape: {x.shape}")
    print(f"xyz min: {x.min(axis=0)}")
    print(f"xyz max: {x.max(axis=0)}")


if __name__ == "__main__":
    main()
