from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def _select_even_trajectories(n_traj: int, count: int) -> np.ndarray:
    if count >= n_traj:
        return np.arange(n_traj, dtype=np.int64)
    if count <= 0:
        raise ValueError("count must be positive")
    idx = np.rint(np.linspace(0, n_traj - 1, int(count))).astype(np.int64)
    idx = np.unique(idx)
    if len(idx) < count:
        missing = int(count) - len(idx)
        candidates = np.setdiff1d(np.arange(n_traj, dtype=np.int64), idx, assume_unique=True)
        idx = np.sort(np.concatenate([idx, candidates[:missing]]))
    return idx.astype(np.int64)


def _select_trajectories(n_traj: int, count: int, strategy: str, explicit_ids: np.ndarray | None = None) -> np.ndarray:
    mode = str(strategy).strip()
    if explicit_ids is not None:
        ids = np.asarray(explicit_ids, dtype=np.int64).reshape(-1)
        if len(ids) == 0:
            raise ValueError("explicit trajectory id list is empty")
        bad = ids[(ids < 0) | (ids >= int(n_traj))]
        if len(bad):
            raise ValueError(f"trajectory ids out of range 0..{int(n_traj) - 1}: {bad.astype(int).tolist()}")
        return np.unique(ids).astype(np.int64)
    if mode == "even":
        return _select_even_trajectories(n_traj, count)
    if mode == "first":
        if count <= 0:
            raise ValueError("count must be positive")
        return np.arange(min(int(count), int(n_traj)), dtype=np.int64)
    raise ValueError(f"unsupported strategy {strategy!r}")


def _subset_npz(
    *,
    input_npz: Path,
    output_npz: Path,
    trajectory_count: int,
    strategy: str,
    explicit_ids: np.ndarray | None = None,
) -> dict[str, object]:
    with np.load(input_npz, allow_pickle=False) as data:
        arrays = {k: np.asarray(data[k]) for k in data.files}

    offsets = np.asarray(arrays["trajectory_offsets"], dtype=np.int64)
    n_traj = max(0, len(offsets) - 1)
    selected_traj = _select_trajectories(n_traj, int(trajectory_count), strategy, explicit_ids=explicit_ids)

    pieces: list[np.ndarray] = []
    out_offsets = [0]
    for traj_i in selected_traj:
        start = int(offsets[int(traj_i)])
        end = int(offsets[int(traj_i) + 1])
        idx = np.arange(start, end, dtype=np.int64)
        pieces.append(idx)
        out_offsets.append(out_offsets[-1] + len(idx))
    keep_idx = np.concatenate(pieces).astype(np.int64) if pieces else np.zeros((0,), dtype=np.int64)
    out_offsets_arr = np.asarray(out_offsets, dtype=np.int64)

    n = int(arrays["x"].shape[0])
    per_sample_keys = {
        k for k, v in arrays.items()
        if np.asarray(v).ndim >= 1 and int(np.asarray(v).shape[0]) == n
    }
    output: dict[str, np.ndarray] = {}
    for key, value in arrays.items():
        arr = np.asarray(value)
        if key in per_sample_keys:
            output[key] = arr[keep_idx]
        else:
            output[key] = arr

    output["source_x"] = arrays["source_x"] if "source_x" in arrays else arrays["x"]
    output["source_index"] = keep_idx.astype(np.int64)
    output["trajectory_offsets"] = out_offsets_arr
    output["source_subset_npz"] = np.asarray(str(input_npz))
    output["selected_trajectory_ids"] = selected_traj.astype(np.int64)
    subset_rule = "explicit_trajectory_ids_from_source_order" if explicit_ids is not None else f"{strategy}_trajectory_ids_from_source_order"
    output["subset_rule"] = np.asarray(subset_rule)

    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_npz, **output)
    return {
        "input": str(input_npz),
        "output": str(output_npz),
        "source_trajectory_count": int(n_traj),
        "selected_trajectory_count": int(len(selected_traj)),
        "strategy": "explicit" if explicit_ids is not None else str(strategy),
        "selected_trajectory_ids": selected_traj.astype(int).tolist(),
        "source_points": int(n),
        "output_points": int(len(keep_idx)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build trajectory-count subsets from the oracle-controller real-iiwa dataset.")
    parser.add_argument(
        "--input-npz",
        type=Path,
        default=Path(
            "datasets/real_iiwa/tasks/RealKuka/processed/train_sets/"
            "oracle_controller_demonstrations_20260720_actual_toolz_arc3mm_v001.npz"
        ),
    )
    parser.add_argument("--counts", default="4,8,16,28")
    parser.add_argument("--strategy", choices=["even", "first"], default="even")
    parser.add_argument("--trajectory-ids", default="", help="Comma-separated explicit source trajectory ids. Overrides --counts.")
    parser.add_argument("--name-suffix", default="", help="Output stem suffix when --trajectory-ids is used, e.g. random20_seed20260721.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/real_iiwa/tasks/RealKuka/processed/train_sets"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("datasets/real_iiwa/tasks/RealKuka/figures/oracle_controller_demonstrations_20260720_traj_subset_summary.json"),
    )
    args = parser.parse_args()

    explicit_ids = np.asarray([int(x.strip()) for x in str(args.trajectory_ids).split(",") if x.strip()], dtype=np.int64)
    counts = [int(len(explicit_ids))] if len(explicit_ids) else [int(x.strip()) for x in str(args.counts).split(",") if x.strip()]
    records = []
    for count in counts:
        if len(explicit_ids):
            suffix = str(args.name_suffix or f"explicit{count}")
            stem = f"oracle_controller_demonstrations_20260720_actual_toolz_arc3mm_{suffix}_v001"
        elif args.strategy == "even":
            stem = f"oracle_controller_demonstrations_20260720_actual_toolz_arc3mm_traj{count}_v001"
        else:
            stem = f"oracle_controller_demonstrations_20260720_actual_toolz_arc3mm_first{count}_v001"
        out = args.output_dir / f"{stem}.npz"
        records.append(
            _subset_npz(
                input_npz=args.input_npz,
                output_npz=out,
                trajectory_count=count,
                strategy=args.strategy,
                explicit_ids=explicit_ids if len(explicit_ids) else None,
            )
        )

    args.summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.summary_json.write_text(json.dumps({"subsets": records}, indent=2), encoding="utf-8")
    print(json.dumps({"subsets": records}, indent=2))
    print(f"saved: {args.summary_json}")


if __name__ == "__main__":
    main()
