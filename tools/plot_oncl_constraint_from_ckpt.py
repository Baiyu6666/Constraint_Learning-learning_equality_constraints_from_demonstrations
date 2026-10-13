#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import numpy as np
import torch

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.dataset_resolve import resolve_dataset
from methods.models.mlp import MLP
from plotting.oncl_plots import _plot_constraint_surface_paper_3d


def _parse_figsize(text: str) -> tuple[float, float] | None:
    t = str(text).strip().lower()
    if not t:
        return None
    parts = [p for p in t.replace("x", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--figsize must look like WIDTH,HEIGHT, for example 1.72,1.44")
    return float(parts[0]), float(parts[1])


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot an ONCL 3D learned constraint from a saved checkpoint.")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--compact", action="store_true")
    ap.add_argument("--no-legend", action="store_true")
    ap.add_argument("--figsize", default="")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    dataset = str(ckpt["dataset"])
    cfg_dict = dict(ckpt.get("cfg", {}))
    cfg_dict["device"] = str(args.device)
    cfg_dict["show_3d_plot"] = False
    cfg_dict["paper_plot_compact"] = bool(args.compact)
    cfg_dict["paper_plot_show_legend"] = not bool(args.no_legend)
    fig_size = _parse_figsize(args.figsize)
    if fig_size is not None:
        cfg_dict["paper_plot_figsize"] = fig_size
    cfg = SimpleNamespace(**cfg_dict)

    x_train = np.asarray(ckpt["x_train"], dtype=np.float32)
    model = MLP(
        in_dim=int(ckpt["in_dim"]),
        hidden=int(ckpt["hidden"]),
        depth=int(ckpt["depth"]),
        out_dim=int(ckpt["constraint_dim"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(str(args.device))
    model.eval()

    ds = resolve_dataset(dataset, cfg)
    labels = tuple(ds.get("axis_labels", [f"x{i + 1}" for i in range(x_train.shape[1])]))
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    _plot_constraint_surface_paper_3d(
        model=model,
        x_train=x_train,
        out_path=args.out,
        axis_labels=(str(labels[0]), str(labels[1]), str(labels[2])),
        cfg=cfg,
        intersection_points=None,
        dataset_name=dataset,
    )
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
