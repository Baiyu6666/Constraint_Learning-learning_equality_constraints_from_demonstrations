#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.constraint_datasets import _knot_spline_union_ctrl, _knot_spline_union_curve


def _parse_figsize(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in str(text).lower().replace("x", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--figsize must look like WIDTH,HEIGHT, for example 1.95,1.10")
    return float(parts[0]), float(parts[1])


def _axis_equal_3d(ax, pts: np.ndarray, *, pad_ratio: float = 0.05) -> None:
    p = np.asarray(pts, dtype=np.float32)
    mins = np.nanmin(p, axis=0)
    maxs = np.nanmax(p, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins)) * (1.0 + float(pad_ratio))
    radius = max(radius, 1e-6)
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - radius), float(center[2] + radius))


def _match_height(path: str, *, target_height: int) -> None:
    if int(target_height) <= 0:
        return
    try:
        from PIL import Image
    except Exception:
        return
    im = Image.open(path).convert("RGBA")
    if im.size[1] == int(target_height):
        return
    if im.size[1] < int(target_height):
        out = Image.new("RGBA", (im.size[0], int(target_height)), (255, 255, 255, 255))
        y = (int(target_height) - im.size[1]) // 2
        out.alpha_composite(im, (0, y))
        out.save(path)
        return
    excess = im.size[1] - int(target_height)
    top = max(0, excess // 2)
    im.crop((0, top, im.size[0], top + int(target_height))).save(path)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compact true-only plot for linked_curves_3d.")
    ap.add_argument("--out", default="outputs/analysis/linked_curves_3d_learned_constraint/linked_curves_3d_true_only_compact.png")
    ap.add_argument("--figsize", default="1.95,1.10")
    ap.add_argument("--target-height", type=int, default=362)
    ap.add_argument("--n", type=int, default=1200)
    ap.add_argument("--elev", type=float, default=28.0)
    ap.add_argument("--azim", type=float, default=-42.0)
    ap.add_argument("--control-point-size", type=float, default=13.0)
    ap.add_argument("--hide-control-points", action="store_true")
    args = ap.parse_args()

    u = np.linspace(0.0, 1.0, int(args.n), endpoint=True, dtype=np.float32)
    curve_a = _knot_spline_union_curve(0, u)
    curve_b = _knot_spline_union_curve(1, u)
    ctrl_b = _knot_spline_union_ctrl()
    pts = np.concatenate([curve_a, curve_b, ctrl_b], axis=0)

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with plt.rc_context(
        {
            "font.size": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
        }
    ):
        fig = plt.figure(figsize=_parse_figsize(args.figsize))
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(
            curve_a[:, 0],
            curve_a[:, 1],
            curve_a[:, 2],
            color="#2563eb",
            linewidth=1.55,
            alpha=0.98,
            label="Trefoil curve",
        )
        ax.plot(
            curve_b[:, 0],
            curve_b[:, 1],
            curve_b[:, 2],
            color="#f97316",
            linewidth=1.55,
            alpha=0.98,
            label="Control-point spline",
        )
        if not bool(args.hide_control_points):
            ax.scatter(
                ctrl_b[:, 0],
                ctrl_b[:, 1],
                ctrl_b[:, 2],
                s=float(args.control_point_size),
                color="#f97316",
                edgecolor="#111827",
                linewidth=0.35,
                alpha=0.98,
                depthshade=False,
                label=None,
            )
        _axis_equal_3d(ax, pts, pad_ratio=0.06)
        ax.view_init(elev=float(args.elev), azim=float(args.azim))
        ax.set_xlabel("$x_1$", labelpad=0)
        ax.set_ylabel("$x_2$", labelpad=0)
        ax.set_zlabel("")
        maxs = np.max(pts, axis=0)
        ax.text(float(maxs[0] + 0.03), float(maxs[1] + 0.03), float(maxs[2] + 0.03), "$x_3$", fontsize=6)
        ax.tick_params(labelsize=5, pad=-2)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.zaxis.set_major_locator(MaxNLocator(nbins=3))
        ax.set_proj_type("persp")
        try:
            ax.dist = 6.6
        except Exception:
            pass
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.03, 0.98),
            ncol=1,
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.25,
            handletextpad=0.35,
        )
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)

    _match_height(out_path, target_height=int(args.target_height))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
