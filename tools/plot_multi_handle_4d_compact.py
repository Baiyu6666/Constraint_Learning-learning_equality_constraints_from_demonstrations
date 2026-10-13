#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
from types import SimpleNamespace

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from datasets.constraint_datasets import (
    _double_torus_extra_handle_mesh,
    lift_double_torus_extra_handle_to_4d,
)
from experiments.config_loader import load_layered_config
from experiments.dataset_resolve import resolve_dataset
from methods.models.mlp import MLP
from methods.utils.projection import project_points_with_steps_numpy


def _parse_figsize(text: str) -> tuple[float, float]:
    parts = [p.strip() for p in str(text).lower().replace("x", ",").split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError("--figsize must look like WIDTH,HEIGHT, for example 1.95,1.62")
    return float(parts[0]), float(parts[1])


def _axis_equal_3d(ax, pts: np.ndarray, *, zoom: float = 1.0) -> None:
    p = np.asarray(pts, dtype=np.float32)
    mins = np.nanmin(p, axis=0)
    maxs = np.nanmax(p, axis=0)
    center = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins))
    radius = max(radius / max(float(zoom), 1e-6), 1e-6)
    ax.set_xlim(float(center[0] - radius), float(center[0] + radius))
    ax.set_ylim(float(center[1] - radius), float(center[1] + radius))
    ax.set_zlim(float(center[2] - 0.78 * radius), float(center[2] + 0.78 * radius))


def _trajectory_slices(x_train: np.ndarray, traj_count: int) -> list[np.ndarray]:
    x = np.asarray(x_train, dtype=np.float32)
    n_traj = max(1, min(int(traj_count), len(x)))
    base = len(x) // n_traj
    out = []
    start = 0
    for i in range(n_traj):
        end = start + base
        if i == n_traj - 1:
            end = len(x)
        if end > start:
            out.append(x[start:end])
        start = end
    return out


def _project_learned_samples(
    *,
    ckpt_path: str,
    seed_points: np.ndarray,
    rng: np.random.Generator,
    n_samples: int,
    noise_std: float,
    device: str,
    batch_size: int,
) -> np.ndarray:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_map = dict(ckpt.get("cfg", {}))
    cfg_map["device"] = str(device)
    model = MLP(
        in_dim=int(ckpt["in_dim"]),
        hidden=int(ckpt["hidden"]),
        depth=int(ckpt["depth"]),
        out_dim=int(ckpt["constraint_dim"]),
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(str(device))
    model.eval()

    seeds = np.asarray(seed_points, dtype=np.float32)
    if len(seeds) > int(n_samples):
        seeds = seeds[rng.choice(len(seeds), size=int(n_samples), replace=False)]
    if float(noise_std) > 0.0:
        seeds = seeds + rng.normal(scale=float(noise_std), size=seeds.shape).astype(np.float32)

    proj_parts = []
    bs = max(1, int(batch_size))
    for start in range(0, len(seeds), bs):
        chunk = seeds[start : start + bs].astype(np.float32)
        proj, _steps = project_points_with_steps_numpy(
            model,
            chunk,
            device=str(device),
            proj_steps=int(cfg_map.get("proj_steps", 250)),
            proj_alpha=float(cfg_map.get("proj_alpha", 0.04)),
            proj_min_steps=int(cfg_map.get("proj_min_steps", 20)),
            f_abs_stop=None,
        )
        proj_parts.append(proj.astype(np.float32))
    learned = np.concatenate(proj_parts, axis=0)
    return learned[np.isfinite(learned).all(axis=1)].astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Compact paper plot for the 4D multi-handle dataset.")
    ap.add_argument("--out", default="outputs/analysis/multi_handle_4d_preview/multi_handle_4d_true_surface_demos_compact.png")
    ap.add_argument("--figsize", default="1.95,1.58")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--max-faces", type=int, default=4200)
    ap.add_argument("--surface-mode", choices=("mesh", "scatter"), default="mesh")
    ap.add_argument("--surface-source", choices=("mesh", "grid"), default="mesh")
    ap.add_argument("--surface-points", type=int, default=24000)
    ap.add_argument("--traj-lines", type=int, default=20)
    ap.add_argument("--demo-point-size", type=float, default=0.35)
    ap.add_argument("--demo-point-alpha", type=float, default=0.10)
    ap.add_argument("--demo-line-alpha", type=float, default=0.48)
    ap.add_argument("--surface-alpha", type=float, default=0.42)
    ap.add_argument("--learned-ckpt", default="")
    ap.add_argument("--learned-samples", type=int, default=26000)
    ap.add_argument("--learned-noise-std", type=float, default=0.10)
    ap.add_argument("--learned-batch-size", type=int, default=4096)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-demos", action="store_true")
    ap.add_argument("--zoom", type=float, default=1.0)
    ap.add_argument("--elev", type=float, default=25.0)
    ap.add_argument("--azim", type=float, default=-45.0)
    ap.add_argument("--hide-axis-labels", action="store_true")
    ap.add_argument("--no-colorbar", action="store_true")
    ap.add_argument("--legend-y", type=float, default=0.91)
    ap.add_argument("--colorbar-x", type=float, default=0.28)
    ap.add_argument("--colorbar-y", type=float, default=0.12)
    ap.add_argument("--colorbar-width", type=float, default=0.36)
    ap.add_argument("--colorbar-height", type=float, default=0.035)
    args = ap.parse_args()

    cfg_map, _ = load_layered_config(os.path.join(_PROJECT_ROOT, "configs"), "oncl", "multi_handle_4d")
    if args.seed is not None:
        cfg_map["seed"] = int(args.seed)
    cfg = SimpleNamespace(**cfg_map)
    ds = resolve_dataset("multi_handle_4d", cfg)
    x_train = np.asarray(ds["x_train"], dtype=np.float32)
    grid4 = np.asarray(ds["grid"], dtype=np.float32)

    verts3, faces = _double_torus_extra_handle_mesh(
        nx=int(getattr(cfg, "extra_handle_mesh_nx", 112)),
        ny=int(getattr(cfg, "extra_handle_mesh_ny", 96)),
        nz=int(getattr(cfg, "extra_handle_mesh_nz", 88)),
        bridge_radius=float(getattr(cfg, "extra_handle_bridge_radius", 0.22)),
    )
    verts4 = lift_double_torus_extra_handle_to_4d(verts3)
    rng = np.random.default_rng(int(getattr(cfg, "seed", 0)) + 4009)
    faces_plot = np.asarray(faces, dtype=np.int64)
    if len(faces_plot) > int(args.max_faces):
        faces_plot = faces_plot[rng.choice(len(faces_plot), size=int(args.max_faces), replace=False)]

    norm = mpl.colors.Normalize(vmin=float(np.percentile(verts4[:, 3], 2.0)), vmax=float(np.percentile(verts4[:, 3], 98.0)))
    cmap = mpl.colormaps.get_cmap("viridis")
    face_x4 = np.mean(verts4[faces_plot, 3], axis=1)
    face_colors = cmap(norm(face_x4))
    face_colors[:, 3] = float(args.surface_alpha)
    learned4 = None
    if str(args.learned_ckpt).strip():
        learned4 = _project_learned_samples(
            ckpt_path=str(args.learned_ckpt),
            seed_points=verts4,
            rng=rng,
            n_samples=int(args.learned_samples),
            noise_std=float(args.learned_noise_std),
            device=str(args.device),
            batch_size=int(args.learned_batch_size),
        )

    trajs = _trajectory_slices(x_train, int(getattr(cfg, "traj_count", 64)))
    if len(trajs) > int(args.traj_lines):
        keep = np.linspace(0, len(trajs) - 1, int(args.traj_lines), dtype=int)
        line_trajs = [trajs[int(i)] for i in keep]
    else:
        line_trajs = trajs

    fig_size = _parse_figsize(args.figsize)
    with plt.rc_context(
        {
            "font.size": 6,
            "axes.labelsize": 6,
            "xtick.labelsize": 5,
            "ytick.labelsize": 5,
            "legend.fontsize": 5,
        }
    ):
        fig = plt.figure(figsize=fig_size)
        ax = fig.add_subplot(111, projection="3d")
        if learned4 is not None:
            if len(learned4) > int(args.surface_points):
                learned_plot = learned4[rng.choice(len(learned4), size=int(args.surface_points), replace=False)]
            else:
                learned_plot = learned4
            ax.scatter(
                learned_plot[:, 0],
                learned_plot[:, 1],
                learned_plot[:, 2],
                s=0.55,
                c=learned_plot[:, 3],
                cmap=cmap,
                norm=norm,
                alpha=0.44,
                depthshade=False,
                label=r"Learned surface in $(x_1,x_2,x_3)$",
                zorder=1,
            )
        elif str(args.surface_mode) == "scatter":
            surf = verts4 if str(args.surface_source) == "mesh" else grid4
            if len(surf) > int(args.surface_points):
                surf = surf[rng.choice(len(surf), size=int(args.surface_points), replace=False)]
            ax.scatter(
                surf[:, 0],
                surf[:, 1],
                surf[:, 2],
                s=0.55,
                c=surf[:, 3],
                cmap=cmap,
                norm=norm,
                alpha=0.42,
                depthshade=False,
                label=r"4D surface in $(x_1,x_2,x_3)$",
                zorder=1,
            )
        else:
            poly = Poly3DCollection(
                verts3[faces_plot],
                facecolors=face_colors,
                edgecolor=(0, 0, 0, 0.018),
                linewidth=0.025,
                label=r"4D surface in $(x_1,x_2,x_3)$",
            )
            ax.add_collection3d(poly)

        if not bool(args.no_demos):
            for i, tr in enumerate(line_trajs):
                ax.plot(
                    tr[:, 0],
                    tr[:, 1],
                    tr[:, 2],
                    color="#e11d48",
                    linewidth=0.36,
                    alpha=float(args.demo_line_alpha),
                    label="Demonstrations" if i == 0 else None,
                    zorder=5,
                )
            ax.scatter(
                x_train[:, 0],
                x_train[:, 1],
                x_train[:, 2],
                s=float(args.demo_point_size),
                c="#111827",
                alpha=float(args.demo_point_alpha),
                depthshade=False,
                zorder=6,
            )

        axis_pts = learned4[:, :3] if learned4 is not None and len(learned4) > 0 else verts3
        _axis_equal_3d(ax, axis_pts, zoom=float(args.zoom))
        ax.view_init(elev=float(args.elev), azim=float(args.azim))
        ax.set_xlabel("" if bool(args.hide_axis_labels) else "$x_1$", labelpad=0)
        ax.set_ylabel("" if bool(args.hide_axis_labels) else "$x_2$", labelpad=0)
        ax.set_zlabel("")
        z_top = float(np.max(verts3[:, 2]))
        if not bool(args.hide_axis_labels):
            ax.text(float(np.max(verts3[:, 0]) + 0.08), float(np.max(verts3[:, 1]) + 0.02), z_top + 0.04, "$x_3$", fontsize=6)
        ax.tick_params(labelsize=5, pad=-2)
        ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        ax.zaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=3))
        ax.set_proj_type("persp")
        try:
            ax.dist = 7.0 / max(float(args.zoom), 1e-6)
        except Exception:
            pass
        handles = [
            Patch(
                facecolor=cmap(0.62),
                edgecolor="none",
                alpha=0.85,
                label=r"Learned surface in $(x_1,x_2,x_3)$" if learned4 is not None else r"4D surface in $(x_1,x_2,x_3)$",
            ),
        ]
        if not bool(args.no_demos):
            handles.append(Line2D([0], [0], color="#e11d48", linewidth=0.55, alpha=0.75, label="Demonstrations"))
        ax.legend(
            handles=handles,
            loc="upper left",
            bbox_to_anchor=(0.03, float(args.legend_y)),
            ncol=1,
            fontsize=5,
            frameon=False,
            borderaxespad=0.0,
            handlelength=1.1,
            handletextpad=0.3,
        )
        if not bool(args.no_colorbar):
            sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
            sm.set_array([])
            cax = inset_axes(
                ax,
                width="100%",
                height="100%",
                loc="lower left",
                bbox_to_anchor=(
                    float(args.colorbar_x),
                    float(args.colorbar_y),
                    float(args.colorbar_width),
                    float(args.colorbar_height),
                ),
                bbox_transform=ax.transAxes,
                borderpad=0.0,
            )
            cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
            cb.set_ticks([norm.vmin, norm.vmax])
            cb.ax.set_title(r"$x_4$", fontsize=5, pad=1)
            cb.ax.tick_params(labelsize=4, length=1.5, pad=1)
            cb.outline.set_linewidth(0.35)
        fig.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)

        out_path = os.path.abspath(args.out)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.0)
        plt.close(fig)
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
