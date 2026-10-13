from __future__ import annotations

import argparse
import csv
import glob
import json
import sys
from pathlib import Path
from types import SimpleNamespace

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import numpy as np
import torch

from evaluation.eval_runner import run_eval_metrics
from evaluation.evaluator import _workspace_pose_analytic_target_embed
from methods import vae as vae_base
from methods.models.autoencoder import VariationalAutoEncoder
from methods.models.mlp import MLP
from methods.utils.projection import project_points_with_steps_numpy
from planning.kinematics import workspace_embed_for_eval, wrap_workspace_pose_rpy_np


def _cfg_obj(mapping: dict) -> SimpleNamespace:
    return SimpleNamespace(**mapping)


def _projector_params(cfg: SimpleNamespace) -> tuple[int, float, int]:
    p = getattr(cfg, "projector", {}) or {}
    return (
        int(p.get("steps", 100)),
        float(p.get("alpha", 0.3)),
        int(p.get("min_steps", 0)),
    )


def _load_model(ckpt_path: Path, device: str):
    ckpt = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    method = str(ckpt.get("method", "") or "")
    if method == "vae":
        model = VariationalAutoEncoder(
            in_dim=int(ckpt["in_dim"]),
            latent_dim=int(ckpt["latent_dim"]),
            hidden=tuple(int(v) for v in ckpt["hidden_dims"]),
        ).to(device)
        model.load_state_dict(ckpt["model_state"])
        model.eval()
        field_model = vae_base.VAEProjectorField(model).to(device)
        field_model.eval()
        return method, field_model, np.asarray(ckpt["x_train"], dtype=np.float32), ckpt["cfg"]

    method = "oncl"
    model = MLP(
        in_dim=int(ckpt["in_dim"]),
        hidden=int(ckpt["hidden"]),
        depth=int(ckpt["depth"]),
        out_dim=int(ckpt["constraint_dim"]),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return method, model, np.asarray(ckpt["x_train"], dtype=np.float32), ckpt["cfg"]


def _summarize(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "max": float(np.max(values)),
    }


def _split_stats(raw: np.ndarray, cfg) -> dict[str, float | dict[str, float]]:
    emb = workspace_embed_for_eval("6d_surf_normal_traj", raw)[:, :6].astype(np.float32)
    tgt = _workspace_pose_analytic_target_embed(emb, cfg)
    pos = np.abs(emb[:, 2] - tgt[:, 2]).astype(np.float32)
    ori_vec = np.linalg.norm(emb[:, 3:6] - tgt[:, 3:6], axis=1).astype(np.float32)
    cos = np.clip(np.sum(emb[:, 3:6] * tgt[:, 3:6], axis=1), -1.0, 1.0)
    ori_deg = np.degrees(np.arccos(cos)).astype(np.float32)
    total = np.sqrt(pos**2 + ori_vec**2).astype(np.float32)
    pos_sq = float(np.sum(pos.astype(np.float64) ** 2))
    ori_sq = float(np.sum(ori_vec.astype(np.float64) ** 2))
    denom = max(pos_sq + ori_sq, 1e-12)
    return {
        "total": _summarize(total),
        "pos_z": _summarize(pos),
        "ori_vec": _summarize(ori_vec),
        "ori_deg": _summarize(ori_deg),
        "pos_sq_fraction": pos_sq / denom,
        "ori_sq_fraction": ori_sq / denom,
    }


def analyze_run(summary_path: Path, device: str) -> dict:
    with summary_path.open("r", encoding="utf-8") as f:
        summary = json.load(f)
    ckpt_path = Path(summary["ckpt_path"])
    method, model, x_train, cfg_map = _load_model(ckpt_path, device)
    cfg_map = dict(cfg_map)
    cfg_map["device"] = device
    cfg = _cfg_obj(cfg_map)
    steps, alpha, min_steps = _projector_params(cfg)

    if method == "vae":
        def project_fn(_model, x0, _eps_stop):
            with torch.no_grad():
                xt = torch.from_numpy(x0.astype(np.float32)).to(device)
                proj = model.project_tensor(xt).detach().cpu().numpy().astype(np.float32)
            return proj, np.ones((len(proj),), dtype=np.float32)
    else:
        def project_fn(_model, x0, eps_stop):
            return project_points_with_steps_numpy(
                _model,
                x0,
                device=device,
                proj_steps=steps,
                proj_alpha=alpha,
                proj_min_steps=min_steps,
                f_abs_stop=float(eps_stop),
            )

    metrics, eval_cfg, artifacts = run_eval_metrics(
        cfg=cfg,
        method_key=method,
        dataset_name=str(summary["dataset"]),
        model=model,
        x_train=x_train,
        project_fn=project_fn,
        embed_fn=lambda q: workspace_embed_for_eval(str(summary["dataset"]), q),
        postprocess_fn=wrap_workspace_pose_rpy_np,
    )
    before = _split_stats(np.asarray(artifacts["x_eval"], dtype=np.float32), eval_cfg)
    after = _split_stats(np.asarray(artifacts["proj"], dtype=np.float32), eval_cfg)
    return {
        "method": method,
        "dataset": str(summary["dataset"]),
        "seed": int(cfg_map.get("seed", -1)),
        "summary_path": str(summary_path),
        "proj_manifold_dist": float(metrics["proj_manifold_dist"]),
        "before": before,
        "after": after,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("summary_glob", nargs="+")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    rows = []
    results = []
    for pattern in args.summary_glob:
        for name in sorted(glob.glob(pattern)):
            result = analyze_run(Path(name), str(args.device))
            results.append(result)
            after = result["after"]
            rows.append(
                {
                    "method": result["method"],
                    "dataset": result["dataset"],
                    "seed": result["seed"],
                    "proj_manifold_dist": result["proj_manifold_dist"],
                    "after_total_mean": after["total"]["mean"],
                    "after_pos_z_mean": after["pos_z"]["mean"],
                    "after_ori_vec_mean": after["ori_vec"]["mean"],
                    "after_ori_deg_mean": after["ori_deg"]["mean"],
                    "after_pos_sq_fraction": after["pos_sq_fraction"],
                    "after_ori_sq_fraction": after["ori_sq_fraction"],
                }
            )
    print(json.dumps(results, indent=2))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        csv_path = out.with_suffix(".csv")
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"saved: {out}")
        print(f"saved: {csv_path}")


if __name__ == "__main__":
    main()
