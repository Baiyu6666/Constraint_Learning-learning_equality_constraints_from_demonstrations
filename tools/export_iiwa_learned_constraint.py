from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def _jsonable(value: object) -> object:
    try:
        import numpy as np

        if isinstance(value, np.ndarray):
            return value.astype(float).tolist()
        if isinstance(value, np.generic):
            return value.item()
    except Exception:
        pass
    if hasattr(value, "detach") and hasattr(value, "cpu"):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return value


def _load_checkpoint(model_path: Path) -> dict[str, object] | None:
    try:
        import torch

        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    except Exception:
        return None
    return ckpt if isinstance(ckpt, dict) else None


def _load_checkpoint_metadata(model_path: Path) -> dict[str, object]:
    ckpt = _load_checkpoint(model_path)
    if ckpt is None:
        return {}
    out: dict[str, object] = {}
    for key in ("dataset", "method", "in_dim", "constraint_dim", "hidden", "depth"):
        if key in ckpt:
            value = ckpt[key]
            if isinstance(value, (str, int, float, bool)) or value is None:
                out[key] = value
    return out


def _load_surface_normalization(surface_model_dir: Path) -> dict[str, object]:
    model_path = surface_model_dir / "calibrated_tcp1845_neural_1d_surface_mlp.pt"
    metadata_path = surface_model_dir / "calibrated_tcp1845_neural_1d_surface.json"
    if not model_path.is_file():
        raise FileNotFoundError(f"surface normalization checkpoint not found: {model_path}")

    try:
        import torch
    except Exception as exc:
        raise RuntimeError("torch is required to read surface normalization parameters") from exc

    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    required = (
        "changing_axis_xy",
        "invariant_axis_xy",
        "u_mean",
        "u_scale",
        "v_mean",
        "v_scale",
        "z_mean",
        "z_scale",
    )
    missing = [key for key in required if key not in ckpt]
    if missing:
        raise KeyError(f"surface normalization checkpoint missing keys: {missing}")

    surface_name = surface_model_dir.name
    tcp_offset_link7_m = None
    tcp_offset_ee_m = None
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            surface_name = str(metadata.get("name", surface_name))
            tcp_offset_link7_m = metadata.get("tcp_offset_link7_m")
            tcp_offset_ee_m = metadata.get("tcp_offset_ee_m")
        except Exception:
            pass

    return {
        "type": "surface_model_uvz_affine",
        "source": surface_name,
        "world_frame": "world",
        "position_input_columns": ["u_normalized", "v_normalized", "z_normalized"],
        "formula": {
            "u_world": "dot([x_world, y_world], changing_axis_xy)",
            "v_world": "dot([x_world, y_world], invariant_axis_xy)",
            "u_normalized": "(u_world - u_mean) / u_scale",
            "v_normalized": "(v_world - v_mean) / v_scale",
            "z_normalized": "(z_world - z_mean) / z_scale",
        },
        "changing_axis_xy": _jsonable(ckpt["changing_axis_xy"]),
        "invariant_axis_xy": _jsonable(ckpt["invariant_axis_xy"]),
        "u_mean": _jsonable(ckpt["u_mean"]),
        "u_scale": _jsonable(ckpt["u_scale"]),
        "v_mean": _jsonable(ckpt["v_mean"]),
        "v_scale": _jsonable(ckpt["v_scale"]),
        "z_mean": _jsonable(ckpt["z_mean"]),
        "z_scale": _jsonable(ckpt["z_scale"]),
        "tcp_offset_link7_m": tcp_offset_link7_m,
        "tcp_offset_ee_m": tcp_offset_ee_m,
    }


def _copy_or_augment_checkpoint(model_path: Path, model_out: Path, normalization: dict[str, object]) -> None:
    ckpt = _load_checkpoint(model_path)
    if ckpt is None:
        shutil.copy2(model_path, model_out)
        return
    try:
        import torch

        exported = dict(ckpt)
        exported["input_format"] = "[u_normalized, v_normalized, z_normalized, tool_z_x, tool_z_y, tool_z_z]"
        exported["input_normalization"] = normalization
        torch.save(exported, model_out)
    except Exception:
        shutil.copy2(model_path, model_out)


def export_model(
    *,
    model_path: Path,
    demo_dir: Path,
    surface_model_dir: Path,
    model_name: str,
    target_subdir: str,
    include_surface_model: bool,
) -> Path:
    if not model_path.is_file():
        raise FileNotFoundError(f"model not found: {model_path}")
    if not surface_model_dir.is_dir():
        raise FileNotFoundError(f"surface normalization dir not found: {surface_model_dir}")
    if not demo_dir.is_dir():
        raise FileNotFoundError(f"demonstration dir not found: {demo_dir}")

    out_dir = demo_dir / target_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    normalization = _load_surface_normalization(surface_model_dir)
    model_out = out_dir / model_name
    _copy_or_augment_checkpoint(model_path, model_out, normalization)

    surface_out = out_dir / "surface_model"
    if include_surface_model:
        if surface_out.exists():
            shutil.rmtree(surface_out)
        shutil.copytree(surface_model_dir, surface_out)
    elif surface_out.exists():
        shutil.rmtree(surface_out)

    metadata = _load_checkpoint_metadata(model_path)
    manifest = {
        "model_file": model_out.name,
        "input_format": "[u_normalized, v_normalized, z_normalized, tool_z_x, tool_z_y, tool_z_z]",
        "orientation_format": "world-frame unit tool_z; rotation about tool_z is ignored",
        "input_normalization": normalization,
        "note": "World xyz normalization is embedded here and in the exported checkpoint; no true surface model is exported by default.",
        **metadata,
    }
    if include_surface_model:
        manifest["surface_model_dir"] = "surface_model"
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export a minimal learned real-iiwa constraint into a demonstration folder.")
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument(
        "--surface-model-dir",
        type=Path,
        default=Path("/home/baiyu/iiwa_share/calibrated_tcp1845_neural_1d_surface_model"),
    )
    parser.add_argument("--model-name", default="learned_constraint_model.pt")
    parser.add_argument("--target-subdir", default="learned_constraint")
    parser.add_argument("--include-surface-model", action="store_true")
    args = parser.parse_args()

    out_dir = export_model(
        model_path=args.model,
        demo_dir=args.demo_dir,
        surface_model_dir=args.surface_model_dir,
        model_name=str(args.model_name),
        target_subdir=str(args.target_subdir),
        include_surface_model=bool(args.include_surface_model),
    )
    print(f"saved: {out_dir}")


if __name__ == "__main__":
    main()
