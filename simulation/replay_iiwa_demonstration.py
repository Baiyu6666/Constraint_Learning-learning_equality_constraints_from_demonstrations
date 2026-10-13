#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from datasets.ros.iiwa_bag import load_iiwa_npz
from datasets.ros.transforms import quat_to_matrix_xyzw


DEFAULT_IIWA_URDF = REPO_ROOT / "datasets/assets/iiwa/iiwa14_pybullet.urdf"
DEFAULT_IIWA_XACRO = REPO_ROOT / "datasets/assets/iiwa/iiwa_ros/iiwa_description/urdf/iiwa14.urdf.xacro"
DEFAULT_IIWA_DESCRIPTION = REPO_ROOT / "datasets/assets/iiwa/iiwa_ros/iiwa_description"
DEFAULT_IIWA_ROS_ROOT = REPO_ROOT / "datasets/assets/iiwa/iiwa_ros"


def _quat_multiply_xyzw(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ax, ay, az, aw = [float(v) for v in np.asarray(a, dtype=np.float64).reshape(4)]
    bx, by, bz, bw = [float(v) for v in np.asarray(b, dtype=np.float64).reshape(4)]
    q = np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float64,
    )
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return q


def _quat_from_two_vectors(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    a = np.asarray(src, dtype=np.float64).reshape(3).copy()
    b = np.asarray(dst, dtype=np.float64).reshape(3).copy()
    a /= max(float(np.linalg.norm(a)), 1e-12)
    b /= max(float(np.linalg.norm(b)), 1e-12)
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot > 1.0 - 1e-10:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    if dot < -1.0 + 1e-10:
        axis = np.cross(a, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
        if float(np.linalg.norm(axis)) < 1e-10:
            axis = np.cross(a, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
        axis /= max(float(np.linalg.norm(axis)), 1e-12)
        return np.asarray([axis[0], axis[1], axis[2], 0.0], dtype=np.float64)
    axis = np.cross(a, b)
    q = np.asarray([axis[0], axis[1], axis[2], 1.0 + dot], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return q


def _canonical_quat(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).reshape(4).copy()
    out /= max(float(np.linalg.norm(out)), 1e-12)
    if float(out[3]) < 0.0:
        out *= -1.0
    return out


def _quat_angle_error_rad(a: np.ndarray, b: np.ndarray) -> float:
    qa = _canonical_quat(a)
    qb = _canonical_quat(b)
    dot = abs(float(np.dot(qa, qb)))
    dot = float(np.clip(dot, -1.0, 1.0))
    return 2.0 * math.acos(dot)


def ensure_default_iiwa_urdf(path: Path) -> Path:
    if path.exists():
        return path
    if path != DEFAULT_IIWA_URDF:
        raise FileNotFoundError(f"URDF not found: {path}")
    if not DEFAULT_IIWA_XACRO.exists():
        raise FileNotFoundError(
            f"Default iiwa xacro not found: {DEFAULT_IIWA_XACRO}. "
            "Clone iiwa_ros under datasets/assets/iiwa/iiwa_ros first."
        )
    xacro_bin = shutil.which("xacro")
    if not xacro_bin:
        raise RuntimeError("xacro not found in PATH; cannot generate default iiwa URDF.")

    env = os.environ.copy()
    existing = env.get("ROS_PACKAGE_PATH", "")
    env["ROS_PACKAGE_PATH"] = (
        str(DEFAULT_IIWA_ROS_ROOT) if not existing else f"{DEFAULT_IIWA_ROS_ROOT}:{existing}"
    )
    proc = subprocess.run(
        [xacro_bin, str(DEFAULT_IIWA_XACRO)],
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"xacro failed with code {proc.returncode}:\n{proc.stderr}")
    text = proc.stdout.replace("package://iiwa_description", str(DEFAULT_IIWA_DESCRIPTION.resolve()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _load_pybullet():
    try:
        import pybullet as p  # type: ignore
        import pybullet_data  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "PyBullet is unavailable. Run with the segment environment, e.g. "
            "`conda run -n segment python simulation/replay_iiwa_demonstration.py ...`."
        ) from exc
    return p, pybullet_data


class _FFmpegVideoWriter:
    def __init__(self, *, out_path: Path, width: int, height: int, fps: float) -> None:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg binary not found in PATH")
        self.out_path = Path(out_path)
        self.width = int(width)
        self.height = int(height)
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            ffmpeg,
            "-y",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{self.width}x{self.height}",
            "-r",
            f"{float(fps):.6f}",
            "-i",
            "-",
            "-an",
            "-vcodec",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(self.out_path),
        ]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def append_data(self, frame: np.ndarray) -> None:
        if self.proc.stdin is None:
            raise RuntimeError("ffmpeg stdin is unavailable")
        arr = np.asarray(frame, dtype=np.uint8)
        expected = (self.height, self.width, 3)
        if arr.shape != expected:
            raise ValueError(f"video frame has shape {arr.shape}, expected {expected}")
        self.proc.stdin.write(arr.tobytes())

    def close(self) -> None:
        if self.proc.stdin is not None:
            self.proc.stdin.close()
        self.proc.wait(timeout=30)
        if self.proc.returncode not in (0, None):
            raise RuntimeError(f"ffmpeg exited with code {self.proc.returncode}")


def _find_link_index(p, robot_id: int, link_name: str, client_id: int) -> int:
    for idx in range(p.getNumJoints(robot_id, physicsClientId=client_id)):
        info = p.getJointInfo(robot_id, idx, physicsClientId=client_id)
        if info[12].decode("utf-8") == link_name:
            return int(idx)
    raise RuntimeError(f"Link not found in URDF: {link_name}")


def _find_joint_indices(p, robot_id: int, joint_names: list[str], client_id: int) -> list[int]:
    by_name = {}
    for idx in range(p.getNumJoints(robot_id, physicsClientId=client_id)):
        info = p.getJointInfo(robot_id, idx, physicsClientId=client_id)
        by_name[info[1].decode("utf-8")] = int(idx)
    missing = [name for name in joint_names if name not in by_name]
    if missing:
        raise RuntimeError(f"Joints not found in URDF: {missing}")
    return [by_name[name] for name in joint_names]


def _draw_trace(p, points: np.ndarray, *, color: tuple[float, float, float], stride: int, client_id: int) -> None:
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[0] < 2:
        return
    step = max(1, int(stride))
    for i in range(0, pts.shape[0] - step, step):
        p.addUserDebugLine(
            pts[i].tolist(),
            pts[i + step].tolist(),
            lineColorRGB=list(color),
            lineWidth=2.0,
            lifeTime=0.0,
            physicsClientId=client_id,
        )


def _make_probe_body(
    p,
    *,
    length: float,
    radius: float,
    color_rgba: tuple[float, float, float, float],
    client_id: int,
) -> int | None:
    if length <= 1e-9:
        return None
    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=float(radius),
        length=float(length),
        rgbaColor=list(color_rgba),
        physicsClientId=client_id,
    )
    return p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual,
        basePosition=[0.0, 0.0, 0.0],
        baseOrientation=[0.0, 0.0, 0.0, 1.0],
        physicsClientId=client_id,
    )


def _update_probe_body(
    p,
    *,
    body_id: int | None,
    ee_pos: np.ndarray,
    ee_quat: np.ndarray,
    tcp_offset_ee: np.ndarray,
    client_id: int,
) -> np.ndarray:
    offset = np.asarray(tcp_offset_ee, dtype=np.float64).reshape(3)
    rot = quat_to_matrix_xyzw(ee_quat)
    tcp_pos = np.asarray(ee_pos, dtype=np.float64).reshape(3) + rot @ offset
    if body_id is None:
        return tcp_pos
    offset_norm = float(np.linalg.norm(offset))
    if offset_norm <= 1e-9:
        return tcp_pos
    center = np.asarray(ee_pos, dtype=np.float64).reshape(3) + rot @ (0.5 * offset)
    local_probe_quat = _quat_from_two_vectors(np.asarray([0.0, 0.0, 1.0]), offset)
    probe_quat = _quat_multiply_xyzw(np.asarray(ee_quat, dtype=np.float64), local_probe_quat)
    p.resetBasePositionAndOrientation(
        body_id,
        center.tolist(),
        probe_quat.tolist(),
        physicsClientId=client_id,
    )
    return tcp_pos


def _camera_target(points: np.ndarray) -> tuple[list[float], float]:
    pts = np.asarray(points, dtype=np.float64)
    if pts.size == 0:
        return [0.55, 0.0, 0.45], 1.4
    lo = np.nanmin(pts, axis=0)
    hi = np.nanmax(pts, axis=0)
    center = 0.5 * (lo + hi)
    span = max(float(np.linalg.norm(hi - lo)), 0.35)
    return center.tolist(), max(1.0, 2.2 * span)


def _nearest_indices(source_t: np.ndarray, query_t: np.ndarray) -> np.ndarray:
    source = np.asarray(source_t, dtype=np.float64).reshape(-1)
    query = np.asarray(query_t, dtype=np.float64).reshape(-1)
    if source.size == 0 or query.size == 0:
        return np.zeros((query.size,), dtype=np.int64)
    idx = np.searchsorted(source, query)
    idx0 = np.clip(idx - 1, 0, source.size - 1)
    idx1 = np.clip(idx, 0, source.size - 1)
    choose_prev = np.abs(query - source[idx0]) <= np.abs(source[idx1] - query)
    return np.where(choose_prev, idx0, idx1).astype(np.int64)


def _capture_frame(p, *, client_id: int, width: int, height: int, target: list[float], distance: float) -> np.ndarray:
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=target,
        distance=float(distance),
        yaw=45.0,
        pitch=-28.0,
        roll=0.0,
        upAxisIndex=2,
        physicsClientId=client_id,
    )
    proj = p.computeProjectionMatrixFOV(
        fov=52.0,
        aspect=float(width) / float(height),
        nearVal=0.01,
        farVal=10.0,
    )
    _, _, rgba, _, _ = p.getCameraImage(
        width=int(width),
        height=int(height),
        viewMatrix=view,
        projectionMatrix=proj,
        renderer=p.ER_BULLET_HARDWARE_OPENGL,
        physicsClientId=client_id,
    )
    arr = np.asarray(rgba, dtype=np.uint8).reshape(int(height), int(width), 4)
    return arr[:, :, :3].copy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay a real iiwa demonstration in PyBullet.")
    parser.add_argument("dataset", type=Path, help="Path to iiwa_demonstration.npz.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_IIWA_URDF)
    parser.add_argument(
        "--ee-link",
        default="iiwa_link_7",
        help="URDF link used as the recorded EE frame. The bag has iiwa_link_7->iiwa_link_ee as identity, while the upstream GitHub URDF has a 4.5cm ee offset.",
    )
    parser.add_argument("--gui", action="store_true", help="Open PyBullet GUI instead of DIRECT.")
    parser.add_argument("--stride", type=int, default=5, help="Replay every Nth recorded joint state.")
    parser.add_argument("--max-frames", type=int, default=0, help="Limit replay frames; 0 means all.")
    parser.add_argument("--realtime", action="store_true", help="Sleep according to recorded timestamps.")
    parser.add_argument("--trace-stride", type=int, default=80)
    parser.add_argument("--probe-radius", type=float, default=0.006)
    parser.add_argument("--tcp-offset", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"))
    parser.add_argument("--video-out", type=Path, default=None)
    parser.add_argument("--video-fps", type=float, default=30.0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--hold-seconds", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    p, pybullet_data = _load_pybullet()
    data = load_iiwa_npz(args.dataset)
    q = np.asarray(data["q"], dtype=np.float64)
    t_joint = np.asarray(data["t_joint"], dtype=np.float64)
    t_pose = np.asarray(data["t_pose"], dtype=np.float64)
    joint_names = [str(x) for x in data["joint_names"].tolist()]
    ee_pos_ref = np.asarray(data["ee_pos"], dtype=np.float64)
    tcp_ref = np.asarray(data["tcp_pos"] if "tcp_pos" in data else data["ee_pos"], dtype=np.float64)
    tcp_offset_ee = (
        np.asarray(args.tcp_offset, dtype=np.float64)
        if args.tcp_offset is not None
        else np.asarray(data["tcp_offset_ee"] if "tcp_offset_ee" in data else [0.0, 0.0, 0.0], dtype=np.float64)
    ).reshape(3)

    urdf = ensure_default_iiwa_urdf(args.urdf.resolve())
    client_id = p.connect(p.GUI if args.gui else p.DIRECT)
    if client_id < 0:
        raise RuntimeError("Failed to connect to PyBullet.")
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath(), physicsClientId=client_id)
        p.resetSimulation(physicsClientId=client_id)
        p.setGravity(0.0, 0.0, -9.81, physicsClientId=client_id)
        p.loadURDF("plane.urdf", basePosition=[0.0, 0.0, -0.02], physicsClientId=client_id)
        robot_id = p.loadURDF(
            str(urdf),
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            useFixedBase=True,
            flags=p.URDF_USE_INERTIA_FROM_FILE,
            physicsClientId=client_id,
        )
        joint_indices = _find_joint_indices(p, robot_id, joint_names, client_id)
        ee_link_idx = _find_link_index(p, robot_id, str(args.ee_link), client_id)
        probe_body = _make_probe_body(
            p,
            length=float(np.linalg.norm(tcp_offset_ee)),
            radius=float(args.probe_radius),
            color_rgba=(0.05, 0.55, 1.0, 0.95),
            client_id=client_id,
        )

        _draw_trace(p, ee_pos_ref, color=(1.0, 0.55, 0.05), stride=args.trace_stride, client_id=client_id)
        _draw_trace(p, tcp_ref, color=(0.05, 0.35, 1.0), stride=args.trace_stride, client_id=client_id)

        frame_indices = np.arange(0, q.shape[0], max(1, int(args.stride)), dtype=np.int64)
        if int(args.max_frames) > 0:
            frame_indices = frame_indices[: int(args.max_frames)]
        pose_lookup = _nearest_indices(t_pose, t_joint[frame_indices] if t_joint.size else frame_indices.astype(np.float64))

        target, camera_distance = _camera_target(tcp_ref)
        writer = None
        if args.video_out is not None:
            writer = _FFmpegVideoWriter(
                out_path=args.video_out,
                width=int(args.width),
                height=int(args.height),
                fps=float(args.video_fps),
            )

        ee_pos_err = []
        tcp_pos_err = []
        ee_ori_err = []
        prev_time = None
        for frame_no, idx in enumerate(frame_indices):
            qi = q[int(idx)]
            for j, joint_idx in enumerate(joint_indices):
                p.resetJointState(
                    robot_id,
                    joint_idx,
                    float(qi[j]),
                    targetVelocity=0.0,
                    physicsClientId=client_id,
                )
            p.stepSimulation(physicsClientId=client_id)
            link_state = p.getLinkState(
                robot_id,
                ee_link_idx,
                computeForwardKinematics=True,
                physicsClientId=client_id,
            )
            ee_pos = np.asarray(link_state[4], dtype=np.float64)
            ee_quat = np.asarray(link_state[5], dtype=np.float64)
            tcp_pos = _update_probe_body(
                p,
                body_id=probe_body,
                ee_pos=ee_pos,
                ee_quat=ee_quat,
                tcp_offset_ee=tcp_offset_ee,
                client_id=client_id,
            )

            pose_idx = int(pose_lookup[frame_no]) if pose_lookup.size else min(int(idx), ee_pos_ref.shape[0] - 1)
            ee_pos_err.append(float(np.linalg.norm(ee_pos - ee_pos_ref[pose_idx])))
            tcp_pos_err.append(float(np.linalg.norm(tcp_pos - tcp_ref[pose_idx])))
            if "ee_quat" in data:
                ee_ori_err.append(float(_quat_angle_error_rad(ee_quat, data["ee_quat"][pose_idx])))

            if writer is not None:
                frame = _capture_frame(
                    p,
                    client_id=client_id,
                    width=int(args.width),
                    height=int(args.height),
                    target=target,
                    distance=camera_distance,
                )
                writer.append_data(frame)
            if args.gui and args.realtime and t_joint.size:
                now = float(t_joint[int(idx)])
                if prev_time is not None:
                    time.sleep(max(0.0, min(0.1, now - prev_time)))
                prev_time = now
            elif args.gui:
                time.sleep(1.0 / max(float(args.video_fps), 1.0))

            if frame_no == 0 or (frame_no + 1) % 200 == 0:
                print(f"[replay] frame {frame_no + 1}/{len(frame_indices)} q_index={int(idx)}")

        if writer is not None:
            writer.close()
            print(f"[saved] video: {args.video_out}")

        print("[summary]")
        print(f"  dataset: {args.dataset}")
        print(f"  urdf: {urdf}")
        print(f"  ee_link: {args.ee_link}")
        print(f"  frames: {len(frame_indices)} / {q.shape[0]} recorded joint states")
        print(f"  tcp_offset_ee: {tcp_offset_ee.tolist()}")
        if ee_pos_err:
            print(f"  ee_pos_err_m mean/max: {float(np.mean(ee_pos_err)):.6g} / {float(np.max(ee_pos_err)):.6g}")
        if tcp_pos_err:
            print(f"  tcp_pos_err_m mean/max: {float(np.mean(tcp_pos_err)):.6g} / {float(np.max(tcp_pos_err)):.6g}")
        if ee_ori_err:
            print(f"  ee_ori_err_rad mean/max: {float(np.mean(ee_ori_err)):.6g} / {float(np.max(ee_ori_err)):.6g}")
        if args.gui and float(args.hold_seconds) > 0.0:
            time.sleep(float(args.hold_seconds))
    finally:
        p.disconnect(physicsClientId=client_id)


if __name__ == "__main__":
    main()
