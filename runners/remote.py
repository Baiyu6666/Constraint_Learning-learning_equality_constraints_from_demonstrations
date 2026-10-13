#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REMOTE_HOST = os.environ.get("LEARNEQ_REMOTE_HOST", "matterhorn")
DEFAULT_REMOTE_HOME = os.environ.get("LEARNEQ_REMOTE_HOME", "/home/bpeng")
DEFAULT_REMOTE_DIR = os.environ.get("LEARNEQ_REMOTE_DIR", f"{DEFAULT_REMOTE_HOME}/LearnEqConstraints")
DEFAULT_REMOTE_RUNS_DIR = os.environ.get("LEARNEQ_REMOTE_RUNS_DIR", f"{DEFAULT_REMOTE_HOME}/runs")
DEFAULT_REMOTE_IMAGE = os.environ.get("LEARNEQ_REMOTE_IMAGE", "learneq:cuda12.8")
DEFAULT_REMOTE_DOCKER_HOME = os.environ.get(
    "LEARNEQ_REMOTE_DOCKER_HOME",
    f"{DEFAULT_REMOTE_HOME}/.cache/learneq-docker-home",
)
DEFAULT_REMOTE_CONTAINER_HOME = os.environ.get("LEARNEQ_REMOTE_CONTAINER_HOME", "/home/learneq")


def shell_join(args: list[str] | tuple[str, ...]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in args)


def run_command(args: list[str], *, cwd: Path | None = None, dry_run: bool = False) -> int:
    print("+", shell_join(args), flush=True)
    if dry_run:
        return 0
    return subprocess.run(args, cwd=str(cwd) if cwd is not None else None).returncode


def trim_remainder(args: list[str]) -> list[str]:
    if args and args[0] == "--":
        return args[1:]
    return args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Sync this repository to a remote server and run the existing "
            "LearnEqConstraints entrypoints inside a Docker container."
        ),
        epilog=(
            "Examples: python -m runners.remote run-benchmark --methods oncl "
            "--datasets 3d_torus_surface_traj --outdir /runs/smoke"
        ),
    )
    parser.add_argument("--remote-host", default=DEFAULT_REMOTE_HOST)
    parser.add_argument("--remote-home", default=DEFAULT_REMOTE_HOME)
    parser.add_argument("--remote-dir", default=DEFAULT_REMOTE_DIR)
    parser.add_argument("--remote-runs-dir", default=DEFAULT_REMOTE_RUNS_DIR)
    parser.add_argument("--remote-image", default=DEFAULT_REMOTE_IMAGE)
    parser.add_argument("--remote-docker-home", default=DEFAULT_REMOTE_DOCKER_HOME)
    parser.add_argument("--remote-container-home", default=DEFAULT_REMOTE_CONTAINER_HOME)
    parser.add_argument("--remote-python", default="python")
    parser.add_argument("--remote-no-sync", action="store_true", help="Skip rsync and run the current remote checkout.")
    parser.add_argument("--remote-dry-run", action="store_true", help="Print commands without executing them.")
    parser.add_argument("--remote-no-tty", action="store_true", help="Do not force SSH TTY allocation.")
    parser.add_argument("--remote-no-sudo", action="store_true", help="Run docker without sudo.")
    parser.add_argument("--remote-build-image", action="store_true", help="Build the Docker image on the remote host before running.")
    parser.add_argument("--remote-build-no-cache", action="store_true", help="Pass --no-cache to docker build.")
    parser.add_argument(
        "--remote-extra-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment variable for docker run. Can be repeated.",
    )
    parser.add_argument(
        "--remote-volume",
        action="append",
        default=[],
        metavar="HOST:CONTAINER[:MODE]",
        help="Extra docker volume mount. Can be repeated.",
    )
    parser.add_argument(
        "--remote-docker-arg",
        action="append",
        default=[],
        metavar="ARG",
        help="Extra raw argument passed to docker run. Can be repeated.",
    )

    parser.add_argument(
        "command",
        choices=["run-one", "run-benchmark", "run-parallel", "module", "script", "fetch"],
        help=(
            "run-one: python -m runners.run_one; "
            "run-benchmark: python -m runners.run_benchmark; "
            "run-parallel: python -m runners.run_parallel; "
            "module/script: arbitrary repo entrypoint; "
            "fetch: rsync remote results back"
        ),
    )
    return parser


def docker_base_args(args: argparse.Namespace) -> list[str]:
    docker_cmd = ["docker"] if args.remote_no_sudo else ["sudo", "docker"]
    return docker_cmd


def docker_env_args(args: argparse.Namespace) -> list[str]:
    env_pairs = [
        f"HOME={args.remote_container_home}",
        "PYTHONPATH=/workspace:/workspace/smp_manifold_learning-master",
        "WANDB_DIR=/runs/wandb",
        "MPLBACKEND=Agg",
        "PYTHONUNBUFFERED=1",
        "USER=learneq",
        "LOGNAME=learneq",
        f"XDG_CACHE_HOME={args.remote_container_home}/.cache",
        f"TORCHINDUCTOR_CACHE_DIR={args.remote_container_home}/.cache/torchinductor",
    ]
    env_pairs.extend(args.remote_extra_env)

    out: list[str] = []
    for pair in env_pairs:
        if "=" not in pair:
            raise ValueError(f"--remote-extra-env must be KEY=VALUE, got {pair!r}")
        out.extend(["-e", pair])
    return out


def docker_volume_args(args: argparse.Namespace) -> list[str]:
    volumes = [
        f"{args.remote_dir}:/workspace",
        f"{args.remote_runs_dir}:/runs",
        f"{args.remote_docker_home}:{args.remote_container_home}",
    ]
    volumes.extend(args.remote_volume)

    out: list[str] = []
    for volume in volumes:
        out.extend(["-v", volume])
    return out


def forwarded_command(args: argparse.Namespace) -> list[str]:
    if args.command == "run-one":
        return [args.remote_python, "-m", "runners.run_one", *trim_remainder(args.forwarded_args)]
    if args.command == "run-benchmark":
        return [args.remote_python, "-m", "runners.run_benchmark", *trim_remainder(args.forwarded_args)]
    if args.command == "run-parallel":
        return [args.remote_python, "-m", "runners.run_parallel", *trim_remainder(args.forwarded_args)]
    if args.command == "module":
        return [args.remote_python, "-m", args.module_name, *trim_remainder(args.module_args)]
    if args.command == "script":
        return [args.remote_python, args.script_path, *trim_remainder(args.script_args)]
    raise ValueError(f"unsupported command: {args.command}")


def ssh_command(args: argparse.Namespace, remote_command: str) -> list[str]:
    cmd = ["ssh"]
    if not args.remote_no_tty:
        cmd.append("-tt")
    cmd.extend([args.remote_host, remote_command])
    return cmd


def ensure_remote_dirs(args: argparse.Namespace) -> int:
    remote_command = shell_join([
        "mkdir",
        "-p",
        args.remote_dir,
        args.remote_runs_dir,
        args.remote_docker_home,
    ])
    return run_command(ssh_command(args, remote_command), cwd=PROJECT_ROOT, dry_run=args.remote_dry_run)


def sync_project(args: argparse.Namespace) -> int:
    ignore_file = PROJECT_ROOT / ".remoteignore"
    rsync_cmd = [
        "rsync",
        "-az",
        "--delete",
        f"--exclude-from={ignore_file}",
        "./",
        f"{args.remote_host}:{args.remote_dir.rstrip('/')}/",
    ]
    return run_command(rsync_cmd, cwd=PROJECT_ROOT, dry_run=args.remote_dry_run)


def build_remote_image(args: argparse.Namespace) -> int:
    docker_build = [
        *docker_base_args(args),
        "build",
        "-f",
        "docker/Dockerfile",
        "-t",
        args.remote_image,
    ]
    if args.remote_build_no_cache:
        docker_build.append("--no-cache")
    docker_build.append(".")
    remote_command = (
        f"cd {shlex.quote(args.remote_dir)} && "
        f"DOCKER_BUILDKIT=1 {shell_join(docker_build)}"
    )
    return run_command(ssh_command(args, remote_command), cwd=PROJECT_ROOT, dry_run=args.remote_dry_run)


def build_remote_run_command(args: argparse.Namespace) -> str:
    inner_cmd = shell_join(forwarded_command(args))
    docker_prefix = [
        *docker_base_args(args),
        "run",
        "--rm",
        "--gpus",
        "all",
        "--shm-size=8g",
    ]
    docker_suffix = [
        "--user",
        "__REMOTE_UID_GID__",
        *docker_env_args(args),
        *docker_volume_args(args),
        *args.remote_docker_arg,
        "-w",
        "/workspace",
        args.remote_image,
        "bash",
        "-lc",
        inner_cmd,
    ]
    docker_command = shell_join(docker_prefix + docker_suffix)
    docker_command = docker_command.replace("__REMOTE_UID_GID__", '"$(id -u):$(id -g)"')
    setup = shell_join([
        "mkdir",
        "-p",
        args.remote_runs_dir,
        args.remote_docker_home,
    ])
    return f"{setup} && cd {shlex.quote(args.remote_dir)} && {docker_command}"


def run_remote(args: argparse.Namespace) -> int:
    return run_command(
        ssh_command(args, build_remote_run_command(args)),
        cwd=PROJECT_ROOT,
        dry_run=args.remote_dry_run,
    )


def fetch_results(args: argparse.Namespace) -> int:
    remote_path = args.remote_path.rstrip("/")
    local_path = Path(args.local_path)
    if not args.remote_dry_run:
        local_path.mkdir(parents=True, exist_ok=True)
    rsync_cmd = [
        "rsync",
        "-az",
        f"{args.remote_host}:{remote_path}/",
        f"{str(local_path).rstrip('/')}/",
    ]
    return run_command(rsync_cmd, cwd=PROJECT_ROOT, dry_run=args.remote_dry_run)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, forwarded = parser.parse_known_args(sys.argv[1:] if argv is None else argv)
    forwarded = trim_remainder(forwarded)
    os.chdir(PROJECT_ROOT)

    if args.command in {"run-one", "run-benchmark", "run-parallel"}:
        args.forwarded_args = forwarded
    elif args.command == "module":
        if not forwarded:
            parser.error("module requires a module name, for example: module benchmark.plot_paper_benchmarks")
        args.module_name = forwarded[0]
        args.module_args = forwarded[1:]
    elif args.command == "script":
        if not forwarded:
            parser.error("script requires a repository script path")
        args.script_path = forwarded[0]
        args.script_args = forwarded[1:]
    elif args.command == "fetch":
        if len(forwarded) != 2:
            parser.error("fetch requires REMOTE_PATH and LOCAL_PATH")
        args.remote_path = forwarded[0]
        args.local_path = forwarded[1]

    if args.command == "fetch":
        return fetch_results(args)

    if not args.remote_no_sync:
        status = ensure_remote_dirs(args)
        if status != 0:
            return status
        status = sync_project(args)
        if status != 0:
            return status

    if args.remote_build_image:
        status = build_remote_image(args)
        if status != 0:
            return status

    return run_remote(args)


if __name__ == "__main__":
    raise SystemExit(main())
