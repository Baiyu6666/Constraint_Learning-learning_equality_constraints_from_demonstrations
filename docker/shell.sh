#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$PWD}"
RUNS_DIR="${RUNS_DIR:-$HOME/runs}"
DOCKER_HOME="${DOCKER_HOME:-$HOME/.cache/learneq-docker-home}"
IMAGE="${IMAGE:-learneq:cuda12.8}"
CONTAINER_HOME="${CONTAINER_HOME:-/home/learneq}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
DOCKER_SUDO="${DOCKER_SUDO:-}"

mkdir -p "$RUNS_DIR" "$DOCKER_HOME"

docker_cmd=()
if [[ -n "$DOCKER_SUDO" ]]; then
    docker_cmd+=("$DOCKER_SUDO")
fi
docker_cmd+=("$DOCKER_BIN")

"${docker_cmd[@]}" run --rm -it --gpus all \
    --user "$(id -u):$(id -g)" \
    --shm-size=8g \
    -e HOME="$CONTAINER_HOME" \
    -e PYTHONPATH=/workspace:/workspace/smp_manifold_learning-master \
    -e WANDB_DIR=/runs/wandb \
    -e MPLBACKEND=Agg \
    -e PYTHONUNBUFFERED=1 \
    -e USER=learneq \
    -e LOGNAME=learneq \
    -e XDG_CACHE_HOME="$CONTAINER_HOME/.cache" \
    -e TORCHINDUCTOR_CACHE_DIR="$CONTAINER_HOME/.cache/torchinductor" \
    -v "$PROJECT_DIR:/workspace" \
    -v "$RUNS_DIR:/runs" \
    -v "$DOCKER_HOME:$CONTAINER_HOME" \
    -w /workspace \
    "$IMAGE" bash
