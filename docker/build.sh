#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-learneq:cuda12.8}"
CONTAINER_HOME="${CONTAINER_HOME:-/home/learneq}"
DOCKER_BIN="${DOCKER_BIN:-docker}"
DOCKER_SUDO="${DOCKER_SUDO:-}"

docker_cmd=()
if [[ -n "$DOCKER_SUDO" ]]; then
    docker_cmd+=("$DOCKER_SUDO")
fi
docker_cmd+=("$DOCKER_BIN")

DOCKER_BUILDKIT=1 "${docker_cmd[@]}" build \
    --progress=plain \
    -f docker/Dockerfile \
    --build-arg "CONTAINER_HOME=$CONTAINER_HOME" \
    -t "$IMAGE" .
