# syntax=docker/dockerfile:1.7
FROM pytorch/pytorch:2.10.0-cuda12.8-cudnn9-runtime

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG CONTAINER_HOME=/home/learneq

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MPLBACKEND=Agg \
    CONTAINER_HOME=${CONTAINER_HOME}

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash-completion \
    build-essential \
    ca-certificates \
    ffmpeg \
    git \
    less \
    libegl1 \
    libgl1 \
    libgl1-mesa-dev \
    libglew-dev \
    libglfw3 \
    libosmesa6-dev \
    libsm6 \
    libx11-6 \
    libxext6 \
    libxfixes3 \
    libxi6 \
    libxinerama1 \
    libxrandr2 \
    libxrender1 \
    nano \
    pkg-config \
    rsync \
    tmux \
    wget \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY requirements.txt /tmp/learneq-requirements.txt
COPY smp_manifold_learning-master /tmp/smp_manifold_learning-master

RUN python -m pip install --upgrade pip setuptools wheel \
    && grep -vE '^[[:space:]]*(torch([<>=~!].*)?|-e[[:space:]]+\./smp_manifold_learning-master)[[:space:]]*$' /tmp/learneq-requirements.txt > /tmp/learneq-docker-requirements.txt \
    && python -m pip install -r /tmp/learneq-docker-requirements.txt \
    && python -m pip install --no-deps -e /tmp/smp_manifold_learning-master

RUN mkdir -p "${CONTAINER_HOME}"

CMD ["bash"]
