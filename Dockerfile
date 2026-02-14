# Voicebox TTS Server
# CUDA 12.9 + Python 3.12 on Ubuntu 24.04
#
# Build:
#   DOCKER_BUILDKIT=1 docker build -t voicebox .
#   DOCKER_BUILDKIT=1 docker build --build-arg CUDA=0 -t voicebox-cpu .
#
# Run:
#   docker compose up -d
#
# syntax=docker/dockerfile:1.4

ARG CUDA=1

# --- Base stage ---
FROM nvidia/cuda:12.9.1-runtime-ubuntu24.04 AS base-cuda
FROM ubuntu:24.04 AS base-cpu

# --- Pick base based on CUDA arg --
FROM base-cuda AS base-1
FROM base-cpu AS base-0
FROM base-${CUDA} AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN --mount=type=cache,target=/var/cache/apt,sharing=locked \
    --mount=type=cache,target=/var/lib/apt,sharing=locked \
    apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-venv \
    python3-dev \
    python3-pip \
    libsndfile1 \
    ffmpeg \
    curl \
    sox \
    && rm -rf /var/lib/apt/lists/*

# --- Dependencies stage (cached layer) ---
FROM base AS deps

ARG CUDA
WORKDIR /app

# Create virtual environment outside /app to survive volume mount
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY backend/requirements.txt ./requirements.txt

RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    if [ "$CUDA" = "1" ]; then \
        pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cu124 && \
        pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu124; \
    else \
        pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu && \
        pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu; \
    fi

# Source is volume-mounted at runtime
ENV HF_HOME=/app/data/huggingface
ENV PATH="/opt/venv/bin:$PATH"

EXPOSE 17493

HEALTHCHECK --interval=60s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:17493/health || exit 1

ENTRYPOINT ["/opt/venv/bin/python3", "-m", "backend.main"]
CMD ["--host", "0.0.0.0", "--port", "17493", "--data-dir", "/app/data"]
