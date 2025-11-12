# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

########################################
# Build ARG/ENV
########################################

WORKDIR /app

ARG PYTHON_VERSION=3.13
ARG HOME=/root
ARG CACHE_HOME=/.cache
ARG CONFIG_HOME=/.config
ARG VIRTUAL_ENV=/.venv

ENV CACHE_HOME=${CACHE_HOME}
ENV CONFIG_HOME=${CONFIG_HOME}
ENV XDG_CACHE_HOME=${CACHE_HOME}
ENV TORCH_HOME=${CACHE_HOME}/torch
ENV HF_HOME=${CACHE_HOME}/huggingface

ENV UV_CACHE_DIR=${CACHE_HOME}/uv
ENV UV_PROJECT_ENVIRONMENT=${VIRTUAL_ENV}
ENV UV_PYTHON=${PYTHON_VERSION}
ENV UV_COMPILE_BYTECODE=1 
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=manual
ENV UV_PYTHON_PREFERENCE=only-managed

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PYTHONPATH="${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages"
ENV LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# OS deps
RUN --mount=type=cache,target=/var/cache/apt \
  --mount=type=cache,target=/var/lib/apt/lists \
  apt-get update && apt-get install -y --no-install-recommends curl ca-certificates

# Python/uv deps
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/
RUN uv python install ${PYTHON_VERSION}

########################################
# Build stages and preloading
########################################
FROM base AS build-deps

COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
  uv sync --locked --no-install-project --no-dev

FROM build-deps AS build-project

COPY . .
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
  uv sync --locked --no-dev

FROM build-project AS build-models

RUN uv run whisperx --version
RUN --mount=type=cache,target=${CACHE_HOME} \
  uv run ./preload-models.py

########################################
# Setup for final build
########################################
FROM base AS production

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV MEDIA_FOLDER="/media"

# Copy over app, cache, venv
COPY --from=build-models /app .
COPY --from=build-models ${CACHE_HOME} ${CACHE_HOME}  
COPY --from=build-models ${VIRTUAL_ENV} ${VIRTUAL_ENV}

EXPOSE 7860

CMD ["python", "app.py"]
