# syntax=docker/dockerfile:1
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04 AS base

########################################
# Build ARG/ENV
########################################

WORKDIR /app

ARG PYTHON_VERSION=3.13
ARG CACHE_HOME=/.cache
ARG VIRTUAL_ENV=/.venv
ARG MODELS_FOLDER=/models

ENV CACHE_HOME=${CACHE_HOME}
ENV TORCH_HOME=${MODELS_FOLDER}/torch
ENV HF_HOME=${MODELS_FOLDER}/huggingface

ENV UV_CACHE_DIR=${CACHE_HOME}/uv
ENV UV_PROJECT_ENVIRONMENT=${VIRTUAL_ENV}
ENV UV_PYTHON=${PYTHON_VERSION}
ENV UV_PYTHON_DOWNLOADS=manual
ENV UV_PYTHON_PREFERENCE=only-managed
ENV UV_LINK_MODE=copy

ENV PATH="${VIRTUAL_ENV}/bin:${PATH}"
ENV PYTHONPATH="${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages"
ENV LD_LIBRARY_PATH="${VIRTUAL_ENV}/lib/python${PYTHON_VERSION}/site-packages/nvidia/cudnn/lib:${LD_LIBRARY_PATH}"

# Python/uv deps
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN --mount=type=cache,target=${UV_CACHE_DIR} \
  uv python install ${PYTHON_VERSION}

########################################
# Build stages and preloading
########################################
FROM base AS build

# Install dependencies
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv sync --locked --no-install-project --no-dev

# Sync the project
COPY . .
RUN --mount=type=cache,target=${UV_CACHE_DIR} \
  uv sync --locked --no-dev

########################################
# Setup for final build
########################################
FROM base AS production

ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV MEDIA_FOLDER="/media"

# Copy over app, cache, venv
COPY --from=build /app .
COPY --from=build ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Make sure nvidia is detected
RUN sed -i '2invidia-smi' /opt/nvidia/nvidia_entrypoint.sh

EXPOSE 7860

CMD ["python", "app.py"]
