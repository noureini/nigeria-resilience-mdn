# Fully pinned, platform-independent environment for the replication package.
#
#   docker build -t nigeria-resilience .
#   docker run --rm -v "$PWD/outputs:/app/outputs" nigeria-resilience
#
# The image runs `python reproduce.py`; results appear in ./outputs. Put the survey
# files in data/raw/ghs/ and data/raw/epar/ (see data/README.md) before building.
# Use this route on Apple-silicon Macs (TensorFlow 2.8 has no arm64 wheels).
FROM --platform=linux/amd64 python:3.9.25-slim-bookworm

COPY --from=ghcr.io/astral-sh/uv:0.11.8 /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies first (cached layer), then the project itself.
COPY pyproject.toml uv.lock .python-version README.md ./
RUN uv sync --frozen --no-install-project --no-dev

COPY . .
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
CMD ["python", "reproduce.py"]
