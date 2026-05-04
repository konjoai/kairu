# Multi-stage build — small final image, single layer for the runtime.
# Targets: linux/amd64 (default), linux/arm64 (built in CI via buildx).
#
#   docker build -t kairu .
#   docker run --rm -p 8000:8000 kairu serve --host 0.0.0.0 --port 8000

ARG PYTHON_VERSION=3.11

# ─── builder ──────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# Install build deps. We do not need apt packages — kairu is pure Python +
# numpy wheels — but we keep this stage isolated so the runtime image stays slim.
RUN pip install --upgrade pip wheel build

COPY pyproject.toml README.md ./
COPY kairu ./kairu

# Build the wheel into /build/dist
RUN python -m build --wheel --outdir /build/dist

# ─── runtime ──────────────────────────────────────────────────────────────
FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

# Non-root user. The container exposes HTTP only — no need for root.
RUN groupadd --system --gid 1001 kairu \
 && useradd  --system --gid kairu --uid 1001 --create-home --shell /usr/sbin/nologin kairu

WORKDIR /app

COPY --from=builder /build/dist/*.whl /tmp/

# Server + Redis extras pulled in (the redis client is small and the entrypoint
# accepts --redis URL flags — having the lib available is a strict win).
RUN pip install --no-cache-dir /tmp/kairu-*.whl 'kairu[server,redis]' \
 && rm /tmp/kairu-*.whl \
 && python -c "import kairu; print('kairu', kairu.__version__)"

USER kairu

EXPOSE 8000

# Default: serve mock model on 0.0.0.0:8000. Override with `docker run` args.
ENTRYPOINT ["kairu"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000", "--model", "mock"]

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        sys.exit(0) if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2).status == 200 else sys.exit(1)" \
        || exit 1
