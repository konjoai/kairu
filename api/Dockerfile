# kairu evaluation API — slim, non-root, deployable to Render / Fly / GKE.
#
#   docker build -f api/Dockerfile -t kairu-eval .
#   docker run --rm -p 8000:8000 kairu-eval

ARG PYTHON_VERSION=3.11

FROM python:${PYTHON_VERSION}-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PORT=8000

RUN groupadd --system --gid 1001 kairu \
 && useradd  --system --gid kairu --uid 1001 --create-home --shell /usr/sbin/nologin kairu

WORKDIR /app

COPY api/requirements.txt /app/api/requirements.txt
RUN pip install --no-cache-dir -r /app/api/requirements.txt

# kairu library — no extras needed, the API only depends on kairu.evaluation
# which is pure stdlib + numpy.  Server extras come from api/requirements.txt.
COPY pyproject.toml README.md /app/
COPY kairu /app/kairu
RUN pip install --no-cache-dir --no-deps . \
 && python -c "import kairu, api.main; print('kairu', kairu.__version__, 'eval-api ready')"

COPY api /app/api

USER kairu

EXPOSE 8000

# Render (and most PaaS) inject $PORT — honour it; fall back to 8000.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

HEALTHCHECK --interval=15s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request,sys,os; \
        port=os.environ.get('PORT','8000'); \
        sys.exit(0) if urllib.request.urlopen(f'http://127.0.0.1:{port}/health', timeout=2).status == 200 else sys.exit(1)" \
        || exit 1
