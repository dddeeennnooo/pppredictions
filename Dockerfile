# syntax=docker/dockerfile:1

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    HOST=0.0.0.0 \
    PORT=10000 \
    FOOTBALL_PREDICTOR_DB=/app/storage/football_predictor.db

WORKDIR /app

COPY requirements.txt ./
RUN python -m pip install --requirement requirements.txt

RUN groupadd --gid 1000 app \
    && useradd --uid 1000 --gid app --create-home --shell /bin/bash app \
    && mkdir -p /app/storage \
    && chown -R app:app /app

COPY --chown=app:app . .

USER app

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:' + os.getenv('PORT', '10000') + '/api/health', timeout=3)"

CMD ["python", "-m", "src.webapp.start"]
