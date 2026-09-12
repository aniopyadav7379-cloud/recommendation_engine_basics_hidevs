# --- Build stage: install deps into a venv for a slimmer final image ---
FROM python:3.12-slim AS builder

WORKDIR /app
COPY pyproject.toml requirements.txt ./
RUN python -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -r requirements.txt

COPY src/ src/
RUN /opt/venv/bin/pip install --no-cache-dir .

# --- Runtime stage ---
FROM python:3.12-slim

RUN useradd --create-home appuser
WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY data/ data/

ENV PATH="/opt/venv/bin:$PATH" \
    RECO_DATA_DIR=/app/data \
    RECO_LOG_LEVEL=INFO

USER appuser
EXPOSE 8000

# Liveness-style check: confirms the process can actually answer HTTP
# requests, independent of whether storage is currently reachable (that
# distinction is what /health/ready is for -- see reco_engine/api.py).
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')" || exit 1

CMD ["uvicorn", "reco_engine.api:app", "--host", "0.0.0.0", "--port", "8000"]
