# =============================================================================
# Polyglot-LLM-Bench — Dockerfile
# =============================================================================
# Build:  docker build -t polyglot-bench .
# Run:    docker run --env-file .env -v ./results:/app/results polyglot-bench
# =============================================================================

# ── Stage 1: Builder ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS builder

WORKDIR /build

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: Runtime ─────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

LABEL maintainer="Polyglot-LLM-Bench Team"
LABEL description="Multi-Language LLM Benchmark via OpenRouter API"

# Create non-root user
RUN groupadd --gid 1000 bench && \
    useradd --uid 1000 --gid bench --create-home bench

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project files
COPY . .

# Create output directories
RUN mkdir -p results .cache && \
    chown -R bench:bench /app

USER bench

# Health check: verify Python and imports work
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "from polyglot_bench.config import Settings; print('OK')" || exit 1

ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python", "benchmark_runner.py"]
CMD ["run", "--dry-run"]
