FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    VVCS_DEVICE=cpu \
    VVCS_OUTPUT_DIR=/app/outputs \
    HF_HOME=/app/.cache/huggingface \
    GRADIO_SERVER_NAME=0.0.0.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 appuser

WORKDIR /app
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY --chown=appuser:appuser . .
RUN mkdir -p /app/outputs /app/data /app/checkpoints /app/.cache/huggingface \
    && chown -R appuser:appuser /app

USER appuser
EXPOSE 7860
HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD curl --fail http://127.0.0.1:7860/ || exit 1

CMD ["python", "app.py", "--host", "0.0.0.0", "--port", "7860"]
