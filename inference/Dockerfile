# Build from repo root: docker build -f inference/Dockerfile .
FROM python:3.12-slim
WORKDIR /app

COPY inference/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ml/ ml/
COPY inference/ inference/

# bake a freshly trained model into the image (synthetic data, deterministic)
RUN python ml/train.py

EXPOSE 8800
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8800/healthz')"]

CMD ["python", "-m", "uvicorn", "inference.server:app", "--host", "0.0.0.0", "--port", "8800"]
