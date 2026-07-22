# Build from repo root: docker build -f inference/Dockerfile .
FROM python:3.12-slim

# uv: fast, reproducible installs straight from the committed lock.
COPY --from=ghcr.io/astral-sh/uv:0.11.30 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1

# Install only the inference service's dependency slice, pinned from uv.lock.
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-emit-project --no-hashes --no-dev --extra inference -o /tmp/req.txt \
 && uv pip install --system --no-cache -r /tmp/req.txt

COPY ml/ ml/
COPY inference/ inference/

# bake a freshly trained model into the image (synthetic data, deterministic);
# .git is outside the build context, so pass the commit for the model manifest:
#   docker build --build-arg EDGESENSE_GIT_COMMIT=$(git rev-parse --short HEAD) ...
ARG EDGESENSE_GIT_COMMIT=""
ENV EDGESENSE_GIT_COMMIT=$EDGESENSE_GIT_COMMIT
RUN python ml/train.py

EXPOSE 8800
HEALTHCHECK --interval=10s --timeout=3s --start-period=20s --retries=5 \
  CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8800/healthz')"]

CMD ["python", "-m", "uvicorn", "inference.server:app", "--host", "0.0.0.0", "--port", "8800"]
