# Build from repo root: docker build -f simulator/Dockerfile .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.30 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1

# Install only the simulator's dependency slice, pinned from uv.lock.
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-emit-project --no-hashes --no-dev --extra simulator -o /tmp/req.txt \
 && uv pip install --system --no-cache -r /tmp/req.txt

COPY simulator/simulate.py .

ENTRYPOINT ["python", "simulate.py"]
CMD ["--broker", "mosquitto", "--port", "1883", "--machines", "3"]
