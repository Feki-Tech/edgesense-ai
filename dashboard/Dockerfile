# Build from repo root: docker build -f dashboard/Dockerfile .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.30 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1

# Install only the dashboard's dependency slice, pinned from uv.lock.
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-emit-project --no-hashes --no-dev --extra dashboard -o /tmp/req.txt \
 && uv pip install --system --no-cache -r /tmp/req.txt

COPY dashboard/app.py .

EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
