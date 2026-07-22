# Build from repo root: docker build -f mcp_server/Dockerfile .
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.30 /uv /uvx /bin/
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1

# Install only the MCP server's dependency slice, pinned from uv.lock.
COPY pyproject.toml uv.lock ./
RUN uv export --frozen --no-emit-project --no-hashes --no-dev --extra mcp-server -o /tmp/req.txt \
 && uv pip install --system --no-cache -r /tmp/req.txt

COPY mcp_server/server.py .

ENV EDGESENSE_MCP_TRANSPORT=streamable-http \
    EDGESENSE_MCP_HOST=0.0.0.0

EXPOSE 8900
CMD ["python", "server.py"]
