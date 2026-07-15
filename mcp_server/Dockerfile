FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server.py .

ENV EDGESENSE_MCP_TRANSPORT=streamable-http \
    EDGESENSE_MCP_HOST=0.0.0.0

EXPOSE 8900
CMD ["python", "server.py"]
