FROM python:3.12-slim
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY simulate.py .

ENTRYPOINT ["python", "simulate.py"]
CMD ["--broker", "mosquitto", "--port", "1883", "--machines", "3"]
