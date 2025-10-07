# Use a slim image and install minimal deps required by opencv
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/
RUN pip install --no-cache-dir -r requirements.txt

# copy app and (optionally) model
COPY ./app /app/app
COPY ./models /app/models

WORKDIR /app/app

EXPOSE 7001

# start uvicorn (1 worker is safer with TF)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7001", "--workers", "1"]
