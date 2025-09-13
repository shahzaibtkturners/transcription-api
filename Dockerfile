FROM python:3.11-slim AS base

# Prevents Python from writing pyc files & enables unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install only what’s necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy only requirements first (better layer caching)
COPY requirements.txt .

# Upgrade pip to latest & install deps
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose FastAPI port
EXPOSE 8000

# Run FastAPI with uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
