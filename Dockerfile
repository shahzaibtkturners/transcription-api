# ----------------------
# Stage 1: Build
# ----------------------
FROM python:3.11-slim AS build

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install build tools only for pip packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy only requirements first (for caching)
COPY requirements.txt .

# Install Python dependencies into /install
RUN pip install --upgrade pip \
    && pip install --prefix=/install --no-cache-dir -r requirements.txt

# ----------------------
# Stage 2: Runtime
# ----------------------
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    tesseract-ocr \
    libtesseract-dev \
    poppler-utils \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from build stage
COPY --from=build /install /usr/local

# Set working directory
WORKDIR /app

# Copy source code
COPY . .

# Expose port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
