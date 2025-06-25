
FROM python:3.11-slim

# Environment setup
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    OLLAMA_HOST=0.0.0.0

# Set working directory
WORKDIR /app

# Install required system packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    curl \
    sudo \
    ca-certificates \
    gnupg2 \
    lsb-release \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | bash

# Copy Python dependencies
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Create non-root user
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# Expose FastAPI and Ollama ports
EXPOSE 8000 11434

# Healthcheck for FastAPI
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# CMD: Start Ollama server and FastAPI app
CMD ["bash", "-c", "ollama serve & sleep 3 && ollama pull llama3.2:3b && uvicorn app:app --host 0.0.0.0 --port 8000"]

# docker run -p 8000:8000 -p 11434:11434 --name ollama-fastapi-app ollama-fastapi-app