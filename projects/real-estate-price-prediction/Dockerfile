FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (Chromium for Selenium scraping)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data/raw data/interim data/processed models logs

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
# Use system chromium in Docker (skip webdriver-manager download)
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver

# Expose ports
EXPOSE 8501 8000

# Healthcheck for the default CMD (Streamlit dashboard on 8501). This image
# is also run as the FastAPI service via docker-compose.yml, which overrides
# both `command:` AND `healthcheck:` per-service (curl against
# GET /health on port 8000 for the api service) — this Dockerfile-level
# HEALTHCHECK is the sane default for running the image standalone (`docker
# run`) with the default CMD, not just "container is running" with no real
# readiness signal.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

# Default command: Streamlit dashboard
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
