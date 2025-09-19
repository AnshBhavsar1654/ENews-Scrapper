# Use a slim Python base
FROM python:3.11-slim

# Prevent interactive tzdata prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install Chromium and Chromedriver and minimal fonts
# On Debian Bookworm, the packages are chromium and chromium-driver
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    fonts-liberation \
    ca-certificates \
    tzdata \
    && rm -rf /var/lib/apt/lists/*

# Set Chrome/Driver paths for the app (matches what scraper.py expects)
ENV CHROME_BIN=/usr/bin/chromium
ENV CHROMEDRIVER_PATH=/usr/bin/chromedriver
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Create workdir
WORKDIR /app

# Copy and install Python dependencies first (better layer caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && \
    pip install -r /app/requirements.txt

# Copy the app code
COPY . /app

# Expose port (Render will still use $PORT)
EXPOSE 8001

# Start the FastAPI app; use $PORT provided by Render
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8001}"]