# Lightweight Python base image
FROM python:3.11-slim

# Environment optimizations for small memory footprint
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install Chromium and Chromedriver (for Selenium) and minimal fonts/graphics deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    fonts-liberation \
    libnss3 \
    libxss1 \
    libgbm1 \
    && rm -rf /var/lib/apt/lists/*

# Let our app and Selenium know where Chromium and Chromedriver live
ENV CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER=/usr/bin/chromedriver

# Create app dir
WORKDIR /app

# Install Python dependencies first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . /app

# Expose default port
EXPOSE 8000

# Start the API; Render sets $PORT at runtime. Use 1 worker for low RAM.
CMD ["bash","-lc","uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 15"]