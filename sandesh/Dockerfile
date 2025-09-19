# Use a slim Python image
FROM python:3.11-slim

# Prevent Python from writing .pyc files and enable unbuffered logs
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies including Chromium and Chromedriver for Selenium
RUN apt-get update && apt-get install -y --no-install-recommends \
    chromium \
    chromium-driver \
    fonts-liberation \
    libnss3 \
    libgconf-2-4 \
    libasound2 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Helpful environment variables for Selenium/Chromium
ENV CHROME_BIN=/usr/bin/chromium \
    CHROMEDRIVER=/usr/bin/chromedriver

# Create common symlinks some drivers/tools expect
RUN ln -sf /usr/bin/chromium /usr/bin/chromium-browser && \
    ln -sf /usr/bin/chromium /usr/bin/google-chrome

# Set workdir
WORKDIR /app

# Install Python deps first (better layer caching)
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose port
EXPOSE 8000

# Start the API. Use the PORT from environment with a default
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
