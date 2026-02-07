FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements first to leverage cache
COPY requirements.txt .

# Install system dependencies for building packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and data
COPY . .

# Expose port (Azure will map to dynamic PORT)
EXPOSE 8000

# Environment variables
ENV PYTHONUNBUFFERED=1
ENV PORT=8000

# Use Gunicorn to serve Flask in production
CMD ["gunicorn", "--bind", "0.0.0.0:8000", "app:app", "--workers", "4"]
