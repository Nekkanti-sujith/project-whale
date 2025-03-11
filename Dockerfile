# Use an official lightweight Python image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    g++ \
    gcc \
    libgdal-dev \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
 && rm -rf /var/lib/apt/lists/*

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV GDAL_VERSION=3.6.2
ENV GDAL_CONFIG=/usr/bin/gdal-config

# Copy the requirements file
COPY requirements.txt .

# Install dependencies in a virtual environment
RUN python -m venv /venv && \
    /venv/bin/pip install --no-cache-dir --upgrade pip && \
    /venv/bin/pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Ensure the script is executable
RUN chmod +x main.py

# Expose the port Cloud Run expects
ENV PORT=8080
EXPOSE 8080

# Set entrypoint and run the application using virtual environment
CMD ["/venv/bin/python", "main.py"]
