# Use an official Python image
FROM python:3.9-slim as builder

# Set working directory inside the container
WORKDIR /app

# Install dependencies for building
RUN apt-get update && apt-get install -y \
    g++ \
    gcc \
    libgdal-dev \
    python3-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy the requirements file
COPY requirements.txt /app/requirements.txt

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Final stage to build the app image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy installed dependencies from the builder image
COPY --from=builder /root/.local /root/.local

# Copy the rest of your application code
COPY . .

# Expose the port for the application
EXPOSE 8080

# Set environment variables
ENV PATH=/root/.local/bin:$PATH

# Run the application
CMD ["python", "main.py"]
