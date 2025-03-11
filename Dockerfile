# Use an official Python image
FROM python:3.9-slim

# Set working directory inside the container
WORKDIR /app

RUN apt-get update && apt-get install -y \
g++ \
gcc \
libgdal-dev \
python3-dev \
libgl1-mesa-glx \
libglib2.0-0\
&& rm -rf /var/lib/apt/lists/*

ENV GDAL_VERSION=3.6.2
ENV GDAL_CONFIG=/usr/bin/gdal-config

# Copy the requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code
COPY . .

# Expose the port for the application (you might not need this if you're not using any web server)
EXPOSE 8080

# Run the application (assuming your main.py contains the code to execute)
CMD ["python", "main.py"]
