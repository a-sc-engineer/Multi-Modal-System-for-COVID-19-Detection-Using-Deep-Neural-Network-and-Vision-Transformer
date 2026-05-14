# Use the official Python 3.10 slim image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for OpenCV/PIL if needed
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Create directories for static files and templates
RUN mkdir -p /app/static /app/templates

# Copy the rest of the application
COPY . .

# Expose the port the app runs on (Hugging Face Spaces exposes port 7860 by default)
# We will run flask on 7860 to be compatible with HF Spaces out of the box
EXPOSE 7860

# Set environment variables for Flask
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=7860

# Run the application using gunicorn for production
CMD ["gunicorn", "-b", "0.0.0.0:7860", "--timeout", "120", "app:app"]
