FROM python:3.12-slim

WORKDIR /app

# Install minimal OS packages required for OpenCV and image operations
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code and entry points
COPY src/ ./src/
COPY main.py .
COPY synthetic_generator.py .

# Configure environment
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

CMD ["python", "main.py", "--generate-synthetic", "--video", "synthetic_pitch_feed.mp4"]
