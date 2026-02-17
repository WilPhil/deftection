# 1. Use Python 3.10
FROM python:3.10-slim

# 2. Install Linux libraries + YARA build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libyara-dev \
    python3-dev \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libx11-xcb1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set working directory
WORKDIR /app

# 4. Copy requirements and install
COPY requirements.txt .

# Upgrade pip and install
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 5. Copy your application code
COPY . .

# 6. Command to run the app
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "api_server:app"]
