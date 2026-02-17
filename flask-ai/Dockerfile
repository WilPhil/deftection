# 1. Use Python 3.10 (More stable for AI libs than 3.13)
FROM python:3.10-slim

# 2. Install the missing Linux libraries (The Fix)
# We force install libgl1 and libxcb so OpenCV stops crashing
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libxcb1 \
    libx11-xcb1 \
    && rm -rf /var/lib/apt/lists/*

# 3. Set the working directory
WORKDIR /app

# 4. Copy your requirements and install them
COPY requirements.txt .

# Install dependencies (Force CPU version of PyTorch to save space)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

# 5. Copy your application code
COPY . .

# 6. Command to run the app (using Gunicorn)
# Make sure 'api_server' matches your filename and 'app' matches your Flask variable
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "api_server:app"]
