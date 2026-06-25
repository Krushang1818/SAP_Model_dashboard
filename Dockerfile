FROM python:3.11-slim

WORKDIR /app

# Install build-essential for compiling bitsandbytes/scipy if necessary
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Install PyTorch with CUDA 12.4 support for GPU acceleration
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu124
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8001

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8001"]
