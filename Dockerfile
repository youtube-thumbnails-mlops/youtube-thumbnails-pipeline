FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

WORKDIR /app

# Install system deps if needed
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install "dvc[s3]"

COPY . .
RUN chmod +x entrypoint.sh

# Default command
ENTRYPOINT ["./entrypoint.sh"]
