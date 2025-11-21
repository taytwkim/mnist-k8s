FROM python:3.11-slim

WORKDIR /app

# (Optional but usually needed for torch/torchvision wheels that compile stuff)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY train/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY model.py ./model.py
COPY train/train.py ./train.py

# Copy existing data so the container can reuse it (or let torchvision download again)
COPY data ./data

# Directory where you expect the artifact to be written
# (make sure train.py saves to ./artifacts/mnist.pt)
RUN mkdir -p artifacts

# Default: train and save model
CMD ["python", "train.py", "--save-model"]
