FROM python:3.11-slim

WORKDIR /app

# Install Python dependencies for the inference app
COPY app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy shared model definition + Flask app code
COPY model.py ./model.py
COPY app/app.py ./app.py

EXPOSE 5000

# Start the Flask app
CMD ["python", "app.py"]
