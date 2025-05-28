FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p logs

# Expose ports for FastAPI and Streamlit
EXPOSE 8000 8501

# Set environment variables
ENV PYTHONPATH=/app

# Command to run the application
CMD ["sh", "-c", "uvicorn orchestrator.main:app --host 0.0.0.0 --port 8000 --reload & streamlit run streamlit_app/app.py --server.port=8501 --server.address=0.0.0.0"]
