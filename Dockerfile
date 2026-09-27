FROM python:3.9-slim

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Copy project
COPY . .

# Create required folders
RUN mkdir -p /app/data/uploaded_pdfs \
    /app/data/chroma_db

# FastAPI port
EXPOSE 8000

# Start application
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]