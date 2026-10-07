FROM python:3.11-slim

# System deps: tesseract (OCR) + libs for OpenCV / PyMuPDF
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

# Pre-create runtime dirs
RUN mkdir -p data vector_store

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
