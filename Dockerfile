FROM python:3.11-slim

WORKDIR /app

# Install CPU-only torch first — prevents pip from pulling the full CUDA version
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploads chroma_db

ENV PORT=7860
ENV FLASK_ENV=production
ENV USE_RERANKER=true
ENV PYTHONUNBUFFERED=1
ENV TOKENIZERS_PARALLELISM=false

EXPOSE 7860

CMD ["gunicorn", "--bind", "0.0.0.0:7860", "--timeout", "120", "--workers", "1", "app:app"]
