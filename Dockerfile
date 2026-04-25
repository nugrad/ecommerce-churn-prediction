FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY src/    ./src/
COPY api/    ./api/
COPY models/ ./models/

RUN mkdir -p ./data/processed
COPY data/processed/feature_matrix.csv ./data/processed/feature_matrix.csv

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]