FROM python:3.12-slim

WORKDIR /app

# Install dependencies first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Mount the datasets at runtime:
#   docker run --rm -v "$PWD/Datasets OP_26 Analytics:/app/Datasets OP_26 Analytics" ev-tariff
CMD ["python", "-m", "src.run_pipeline"]
