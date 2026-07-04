FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY api/requirements_api.txt /app/requirements_api.txt
COPY requirements_inference.txt /app/requirements_inference.txt

RUN pip install --no-cache-dir -r requirements_api.txt -r requirements_inference.txt

COPY api/ /app/

RUN python manage.py migrate --noinput

EXPOSE 8080

CMD ["gunicorn", "api_config.wsgi:application", "--bind", "0.0.0.0:8080"]
