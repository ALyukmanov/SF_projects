# Сервис рекомендаций (item-item co-occurrence) — FastAPI + uvicorn.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    RETAIL_RECOMMENDER_MODEL_DIR=/app/artifacts/model \
    OPENBLAS_NUM_THREADS=1 \
    PORT=8000

WORKDIR /app

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

# исходники, конфиг и артефакт модели (сырые CSV в образ не кладутся)
COPY src/ src/
COPY configs/ configs/
COPY artifacts/model/ artifacts/model/

# упасть на этапе сборки, если импорт приложения сломан
RUN python -c "import retail_recommender.service.app"

RUN useradd --create-home --uid 10001 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health',timeout=3).status==200 else 1)"

CMD ["uvicorn", "retail_recommender.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
