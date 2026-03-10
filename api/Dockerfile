FROM python:3.11-slim

WORKDIR /app

COPY api/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api/ /app/api/
COPY data/ /app/data/

ENV DATA_DIR=/app/data
ENV MODEL_DIR=/app/model
ENV DB_PATH=/app/db.sqlite3

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
