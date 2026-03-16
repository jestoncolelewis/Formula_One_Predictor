# F1 Predictor API

Python FastAPI backend for the Formula One race position predictor. Uses TensorFlow Decision Forests for ML inference, with automated data ingestion from the Jolpica API.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

On first startup, the API automatically migrates `data/final_rolling.csv` into SQLite and trains the model.

## Docker

```bash
docker build -t f1-predictor-api .
docker run -p 8000:8000 f1-predictor-api
```

## Railway Deployment

In Railway, add this as a service with root directory set to `api/`. Set these environment variables:
- `DB_PATH` — path to SQLite file (attach a volume for persistence)
- `MODEL_DIR` — path to model directory (attach a volume for persistence)

Railway auto-detects the Dockerfile. Custom start command if not using Docker:
```
uvicorn app.main:app --host 0.0.0.0 --port $PORT
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/drivers` | GET | Current season drivers |
| `/api/circuits` | GET | Circuit list |
| `/api/predict` | POST | Position probability prediction |
| `/api/analysis/single-race` | GET | Latest race predictions |
| `/api/analysis/stats` | GET | Model accuracy and training stats |
| `/api/analysis/confusion-matrix` | GET | Confusion matrix |
| `/api/data/ingest` | POST | Ingest new race data and retrain |
| `/api/schedule/{year}` | GET | Season race schedule |

## Data Ingestion

```bash
curl -X POST http://localhost:8000/api/data/ingest \
  -H "Content-Type: application/json" \
  -d '{"year": 2024, "round": 1}'
```

The model automatically retrains after ingestion.
