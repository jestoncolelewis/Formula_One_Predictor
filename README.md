# The Formula One Predictor
### Final project for Bachelor's of Science in Computer Science at WGU by Jeston Cole Lewis

An F1 race position predictor using TensorFlow Decision Forests, with a Ruby on Rails frontend and Python FastAPI backend.

## Architecture

- **Frontend**: Ruby on Rails app (predictor form, analysis dashboard, data ingestion admin)
- **Backend**: Python FastAPI service (ML model, predictions, data pipeline)
- **Data Ingestion**: Fetches new race results from the Jolpica API (Ergast successor) and auto-retrains the model

## Quick Start with Docker

```bash
docker-compose up --build
```

- Frontend: http://localhost:3000
- API: http://localhost:8000

## Local Development

### API (Python)

```bash
cd api
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cd ..
uvicorn api.main:app --reload --port 8000
```

### Frontend (Rails)

```bash
cd frontend
bundle install
F1_API_URL=http://localhost:8000 bin/rails server -p 3000
```

## Data Ingestion

New race data can be ingested via the admin panel at `/admin/ingest` or directly via the API:

```bash
# Fetch and ingest results for 2024 Round 1
curl -X POST http://localhost:8000/api/data/ingest \
  -H "Content-Type: application/json" \
  -d '{"year": 2024, "round": 1}'
```

The model automatically retrains after new data is ingested.

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
