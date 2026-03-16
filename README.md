# The Formula One Predictor
### Final project for Bachelor's of Science in Computer Science at WGU by Jeston Cole Lewis

An F1 race position predictor using TensorFlow Decision Forests, split into two independently deployable services.

## Architecture

| Service | Directory | Deploys as |
|---------|-----------|------------|
| **API** | `api/` | Python FastAPI — ML inference, data pipeline, Jolpica ingestion |
| **Frontend** | `frontend/` | Ruby on Rails — predictor form, analysis dashboard, admin panel |

Each directory is a standalone project with its own Dockerfile, README, and dependencies. They can be deployed as separate Railway services (or any platform) and communicate over HTTP via the `F1_API_URL` environment variable.

## Splitting into Separate Repos

Each service is self-contained and ready to be pushed to its own repo:

```bash
# API repo
gh repo create jestoncolelewis/f1-predictor-api --public
cd api
git init && git add -A && git commit -m "initial commit"
git remote add origin https://github.com/jestoncolelewis/f1-predictor-api.git
git push -u origin main

# Frontend repo
gh repo create jestoncolelewis/f1-predictor-frontend --public
cd ../frontend
git init && git add -A && git commit -m "initial commit"
git remote add origin https://github.com/jestoncolelewis/f1-predictor-frontend.git
git push -u origin main
```

## Quick Start (both services locally)

```bash
docker-compose up --build
```

- Frontend: http://localhost:3000
- API: http://localhost:8000

## Data Ingestion

New race data can be ingested via the admin panel at `/admin/ingest` or directly via the API:

```bash
curl -X POST http://localhost:8000/api/data/ingest \
  -H "Content-Type: application/json" \
  -d '{"year": 2024, "round": 1}'
```

The model automatically retrains after new data is ingested.
