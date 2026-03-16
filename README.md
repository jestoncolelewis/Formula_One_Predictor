# The Formula One Predictor
### Final project for Bachelor's of Science in Computer Science at WGU by Jeston Cole Lewis

An F1 race position predictor using TensorFlow Decision Forests, split into two independently deployable services.

## Architecture

| Service | Directory | Stack |
|---------|-----------|-------|
| **API** | `api/` | Python FastAPI — ML inference, data pipeline, Jolpica ingestion |
| **Frontend** | `frontend/` | Ruby on Rails — predictor form, analysis dashboard, admin panel |

Each directory is a standalone project with its own Dockerfile, README, and dependencies. They communicate over HTTP via the `F1_API_URL` environment variable.

## Quick Start (both services locally)

```bash
docker-compose up --build
```

- Frontend: http://localhost:3000
- API: http://localhost:8000

## Railway Deployment

Both services deploy as a single Railway project with two services:

1. Create a new project in Railway
2. Add a service for the **API**: point it at this repo, set the root directory to `api/`
3. Add a service for the **Frontend**: point it at this repo, set the root directory to `frontend/`
4. Set environment variables:

**API service:**
| Variable | Value |
|----------|-------|
| `DB_PATH` | `/app/db.sqlite3` (attach a volume for persistence) |
| `MODEL_DIR` | `/app/model` (attach a volume for persistence) |

**Frontend service:**
| Variable | Value |
|----------|-------|
| `F1_API_URL` | Internal Railway URL of the API service (e.g. `http://api.railway.internal:8000`) |
| `SECRET_KEY_BASE` | Generate with `bin/rails secret` |
| `RAILS_ENV` | `production` |

Railway auto-detects the Dockerfile in each root directory. Pushing changes to `api/` only rebuilds the API service, and pushing changes to `frontend/` only rebuilds the frontend.

## Data Ingestion

New race data can be ingested via the admin panel at `/admin/ingest` or directly via the API:

```bash
curl -X POST http://localhost:8000/api/data/ingest \
  -H "Content-Type: application/json" \
  -d '{"year": 2024, "round": 1}'
```

The model automatically retrains after new data is ingested.
