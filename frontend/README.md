# F1 Predictor Frontend

Ruby on Rails frontend for the Formula One race position predictor. Communicates with the F1 Predictor API for ML predictions and data.

## Pages

- **Predictor** (`/`) — Select a driver, circuit, and grid position to see finishing probabilities
- **Analysis** (`/analysis`) — Model accuracy, training data stats, single-race predictions, confusion matrix
- **Data Ingest** (`/admin/ingest`) — Fetch season schedules and ingest new race results

## Setup

```bash
bundle install
F1_API_URL=http://localhost:8000 bin/rails server -p 3000
```

Requires the API service running at `F1_API_URL`.

## Docker

```bash
docker build -t f1-predictor-frontend .
docker run -p 3000:80 -e F1_API_URL=http://your-api-url:8000 -e SECRET_KEY_BASE=your-secret f1-predictor-frontend
```

## Railway Deployment

Create a Railway project from this repo. Set these environment variables:
- `F1_API_URL` — Public URL of your deployed API service
- `SECRET_KEY_BASE` — generate with `bin/rails secret`
- `RAILS_ENV` — `production`

Railway auto-detects the Dockerfile.
