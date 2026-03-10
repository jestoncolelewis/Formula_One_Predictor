import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from api.database import init_db
from api.data_pipeline import (
    PREDICTORS, POSITION_COLUMNS,
    load_data_from_csv, load_data_from_db, save_data_to_db,
    process_raw_results, get_current_season, get_circuits_list,
    get_training_test_split,
)
from api.model import f1_model
from api.ingestion import fetch_race_results, fetch_season_schedule

import pandas as pd

DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "data"))
CSV_PATH = os.path.join(DATA_DIR, "final_rolling.csv")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database and load model on startup."""
    init_db()

    # Migrate CSV to DB if database is empty
    data = load_data_from_db()
    if data.empty and os.path.exists(CSV_PATH):
        data = load_data_from_csv(CSV_PATH)
        save_data_to_db(data)

    # Load or train model
    if not f1_model.load():
        data = load_data_from_db()
        training, _ = get_training_test_split(data)
        f1_model.train(training)

    yield


app = FastAPI(title="F1 Predictor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Request/Response models ---

class PredictRequest(BaseModel):
    driver: str
    circuit: str
    grid: int


class IngestRequest(BaseModel):
    year: int
    round: int


# --- Endpoints ---

@app.get("/api/drivers")
def get_drivers():
    """Current season driver list."""
    data = load_data_from_db()
    season = get_current_season(data)
    drivers = sorted(season["driverRef"].unique().tolist())
    return {"drivers": drivers}


@app.get("/api/circuits")
def get_circuits():
    """Current season circuit list."""
    data = load_data_from_db()
    circuits = get_circuits_list(data)
    return {"circuits": circuits.to_dict("records")}


@app.post("/api/predict")
def predict(req: PredictRequest):
    """Predict finishing position probabilities for a driver/circuit/grid combo."""
    if f1_model.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = load_data_from_db()
    season = get_current_season(data)
    latest_race = season[season["raceId"] == season["raceId"].max()]
    circuits = get_circuits_list(data)

    # Look up codes
    driver_rows = latest_race[latest_race["driverRef"] == req.driver]
    if driver_rows.empty:
        raise HTTPException(status_code=404, detail=f"Driver '{req.driver}' not found in current season")

    circuit_rows = circuits[circuits["circuitRef"] == req.circuit]
    if circuit_rows.empty:
        raise HTTPException(status_code=404, detail=f"Circuit '{req.circuit}' not found")

    driver_row = driver_rows.iloc[0]
    driver_df = pd.DataFrame({
        "driver_code": [int(driver_row["driver_code"])],
        "constructor_code": [int(driver_row["constructor_code"])],
        "circuit_code": [int(circuit_rows.iloc[0]["circuit_code"])],
        "grid_rolling": [float(driver_row["grid_rolling"])] if pd.notna(driver_row["grid_rolling"]) else [0.0],
        "position_rolling": [float(driver_row["position_rolling"])] if pd.notna(driver_row["position_rolling"]) else [0.0],
        "pos_delta_rolling": [float(driver_row["pos_delta_rolling"])] if pd.notna(driver_row["pos_delta_rolling"]) else [0.0],
        "grid": [req.grid],
        "pos_delta": [float(req.grid)],
    })

    result = f1_model.predict_single(driver_df)

    probabilities = {str(pos): float(result[pos].iloc[0]) for pos in POSITION_COLUMNS}

    return {
        "driver": req.driver,
        "circuit": req.circuit,
        "grid": req.grid,
        "probabilities": probabilities,
    }


@app.get("/api/analysis/single-race")
def analysis_single_race():
    """Latest race prediction table from test data."""
    if f1_model.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = load_data_from_db()
    _, test = get_training_test_split(data)

    full_table, _ = f1_model.predict_batch(test)
    latest_race_id = full_table["raceId"].max()
    single = full_table[full_table["raceId"] == latest_race_id].copy()
    single = single.sort_values("grid")

    rows = []
    for _, row in single.iterrows():
        rows.append({
            "driver": row["driverRef"],
            "grid": int(row["grid"]),
            "circuit": row["circuitRef"],
            "probabilities": {str(pos): float(row[pos]) for pos in POSITION_COLUMNS},
        })

    return {"race_id": int(latest_race_id), "predictions": rows}


@app.get("/api/analysis/stats")
def analysis_stats():
    """Training data description and model accuracy."""
    data = load_data_from_db()
    training, _ = get_training_test_split(data)

    description = training.describe().to_dict()
    accuracy = f1_model.accuracy if f1_model.accuracy else 0.0

    return {
        "accuracy": accuracy,
        "accuracy_pct": round(accuracy * 100, 2),
        "training_rows": len(training),
        "total_rows": len(data),
        "description": description,
    }


@app.get("/api/analysis/confusion-matrix")
def analysis_confusion_matrix():
    """Confusion matrix for test data predictions."""
    if f1_model.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    data = load_data_from_db()
    _, test = get_training_test_split(data)
    cm = f1_model.get_confusion_matrix(test)

    return {"confusion_matrix": cm, "labels": POSITION_COLUMNS}


@app.post("/api/data/ingest")
async def ingest_race(req: IngestRequest):
    """
    Fetch new race data from Jolpica API, process it, and retrain the model.

    This endpoint:
    1. Fetches results from Jolpica for the given year/round
    2. Processes the raw data (encoding, rolling features)
    3. Saves to database
    4. Automatically retrains the model
    """
    try:
        new_results = await fetch_race_results(req.year, req.round)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch race data: {str(e)}")

    # Load existing data and reprocess everything together
    existing_data = load_data_from_db()

    # Check for duplicate race
    race_id = req.year * 100 + req.round
    if not existing_data.empty and race_id in existing_data["raceId"].values:
        raise HTTPException(status_code=409, detail=f"Race {req.year} round {req.round} already exists in database")

    # Process all data together for consistent encoding and rolling features
    combined = process_raw_results(new_results, existing_data)
    save_data_to_db(combined)

    # Auto-retrain
    training, _ = get_training_test_split(combined)
    accuracy = f1_model.train(training)

    return {
        "message": f"Ingested {len(new_results)} results for {req.year} round {req.round}",
        "new_total_rows": len(combined),
        "model_accuracy": round(accuracy * 100, 2),
    }


@app.get("/api/schedule/{year}")
async def get_schedule(year: int):
    """Fetch race schedule for a season from Jolpica API."""
    try:
        schedule = await fetch_season_schedule(year)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch schedule: {str(e)}")
    return {"year": year, "races": schedule}
