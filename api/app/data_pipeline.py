import numpy as np
import pandas as pd

from app.database import SessionLocal, RaceResult


PREDICTORS = [
    "grid", "position", "pos_delta", "driver_code", "constructor_code",
    "circuit_code", "grid_rolling", "position_rolling", "pos_delta_rolling"
]

POSITION_COLUMNS = list(range(1, 21))


def update_names(series: pd.Series) -> pd.Series:
    """Convert refs like 'max_verstappen' to 'Max Verstappen'."""
    return series.str.replace("_", " ").str.title()


def rolling_finish_avg(group: pd.DataFrame, cols: list, new_cols: list) -> pd.DataFrame:
    """Compute rolling 3-race average for given columns within a driver group."""
    group = group.sort_values("raceId")
    rolling_stats = group[cols].rolling(3, closed="left").mean()
    group[new_cols] = rolling_stats
    return group


def process_raw_results(results_df: pd.DataFrame, existing_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Process raw race results into the final_rolling format.

    Expects results_df with columns:
        raceId, grid, position, year, date, time, circuitRef, driverRef, constructorRef

    If existing_data is provided, uses it for consistent categorical encoding
    and rolling computation across historical + new data.
    """
    df = results_df.copy()

    # Clean refs
    df["circuitRef"] = update_names(df["circuitRef"])
    df["constructorRef"] = update_names(df["constructorRef"])
    df["driverRef"] = update_names(df["driverRef"])

    if existing_data is not None:
        combined = pd.concat([existing_data, df], ignore_index=True)
    else:
        combined = df.copy()

    # Encode categoricals on combined data for consistency
    combined["circuit_code"] = combined["circuitRef"].astype("category").cat.codes
    combined["driver_code"] = combined["driverRef"].astype("category").cat.codes
    combined["constructor_code"] = combined["constructorRef"].astype("category").cat.codes

    # Filter to valid finishing positions
    combined["position"] = combined["position"].astype(float)
    combined = combined[combined["position"] <= 20.0]

    # Compute pos_delta
    combined["pos_delta"] = combined["grid"] - combined["position"]

    # Compute rolling features
    cols = ["grid", "position", "pos_delta"]
    new_cols = [f"{c}_rolling" for c in cols]
    combined = (
        combined.groupby("driverRef", group_keys=False)
        .apply(lambda x: rolling_finish_avg(x, cols, new_cols))
    )
    combined = combined.sort_values(["raceId", "position"])
    combined.index = range(combined.shape[0])

    return combined


def load_data_from_db() -> pd.DataFrame:
    """Load all race results from the database."""
    session = SessionLocal()
    try:
        results = session.query(RaceResult).all()
        if not results:
            return pd.DataFrame()
        records = [{c.name: getattr(r, c.name) for c in RaceResult.__table__.columns if c.name != "id"} for r in results]
        return pd.DataFrame(records)
    finally:
        session.close()


def save_data_to_db(df: pd.DataFrame):
    """Replace all race results in the database with the given DataFrame."""
    session = SessionLocal()
    try:
        session.query(RaceResult).delete()
        records = df.to_dict("records")
        for record in records:
            session.add(RaceResult(**record))
        session.commit()
    finally:
        session.close()


def load_data_from_csv(path: str) -> pd.DataFrame:
    """Load and clean data from CSV (used for initial migration)."""
    data = pd.read_csv(path)
    data.dropna(subset="position", inplace=True)
    data["position"] = data["position"].astype(int)
    return data


def get_current_season(data: pd.DataFrame) -> pd.DataFrame:
    """Get data for the most recent season."""
    return data[data["year"] == data["year"].max()]


def get_circuits_list(data: pd.DataFrame) -> pd.DataFrame:
    """Get unique circuits with their codes."""
    circuits = data[["circuitRef", "circuit_code"]].drop_duplicates()
    return circuits.sort_values("circuitRef").reset_index(drop=True)


def get_training_test_split(data: pd.DataFrame, split_year: int = 2022):
    """Split data into training and test sets."""
    training = data[data["year"] < split_year]
    test = data[data["year"] >= split_year]
    return training, test
