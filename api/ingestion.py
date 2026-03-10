import httpx
import pandas as pd

JOLPICA_BASE = "https://api.jolpi.ca/ergast/f1"


async def fetch_race_results(year: int, round_num: int) -> pd.DataFrame:
    """
    Fetch race results from the Jolpica API (Ergast successor).

    Returns a DataFrame with columns matching the raw input schema:
        raceId, grid, position, year, date, time, circuitRef, driverRef, constructorRef
    """
    url = f"{JOLPICA_BASE}/{year}/{round_num}/results.json"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

    race_data = data["MRData"]["RaceTable"]["Races"]
    if not race_data:
        raise ValueError(f"No race data found for {year} round {round_num}")

    race = race_data[0]
    circuit_ref = race["Circuit"]["circuitId"]
    race_date = race["date"]
    race_time = race.get("time", "00:00:00").rstrip("Z")

    # Build a synthetic raceId from year and round
    race_id = year * 100 + round_num

    rows = []
    for result in race["Results"]:
        position = result.get("position")
        if position is None or position == "R":
            continue
        rows.append({
            "raceId": race_id,
            "grid": float(result["grid"]),
            "position": float(position),
            "year": year,
            "date": race_date,
            "time": race_time,
            "circuitRef": circuit_ref,
            "driverRef": result["Driver"]["driverId"],
            "constructorRef": result["Constructor"]["constructorId"],
        })

    return pd.DataFrame(rows)


async def fetch_season_schedule(year: int) -> list[dict]:
    """Fetch the race schedule for a season."""
    url = f"{JOLPICA_BASE}/{year}.json"

    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        data = resp.json()

    races = data["MRData"]["RaceTable"]["Races"]
    return [
        {
            "round": int(r["round"]),
            "raceName": r["raceName"],
            "circuitId": r["Circuit"]["circuitId"],
            "circuitName": r["Circuit"]["circuitName"],
            "date": r["date"],
        }
        for r in races
    ]
