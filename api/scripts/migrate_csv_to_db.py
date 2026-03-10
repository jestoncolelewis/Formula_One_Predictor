#!/usr/bin/env python3
"""One-time migration script to load CSV data into SQLite database."""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from api.database import init_db
from api.data_pipeline import load_data_from_csv, save_data_to_db, load_data_from_db

CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "data", "final_rolling.csv")


def main():
    print("Initializing database...")
    init_db()

    existing = load_data_from_db()
    if not existing.empty:
        print(f"Database already has {len(existing)} rows. Skipping migration.")
        print("To force re-migration, delete db.sqlite3 and run again.")
        return

    print(f"Loading data from {CSV_PATH}...")
    data = load_data_from_csv(CSV_PATH)
    print(f"Loaded {len(data)} rows.")

    print("Saving to database...")
    save_data_to_db(data)

    # Verify
    verify = load_data_from_db()
    print(f"Migration complete. Database now has {len(verify)} rows.")


if __name__ == "__main__":
    main()
