"""
F1 Data Collection — Phase 1
Uses FastF1 to pull race telemetry, lap times, pit stops,
weather, and session results for a given season.

Install:
    pip install fastf1 pandas

Usage:
    python f1_data_collection.py
"""

import fastf1
import pandas as pd
from pathlib import Path
import time

# ── Setup ──────────────────────────────────────────────────────────────────────

# FastF1 caches API responses locally — essential for performance.
# Without this, every run re-downloads everything (slow + rate-limited).
CACHE_DIR = Path("f1_cache")
CACHE_DIR.mkdir(exist_ok=True)
fastf1.Cache.enable_cache(str(CACHE_DIR))

OUTPUT_DIR = Path("f1_data")
OUTPUT_DIR.mkdir(exist_ok=True)


# ── 1. Get the season calendar ─────────────────────────────────────────────────

def get_season_schedule(year: int) -> pd.DataFrame:
    """
    Returns the full race calendar for a given season.
    Columns: RoundNumber, Country, Location, EventName, EventDate, EventFormat
    """
    schedule = fastf1.get_event_schedule(year)
    # Drop testing sessions (RoundNumber 0)
    schedule = schedule[schedule["RoundNumber"] > 0].reset_index(drop=True)
    print(f"  {year} season: {len(schedule)} rounds found")
    return schedule


# ── 2. Load a single race session ─────────────────────────────────────────────

def load_race_session(year: int, round_number: int) -> fastf1.core.Session:
    """
    Loads a race session with all telemetry.
    session.laps        -> lap-by-lap data for all drivers
    session.results     -> final classification
    session.weather_data -> weather per lap
    """
    session = fastf1.get_session(year, round_number, "R")  # "R" = Race
    session.load(
        laps=True,
        telemetry=False,   # Set True if you want per-point speed/throttle data
        weather=True,
        messages=False,
    )
    return session


# ── 3. Extract lap data ────────────────────────────────────────────────────────

def extract_lap_data(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Returns a clean lap-level DataFrame.
    Key columns:
        Driver, Team, LapNumber, LapTime (seconds), Stint,
        Compound (tire), TyreLife, PitOutTime, PitInTime, IsPersonalBest
    """
    laps = session.laps.copy()

    # Convert timedelta columns to float seconds for easier math
    time_cols = ["LapTime", "Sector1Time", "Sector2Time", "Sector3Time"]
    for col in time_cols:
        if col in laps.columns:
            laps[f"{col}Sec"] = laps[col].dt.total_seconds()

    # Add race metadata
    laps["Year"] = session.event["EventDate"].year
    laps["RoundNumber"] = session.event["RoundNumber"]
    laps["EventName"] = session.event["EventName"]
    laps["Circuit"] = session.event["Location"]

    keep = [
        "Year", "RoundNumber", "EventName", "Circuit",
        "Driver", "Team", "LapNumber",
        "LapTimeSec", "Sector1TimeSec", "Sector2TimeSec", "Sector3TimeSec",
        "Stint", "Compound", "TyreLife",
        "PitOutTime", "PitInTime", "IsPersonalBest",
        "TrackStatus", "Position",
    ]
    keep = [c for c in keep if c in laps.columns]
    return laps[keep]


# ── 4. Extract pit stop data ───────────────────────────────────────────────────

def extract_pit_stops(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Derives pit stop events from lap data.
    A pit stop = a lap where PitInTime is not null (driver came into the pits).
    Returns: Driver, LapNumber, Stint, Compound, TyreLife at stop, StopDuration
    """
    laps = session.laps.copy()

    pit_laps = laps[laps["PitInTime"].notna()].copy()
    if pit_laps.empty:
        return pd.DataFrame()

    pit_laps["StopDurationSec"] = (
        pit_laps["PitOutTime"] - pit_laps["PitInTime"]
    ).dt.total_seconds()

    pit_laps["Year"] = session.event["EventDate"].year
    pit_laps["EventName"] = session.event["EventName"]
    pit_laps["Circuit"] = session.event["Location"]

    keep = ["Year", "EventName", "Circuit", "Driver", "Team",
            "LapNumber", "Stint", "Compound", "TyreLife", "StopDurationSec"]
    keep = [c for c in keep if c in pit_laps.columns]
    return pit_laps[keep].reset_index(drop=True)


# ── 5. Extract race results ────────────────────────────────────────────────────

def extract_results(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Returns final race classification.
    Key columns: Position, Driver, Team, Points, Status (Finished / DNF reason),
                 GridPosition, Time (total race time for finishers)
    """
    results = session.results.copy()
    results["Year"] = session.event["EventDate"].year
    results["RoundNumber"] = session.event["RoundNumber"]
    results["EventName"] = session.event["EventName"]
    results["Circuit"] = session.event["Location"]

    keep = [
        "Year", "RoundNumber", "EventName", "Circuit",
        "Position", "Abbreviation", "FullName", "TeamName",
        "GridPosition", "Points", "Status", "Time",
        "Q1", "Q2", "Q3",  # qualifying times if available
    ]
    keep = [c for c in keep if c in results.columns]
    return results[keep].reset_index(drop=True)


# ── 6. Extract weather data ────────────────────────────────────────────────────

def extract_weather(session: fastf1.core.Session) -> pd.DataFrame:
    """
    Returns session weather sampled over time.
    Key columns: Time, AirTemp, TrackTemp, Humidity, Pressure, Rainfall, WindSpeed
    """
    weather = session.weather_data.copy()
    weather["Year"] = session.event["EventDate"].year
    weather["EventName"] = session.event["EventName"]
    return weather.reset_index(drop=True)


# ── 7. Collect a full season ───────────────────────────────────────────────────

def collect_season(year: int, max_rounds: int = None) -> dict:
    """
    Loops over every round in a season and collects:
      - lap data
      - pit stops
      - race results
      - weather

    Returns a dict of DataFrames keyed by table name.
    Set max_rounds to an int during development to limit API calls.
    """
    schedule = get_season_schedule(year)
    if max_rounds:
        schedule = schedule.head(max_rounds)

    all_laps, all_pits, all_results, all_weather = [], [], [], []

    for _, event in schedule.iterrows():
        round_num = event["RoundNumber"]
        name = event["EventName"]
        print(f"  Loading round {round_num}: {name}...")

        try:
            session = load_race_session(year, round_num)

            all_laps.append(extract_lap_data(session))
            all_pits.append(extract_pit_stops(session))
            all_results.append(extract_results(session))
            all_weather.append(extract_weather(session))

            # Be polite to the API — small delay between rounds
            time.sleep(1)

        except Exception as e:
            print(f"    Warning: skipping round {round_num} — {e}")
            continue

    return {
        "laps":    pd.concat(all_laps,    ignore_index=True) if all_laps    else pd.DataFrame(),
        "pits":    pd.concat(all_pits,    ignore_index=True) if all_pits    else pd.DataFrame(),
        "results": pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame(),
        "weather": pd.concat(all_weather, ignore_index=True) if all_weather else pd.DataFrame(),
    }


# ── 8. Save to CSV ─────────────────────────────────────────────────────────────

def save_season(year: int, tables: dict):
    for name, df in tables.items():
        if df.empty:
            print(f"  Skipping empty table: {name}")
            continue
        path = OUTPUT_DIR / f"{year}_{name}.csv"
        df.to_csv(path, index=False)
        print(f"  Saved {len(df):,} rows -> {path}")


# ── 9. Quick data quality check ────────────────────────────────────────────────

def data_quality_report(tables: dict):
    print("\n── Data Quality Report ──────────────────────────")
    for name, df in tables.items():
        if df.empty:
            print(f"  {name}: EMPTY")
            continue
        null_pct = (df.isnull().sum() / len(df) * 100).round(1)
        high_null = null_pct[null_pct > 20]
        print(f"\n  {name}: {len(df):,} rows x {df.shape[1]} cols")
        print(f"    dtypes: {df.dtypes.value_counts().to_dict()}")
        if not high_null.empty:
            print(f"    High-null columns (>20%): {high_null.to_dict()}")
        else:
            print(f"    No high-null columns")


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Quick start: collect just 3 rounds of 2023 to test the pipeline ──
    # Once this works, change years to range(2018, 2025) and max_rounds=None

    TARGET_YEARS = [2023]   # expand to [2018, 2019, ..., 2024] for full project
    MAX_ROUNDS   = 3        # set to None for full season

    all_tables = {"laps": [], "pits": [], "results": [], "weather": []}

    for year in TARGET_YEARS:
        print(f"\nCollecting {year} season (rounds 1-{MAX_ROUNDS or 'all'})...")
        tables = collect_season(year, max_rounds=MAX_ROUNDS)
        save_season(year, tables)
        for key in all_tables:
            if not tables[key].empty:
                all_tables[key].append(tables[key])

    # Merge across years
    merged = {k: pd.concat(v, ignore_index=True) for k, v in all_tables.items() if v}

    data_quality_report(merged)

    print("\nDone. Your data is in the f1_data/ folder.")
    print("Next step: open a Jupyter notebook and import from there.")
    print("  e.g.  laps = pd.read_csv('f1_data/2023_laps.csv')")
