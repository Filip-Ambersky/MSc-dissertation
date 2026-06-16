import time
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"

OUT_DIR = Path("raw data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SLEEP_SECONDS = 0.25
MAX_RETRIES = 5

# B1610 is huge. Keep chunk_days=1.
# The output will still be one final CSV, but internally it downloads one day at a time.
DATASETS = {
    "FUELHH": {
        "endpoint": f"{BASE_URL}/datasets/FUELHH/stream",
        "output_file": OUT_DIR / "raw_FUELHH_2023_2025.csv",
        "chunk_days": 7,
    },
    "B1430_DAG": {
        "endpoint": f"{BASE_URL}/datasets/DAG/stream",
        "output_file": OUT_DIR / "raw_B1430_DAG_2023_2025.csv",
        "chunk_days": 7,
    },
    "B1440_DGWS": {
        "endpoint": f"{BASE_URL}/datasets/DGWS/stream",
        "output_file": OUT_DIR / "raw_B1440_DGWS_2023_2025.csv",
        "chunk_days": 7,
    },
}


# ============================================================
# HELPERS
# ============================================================

def extract_records(payload):
    """
    Extract records from common Elexon response shapes.
    """
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ["data", "items", "results"]:
            if key in payload and isinstance(payload[key], list):
                return payload[key]

    return []


def request_json(url, params, max_retries=MAX_RETRIES):
    """
    GET request with retries.
    """
    for attempt in range(max_retries):
        response = requests.get(url, params=params, timeout=180)

        if response.status_code == 429:
            wait = 10 * (attempt + 1)
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue

        if response.status_code >= 500:
            wait = 10 * (attempt + 1)
            print(f"Server error {response.status_code}. Waiting {wait}s...")
            time.sleep(wait)
            continue

        if response.status_code >= 400:
            print("\nRequest failed")
            print("URL:", response.url)
            print("Status:", response.status_code)
            print("Text:", response.text[:1000])

        response.raise_for_status()
        return response.json()

    raise RuntimeError(f"Failed after {max_retries} attempts: {url}")


def iter_date_chunks(start_date, end_date, chunk_days):
    """
    Makes [start, end) datetime chunks.
    """
    start = datetime.strptime(start_date, "%Y-%m-%d")
    final_exclusive = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)

    current = start

    while current < final_exclusive:
        chunk_end = min(current + timedelta(days=chunk_days), final_exclusive)
        yield current, chunk_end
        current = chunk_end


def fetch_dataset_chunk(endpoint, start_dt, end_dt):
    """
    Pull one raw chunk from one Elexon stream endpoint.
    """
    params = {
        "from": start_dt.strftime("%Y-%m-%dT%H:%MZ"),
        "to": end_dt.strftime("%Y-%m-%dT%H:%MZ"),
    }

    payload = request_json(endpoint, params=params)
    records = extract_records(payload)

    return pd.DataFrame(records)


def append_df_to_csv(df, output_file, header_written):
    """
    Append a DataFrame to one final CSV.
    """
    if df.empty:
        return header_written

    df.to_csv(
        output_file,
        mode="a",
        index=False,
        header=not header_written,
    )

    return True


# ============================================================
# DOWNLOAD ONE DATASET TO ONE FINAL CSV
# ============================================================

def download_dataset_to_single_csv(dataset_name, config):
    endpoint = config["endpoint"]
    output_file = config["output_file"]
    chunk_days = config["chunk_days"]

    if output_file.exists():
        print(f"\nDeleting existing file so it can be rebuilt: {output_file}")
        output_file.unlink()

    print("\n============================================================")
    print(f"Downloading {dataset_name}")
    print(f"Endpoint: {endpoint}")
    print(f"Output file: {output_file}")
    print(f"Internal chunk size: {chunk_days} day(s)")
    print("============================================================")

    header_written = False
    total_rows = 0
    first_columns = None

    for start_dt, end_dt in iter_date_chunks(START_DATE, END_DATE, chunk_days):
        label = f"{start_dt:%Y-%m-%d} to {end_dt:%Y-%m-%d}"
        print(f"Downloading {dataset_name}: {label}")

        try:
            df = fetch_dataset_chunk(endpoint, start_dt, end_dt)

            if df.empty:
                print("  Empty response")
                time.sleep(SLEEP_SECONDS)
                continue

            # Keep one consistent raw schema after the first non-empty chunk.
            # This avoids broken CSV structure if later chunks have columns in different order.
            if first_columns is None:
                first_columns = df.columns.tolist()
            else:
                for col in first_columns:
                    if col not in df.columns:
                        df[col] = pd.NA
                df = df[first_columns]

            header_written = append_df_to_csv(
                df=df,
                output_file=output_file,
                header_written=header_written,
            )

            total_rows += len(df)

            print(f"  Appended rows: {len(df)}")
            print(f"  Total rows so far: {total_rows}")

        except Exception as e:
            print(f"  ERROR for {dataset_name}, {label}")
            print(f"  {e}")
            # Continue to next chunk instead of killing the whole multi-year run.
            # If you prefer strict mode, replace this with: raise

        time.sleep(SLEEP_SECONDS)

    print(f"\nFinished {dataset_name}")
    print(f"Saved: {output_file}")
    print(f"Total rows written: {total_rows}")


# ============================================================
# RUN ALL DATASETS
# ============================================================

def main():
    for dataset_name, config in DATASETS.items():
        download_dataset_to_single_csv(dataset_name, config)

    print("\nDone. Final raw CSVs saved in:")
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()