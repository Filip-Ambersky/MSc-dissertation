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

# B1610 is huge, so keep it daily.
# Your earlier 1-day B1610 test returned ~500k rows.
DATASETS = {
    "TSDF": {
        "endpoint": f"{BASE_URL}/datasets/TSDF/stream",
        "folder": "TSDF",
        "chunk_days": 7,
    },
    "B1610": {
        "endpoint": f"{BASE_URL}/datasets/B1610/stream",
        "folder": "B1610",
        "chunk_days": 1,
    },
    "FUELHH": {
        "endpoint": f"{BASE_URL}/datasets/FUELHH/stream",
        "folder": "FUELHH",
        "chunk_days": 7,
    },
    "DAG_B1430": {
        "endpoint": f"{BASE_URL}/datasets/DAG/stream",
        "folder": "DAG_B1430",
        "chunk_days": 7,
    },
    "DGWS_B1440": {
        "endpoint": f"{BASE_URL}/datasets/DGWS/stream",
        "folder": "DGWS_B1440",
        "chunk_days": 7,
    },
}


# ============================================================
# HELPERS
# ============================================================

def extract_records(payload):
    """
    Extract list-like data from common Elexon response shapes.
    Stream endpoints often return either a list or {"data": [...]}.
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
    GET request with simple retry handling.
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
    Creates [start, end) datetime chunks.

    Example:
        2023-01-01T00:00Z to 2023-01-08T00:00Z
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


def download_dataset(dataset_name, config):
    """
    Download one dataset into raw CSV chunks.
    """
    dataset_dir = OUT_DIR / config["folder"]
    dataset_dir.mkdir(parents=True, exist_ok=True)

    endpoint = config["endpoint"]
    chunk_days = config["chunk_days"]

    print(f"\n============================================================")
    print(f"Downloading {dataset_name}")
    print(f"Endpoint: {endpoint}")
    print(f"Chunk size: {chunk_days} day(s)")
    print(f"Output folder: {dataset_dir}")
    print(f"============================================================")

    manifest_rows = []

    for start_dt, end_dt in iter_date_chunks(START_DATE, END_DATE, chunk_days):
        label = f"{start_dt:%Y%m%d}_{end_dt:%Y%m%d}"
        out_file = dataset_dir / f"raw_{dataset_name}_{label}.csv"

        if out_file.exists():
            print(f"Already exists, skipping: {out_file.name}")
            manifest_rows.append({
                "dataset": dataset_name,
                "start": start_dt.isoformat(),
                "end_exclusive": end_dt.isoformat(),
                "file": str(out_file),
                "status": "already_exists",
                "rows": None,
                "columns": None,
            })
            continue

        print(f"Downloading {dataset_name}: {label}")

        try:
            df = fetch_dataset_chunk(endpoint, start_dt, end_dt)

            df.to_csv(out_file, index=False)

            print(f"  Saved: {out_file.name}")
            print(f"  Rows: {len(df)}")
            print(f"  Columns: {df.columns.tolist()}")

            manifest_rows.append({
                "dataset": dataset_name,
                "start": start_dt.isoformat(),
                "end_exclusive": end_dt.isoformat(),
                "file": str(out_file),
                "status": "ok",
                "rows": len(df),
                "columns": "|".join(df.columns.tolist()),
            })

        except Exception as e:
            error_file = dataset_dir / f"ERROR_{dataset_name}_{label}.txt"
            error_file.write_text(str(e), encoding="utf-8")

            print(f"  ERROR saved to: {error_file.name}")
            print(f"  {e}")

            manifest_rows.append({
                "dataset": dataset_name,
                "start": start_dt.isoformat(),
                "end_exclusive": end_dt.isoformat(),
                "file": str(error_file),
                "status": "error",
                "rows": None,
                "columns": None,
            })

        time.sleep(SLEEP_SECONDS)

    manifest = pd.DataFrame(manifest_rows)
    manifest_file = dataset_dir / f"manifest_{dataset_name}.csv"
    manifest.to_csv(manifest_file, index=False)

    print(f"\nManifest saved: {manifest_file}")


# ============================================================
# RUN
# ============================================================


for dataset_name, config in DATASETS.items():
    download_dataset(dataset_name, config)

print("\nDone. Raw data saved in:")
print(OUT_DIR.resolve())