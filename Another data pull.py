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

# These stream endpoints are downloaded in internal chunks,
# but only ONE final CSV is saved per dataset.
STREAM_DATASETS = {
    "TSDF": {
        "endpoint": f"{BASE_URL}/datasets/TSDF/stream",
        "output_file": OUT_DIR / "raw_TSDF_2023_2025.csv",
        "chunk_days": 7,
    },
    "B1610": {
        "endpoint": f"{BASE_URL}/datasets/B1610/stream",
        "output_file": OUT_DIR / "raw_B1610_2023_2025.csv",
        "chunk_days": 1,
    },
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
    "PN_STREAM": {
        "endpoint": f"{BASE_URL}/datasets/PN/stream",
        "output_file": OUT_DIR / "raw_PN_STREAM_2023_2025.csv",
        "chunk_days": 1,
    },
}

# Market-wide physical notification endpoint.
# This is slow because it requests one settlement period at a time.
FPN_MARKET_WIDE_CONFIG = {
    "endpoint": f"{BASE_URL}/balancing/physical/all",
    "output_file": OUT_DIR / "raw_FPN_MARKET_WIDE_2023_2025.csv",
    "dataset": "PN",
}


# ============================================================
# HELPERS
# ============================================================

def extract_records(payload):
    """
    Extract records from common Elexon response shapes.
    """
    if payload is None:
        return []

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        for key in ["data", "items", "results"]:
            if key in payload and isinstance(payload[key], list):
                return payload[key]

    return []


def request_json(url, params, max_retries=MAX_RETRIES, allow_empty_statuses=False):
    """
    GET request with retries.
    If allow_empty_statuses=True, 400/404 are treated as empty responses.
    This is useful for settlement periods that do not exist on normal days.
    """
    for attempt in range(max_retries):
        response = requests.get(url, params=params, timeout=180)

        if allow_empty_statuses and response.status_code in [400, 404]:
            return None

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


def iter_dates(start_date, end_date):
    """
    Iterates over calendar dates.
    """
    current = datetime.strptime(start_date, "%Y-%m-%d").date()
    end = datetime.strptime(end_date, "%Y-%m-%d").date()

    while current <= end:
        yield current
        current += timedelta(days=1)


def append_df_to_csv(df, output_file, header_written, first_columns):
    """
    Append a DataFrame to one final CSV while keeping a consistent column order.
    """
    if df.empty:
        return header_written, first_columns

    if first_columns is None:
        first_columns = df.columns.tolist()
    else:
        for col in first_columns:
            if col not in df.columns:
                df[col] = pd.NA

        # Keep only the original schema so the CSV structure does not break.
        df = df[first_columns]

    df.to_csv(
        output_file,
        mode="a",
        index=False,
        header=not header_written,
    )

    return True, first_columns


# ============================================================
# STREAM DATASET DOWNLOAD
# ============================================================

def fetch_stream_chunk(endpoint, start_dt, end_dt):
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


def download_stream_dataset_to_single_csv(dataset_name, config):
    endpoint = config["endpoint"]
    output_file = config["output_file"]
    chunk_days = config["chunk_days"]

    if output_file.exists():
        print(f"\nDeleting existing file so it can be rebuilt: {output_file}")
        output_file.unlink()

    print("\n============================================================")
    print(f"Downloading stream dataset: {dataset_name}")
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
            df = fetch_stream_chunk(endpoint, start_dt, end_dt)

            if df.empty:
                print("  Empty response")
                time.sleep(SLEEP_SECONDS)
                continue

            header_written, first_columns = append_df_to_csv(
                df=df,
                output_file=output_file,
                header_written=header_written,
                first_columns=first_columns,
            )

            total_rows += len(df)

            print(f"  Appended rows: {len(df)}")
            print(f"  Total rows so far: {total_rows}")

        except Exception as e:
            print(f"  ERROR for {dataset_name}, {label}")
            print(f"  {e}")

        time.sleep(SLEEP_SECONDS)

    print(f"\nFinished {dataset_name}")
    print(f"Saved: {output_file}")
    print(f"Total rows written: {total_rows}")


# ============================================================
# FPN / MARKET-WIDE PHYSICAL PN DOWNLOAD
# ============================================================

def fetch_market_wide_physical_pn(settlement_date, settlement_period):
    """
    Pull market-wide physical PN data for one settlement date and settlement period.

    This uses:
        /balancing/physical/all
        dataset=PN

    In the Insights/BSC wording, the public PN/FPN-style physical notification
    displayed at Gate Closure is the relevant final physical position.
    """
    params = {
        "dataset": "PN",
        "settlementDate": settlement_date,
        "settlementPeriod": settlement_period,
        "format": "json",
    }

    payload = request_json(
        FPN_MARKET_WIDE_CONFIG["endpoint"],
        params=params,
        allow_empty_statuses=True,
    )

    records = extract_records(payload)
    return pd.DataFrame(records)


def download_fpn_market_wide_to_single_csv():
    output_file = FPN_MARKET_WIDE_CONFIG["output_file"]

    if output_file.exists():
        print(f"\nDeleting existing file so it can be rebuilt: {output_file}")
        output_file.unlink()

    print("\n============================================================")
    print("Downloading FPN_MARKET_WIDE")
    print(f"Endpoint: {FPN_MARKET_WIDE_CONFIG['endpoint']}")
    print("Dataset parameter: PN")
    print(f"Output file: {output_file}")
    print("Internal loop: every date and settlement period 1-50")
    print("============================================================")

    header_written = False
    total_rows = 0
    first_columns = None

    for day in iter_dates(START_DATE, END_DATE):
        settlement_date = day.isoformat()
        print(f"Downloading FPN_MARKET_WIDE for {settlement_date}")

        # Use 1-50 to cover clock-change days.
        for sp in range(1, 51):
            try:
                df = fetch_market_wide_physical_pn(
                    settlement_date=settlement_date,
                    settlement_period=sp,
                )

                if df.empty:
                    continue

                header_written, first_columns = append_df_to_csv(
                    df=df,
                    output_file=output_file,
                    header_written=header_written,
                    first_columns=first_columns,
                )

                total_rows += len(df)

            except Exception as e:
                print(f"  ERROR for {settlement_date} SP{sp}")
                print(f"  {e}")

            time.sleep(SLEEP_SECONDS)

        print(f"  Total rows so far: {total_rows}")

    print("\nFinished FPN_MARKET_WIDE")
    print(f"Saved: {output_file}")
    print(f"Total rows written: {total_rows}")


# ============================================================
# RUN ALL DATASETS
# ============================================================

def main():
    for dataset_name, config in STREAM_DATASETS.items():
        download_stream_dataset_to_single_csv(dataset_name, config)

    download_fpn_market_wide_to_single_csv()

    print("\nDone. Final raw CSVs saved in:")
    print(OUT_DIR.resolve())


if __name__ == "__main__":
    main()