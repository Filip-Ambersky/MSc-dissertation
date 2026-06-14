import time
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

YEARS = [2023, 2024, 2025]

BASE_DIR = Path("demand data")
RAW_DIR = BASE_DIR / "raw"
MAPPING_DIR = BASE_DIR / "mapping"
FINAL_DIR = BASE_DIR / "final"

# B1610 is very large, so request it in smaller time chunks
B1610_CHUNK_DAYS = 7

# Also request B1610 only for batches of BMUs, not all BMUs at once
BMU_BATCH_SIZE = 100

# Pause between requests to avoid hammering the API
REQUEST_SLEEP_SECONDS = 0.25


# ============================================================
# API URLS
# ============================================================

NESO_DATASTORE_URL = "https://api.neso.energy/api/3/action/datastore_search"

ELEXON_BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
ELEXON_BMUNITS_URL = f"{ELEXON_BASE_URL}/reference/bmunits/all"
ELEXON_B1610_STREAM_URL = f"{ELEXON_BASE_URL}/datasets/B1610/stream"


# NESO resource IDs confirmed from the Historic Demand Data pages
NESO_HISTORIC_DEMAND_RESOURCES = {
    2023: "bf5ab335-9b40-4ea4-b93a-ab4af7bce003",
    2024: "f6d02c0f-957b-48cb-82ee-09003f2ba759",
    2025: "b2bde559-3455-4021-b179-dfe60c0337b0",
}


# ============================================================
# GENERAL HELPERS
# ============================================================

def make_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    (MAPPING_DIR / "b1610_ew_generation_chunks").mkdir(parents=True, exist_ok=True)


def request_json(url, params=None, max_retries=5):
    """
    Send a GET request and return JSON.
    Retries on rate limits or temporary server errors.
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

        response.raise_for_status()
        return response.json()

    raise RuntimeError(f"Failed after {max_retries} attempts: {url}")


def extract_records(payload):
    """
    Extract rows from common API response shapes.
    """

    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]

        if "result" in payload and isinstance(payload["result"], dict):
            if "records" in payload["result"]:
                return payload["result"]["records"]

    return []


def clean_key(series):
    """
    Cleans BMU IDs for safer joins.
    """

    return series.astype("string").str.strip().str.upper()


def iter_date_chunks(start_dt, end_dt, chunk_days):
    """
    Splits a date range into smaller chunks.
    """

    current = start_dt

    while current < end_dt:
        chunk_end = min(current + timedelta(days=chunk_days), end_dt)
        yield current, chunk_end
        current = chunk_end


def batch_list(items, batch_size):
    """
    Splits a list into batches.
    """

    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


# ============================================================
# 1. NESO ENGLAND & WALES DEMAND
# ============================================================

def download_neso_historic_demand(year):
    """
    Downloads raw NESO Historic Demand Data for one year.
    """

    output_file = RAW_DIR / f"raw_neso_historic_demand_{year}.csv"

    if output_file.exists():
        print(f"Loading cached NESO demand {year}")
        return pd.read_csv(output_file)

    resource_id = NESO_HISTORIC_DEMAND_RESOURCES[year]

    params = {
        "resource_id": resource_id,
        "limit": 50000,
    }

    print(f"Downloading NESO Historic Demand Data {year}")
    payload = request_json(NESO_DATASTORE_URL, params=params)
    records = extract_records(payload)

    df = pd.DataFrame(records)

    df.to_csv(output_file, index=False)
    print(f"Saved {output_file}")

    return df


def build_ew_demand_table():
    """
    Creates:
        demand data/mapping/ew_demand_by_sp.csv

    Main columns:
        settlement_date
        settlement_period
        timestamp
        england_wales_demand_mw
    """

    frames = []

    for year in YEARS:
        raw = download_neso_historic_demand(year)

        required_cols = [
            "SETTLEMENT_DATE",
            "SETTLEMENT_PERIOD",
            "ENGLAND_WALES_DEMAND",
        ]

        missing = [c for c in required_cols if c not in raw.columns]
        if missing:
            raise ValueError(f"NESO demand data missing columns: {missing}")

        df = raw.copy()

        df["settlement_date"] = pd.to_datetime(
            df["SETTLEMENT_DATE"],
            dayfirst=True,
            errors="coerce"
        ).dt.date

        df["settlement_period"] = pd.to_numeric(
            df["SETTLEMENT_PERIOD"],
            errors="coerce"
        ).astype("Int64")

        df["england_wales_demand_mw"] = pd.to_numeric(
            df["ENGLAND_WALES_DEMAND"],
            errors="coerce"
        )

        df["timestamp"] = (
            pd.to_datetime(df["settlement_date"].astype(str))
            + pd.to_timedelta((df["settlement_period"] - 1) * 30, unit="min")
        )

        # Optional useful context columns
        optional_columns = {}

        if "ND" in df.columns:
            optional_columns["gb_national_demand_mw"] = pd.to_numeric(df["ND"], errors="coerce")

        if "TSD" in df.columns:
            optional_columns["gb_transmission_system_demand_mw"] = pd.to_numeric(df["TSD"], errors="coerce")

        if "SCOTTISH_TRANSFER" in df.columns:
            optional_columns["scottish_transfer_mw"] = pd.to_numeric(df["SCOTTISH_TRANSFER"], errors="coerce")

        for col_name, values in optional_columns.items():
            df[col_name] = values

        keep_cols = [
            "settlement_date",
            "settlement_period",
            "timestamp",
            "england_wales_demand_mw",
            "gb_national_demand_mw",
            "gb_transmission_system_demand_mw",
            "scottish_transfer_mw",
        ]

        keep_cols = [c for c in keep_cols if c in df.columns]

        frames.append(df[keep_cols])

    demand = pd.concat(frames, ignore_index=True)

    demand = (
        demand
        .dropna(subset=["settlement_date", "settlement_period", "england_wales_demand_mw"])
        .sort_values(["settlement_date", "settlement_period"])
        .reset_index(drop=True)
    )

    output_file = MAPPING_DIR / "ew_demand_by_sp.csv"
    demand.to_csv(output_file, index=False)

    print(f"Saved {output_file}")

    return demand


# ============================================================
# 2. BMU REGION MAPPING
# ============================================================

def download_bmunits_reference():
    """
    Downloads Elexon BM Units reference data.
    """

    output_file = RAW_DIR / "raw_elexon_bmunits.csv"

    if output_file.exists():
        print("Loading cached Elexon BM Units reference")
        return pd.read_csv(output_file)

    print("Downloading Elexon BM Units reference")

    payload = request_json(
        ELEXON_BMUNITS_URL,
        params={"format": "json"}
    )

    records = extract_records(payload)
    df = pd.DataFrame(records)

    df.to_csv(output_file, index=False)
    print(f"Saved {output_file}")

    return df


def build_bmu_region_mapping():
    """
    Creates:
        demand data/mapping/bmu_region_mapping.csv
        demand data/mapping/ew_generation_bmu_mapping.csv

    Uses:
        gspGroupId == "_N"  -> South Scotland
        gspGroupId == "_P"  -> North Scotland
        everything else     -> England & Wales
    """

    raw = download_bmunits_reference()

    required_cols = [
        "nationalGridBmUnit",
        "elexonBmUnit",
        "fuelType",
        "gspGroupId",
        "gspGroupName",
        "interconnectorId",
        "generationCapacity",
        "productionOrConsumptionFlag",
        "bmUnitType",
    ]

    missing = [c for c in required_cols if c not in raw.columns]
    if missing:
        raise ValueError(f"Elexon BMU reference missing columns: {missing}")

    mapping = pd.DataFrame()

    mapping["national_grid_bmu"] = clean_key(raw["nationalGridBmUnit"])
    mapping["elexon_bmu"] = clean_key(raw["elexonBmUnit"])
    mapping["fuel_type"] = raw["fuelType"]
    mapping["lead_party_name"] = raw["leadPartyName"]
    mapping["bm_unit_name"] = raw["bmUnitName"]
    mapping["bm_unit_type"] = raw["bmUnitType"]
    mapping["gsp_group_id"] = clean_key(raw["gspGroupId"])
    mapping["gsp_group_name"] = raw["gspGroupName"]
    mapping["interconnector_id"] = raw["interconnectorId"]
    mapping["production_or_consumption_flag"] = raw["productionOrConsumptionFlag"]

    mapping["generation_capacity_mw"] = pd.to_numeric(
        raw["generationCapacity"],
        errors="coerce"
    )

    mapping["demand_capacity_mw"] = pd.to_numeric(
        raw["demandCapacity"],
        errors="coerce"
    )

    # Basic regional split
    mapping["region"] = "England_Wales"
    mapping.loc[mapping["gsp_group_id"] == "_N", "region"] = "South_Scotland"
    mapping.loc[mapping["gsp_group_id"] == "_P", "region"] = "North_Scotland"

    # Missing GSP group cannot be safely assigned
    mapping.loc[
        mapping["gsp_group_id"].isna()
        | (mapping["gsp_group_id"].astype("string").str.strip() == ""),
        "region"
    ] = "Unknown"

    # Interconnectors are not domestic England/Wales generation
    mapping["is_interconnector"] = (
        mapping["interconnector_id"].astype("string").notna()
        & (mapping["interconnector_id"].astype("string").str.strip() != "")
    )

    # Generation-like BMU filter.
    # This avoids counting supplier/demand BMUs as generation.
    mapping["is_generation_like"] = (
        (mapping["generation_capacity_mw"].fillna(0) > 0)
        | mapping["fuel_type"].notna()
    )

    mapping["include_in_ew_generation"] = (
        (mapping["region"] == "England_Wales")
        & (~mapping["is_interconnector"])
        & (mapping["is_generation_like"])
    )

    mapping = mapping.drop_duplicates()

    output_file = MAPPING_DIR / "bmu_region_mapping.csv"
    mapping.to_csv(output_file, index=False)

    ew_mapping = mapping[mapping["include_in_ew_generation"]].copy()

    ew_output_file = MAPPING_DIR / "ew_generation_bmu_mapping.csv"
    ew_mapping.to_csv(ew_output_file, index=False)

    print(f"Saved {output_file}")
    print(f"Saved {ew_output_file}")
    print(f"England/Wales generation BMUs used: {len(ew_mapping)}")

    return mapping, ew_mapping


# ============================================================
# 3. ELEXON B1610 ENGLAND & WALES GENERATION
# ============================================================

def fetch_b1610_stream_chunk(start_dt, end_dt, bm_units):
    """
    Downloads one B1610 chunk for a batch of BMUs.

    Important actual B1610 columns from your test:
        bmUnit
        nationalGridBmUnitId
        settlementDate
        settlementPeriod
        quantity
    """

    params = [
        ("from", start_dt.strftime("%Y-%m-%dT%H:%MZ")),
        ("to", end_dt.strftime("%Y-%m-%dT%H:%MZ")),
        ("settlementPeriodFrom", 1),
        ("settlementPeriodTo", 50),
        ("format", "json"),
    ]

    # Repeat bmUnit parameter for each BMU in the batch.
    # This avoids downloading all BMUs in one huge response.
    for bmu in bm_units:
        params.append(("bmUnit", bmu))

    payload = request_json(ELEXON_B1610_STREAM_URL, params=params)
    records = extract_records(payload)

    return pd.DataFrame(records)


def aggregate_b1610_chunk_to_ew_generation(raw_chunk):
    """
    Aggregates one B1610 response to generation by settlement period.

    B1610 quantity is MWh per settlement period.
    We count positive values as generation.
    """

    if raw_chunk.empty:
        return pd.DataFrame(
            columns=[
                "settlement_date",
                "settlement_period",
                "england_wales_generation_mwh",
                "b1610_rows_used",
                "bmu_count_used",
            ]
        )

    required_cols = [
        "bmUnit",
        "settlementDate",
        "settlementPeriod",
        "quantity",
    ]

    missing = [c for c in required_cols if c not in raw_chunk.columns]
    if missing:
        raise ValueError(f"B1610 response missing columns: {missing}")

    df = raw_chunk.copy()

    df["settlement_date"] = pd.to_datetime(
        df["settlementDate"],
        errors="coerce"
    ).dt.date

    df["settlement_period"] = pd.to_numeric(
        df["settlementPeriod"],
        errors="coerce"
    ).astype("Int64")

    df["quantity_mwh"] = pd.to_numeric(
        df["quantity"],
        errors="coerce"
    )

    df["bmu"] = clean_key(df["bmUnit"])

    # Only positive metered volumes are treated as generation.
    # Negative values are import/consumption behaviour.
    df["positive_generation_mwh"] = df["quantity_mwh"].clip(lower=0)

    agg = (
        df
        .groupby(["settlement_date", "settlement_period"], as_index=False)
        .agg(
            england_wales_generation_mwh=("positive_generation_mwh", "sum"),
            b1610_rows_used=("positive_generation_mwh", "size"),
            bmu_count_used=("bmu", "nunique"),
        )
    )

    return agg


def build_ew_generation_table(ew_mapping):
    """
    Creates:
        demand data/mapping/ew_generation_by_sp.csv

    It fetches B1610 only for England/Wales generation BMUs.
    This is much smaller than requesting all BMUs.
    """

    # Use Elexon BMU IDs for the B1610 bmUnit parameter
    ew_bmus = (
        ew_mapping["elexon_bmu"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    if not ew_bmus:
        raise RuntimeError("No England/Wales BMUs found for generation aggregation.")

    all_chunk_files = []

    for year in YEARS:
        year_start = datetime(year, 1, 1)
        year_end = datetime(year + 1, 1, 1)

        for chunk_start, chunk_end in iter_date_chunks(year_start, year_end, B1610_CHUNK_DAYS):
            chunk_label = f"{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}"

            for batch_number, bmu_batch in enumerate(batch_list(ew_bmus, BMU_BATCH_SIZE), start=1):
                output_file = (
                    MAPPING_DIR
                    / "b1610_ew_generation_chunks"
                    / f"ew_b1610_agg_{chunk_label}_batch_{batch_number:03d}.csv"
                )

                all_chunk_files.append(output_file)

                if output_file.exists():
                    print(f"Loading cached {output_file.name}")
                    continue

                print(
                    f"Downloading B1610 {chunk_label}, "
                    f"BMU batch {batch_number}, "
                    f"{len(bmu_batch)} BMUs"
                )

                raw_chunk = fetch_b1610_stream_chunk(
                    start_dt=chunk_start,
                    end_dt=chunk_end,
                    bm_units=bmu_batch,
                )

                agg_chunk = aggregate_b1610_chunk_to_ew_generation(raw_chunk)
                agg_chunk.to_csv(output_file, index=False)

                time.sleep(REQUEST_SLEEP_SECONDS)

    # Load all saved aggregate chunk files
    frames = []

    for file in all_chunk_files:
        if file.exists():
            part = pd.read_csv(file)

            if not part.empty:
                frames.append(part)

    if not frames:
        raise RuntimeError("No B1610 aggregate chunks were created.")

    generation = pd.concat(frames, ignore_index=True)

    generation["settlement_date"] = pd.to_datetime(
        generation["settlement_date"],
        errors="coerce"
    ).dt.date

    generation["settlement_period"] = pd.to_numeric(
        generation["settlement_period"],
        errors="coerce"
    ).astype("Int64")

    generation["england_wales_generation_mwh"] = pd.to_numeric(
        generation["england_wales_generation_mwh"],
        errors="coerce"
    )

    # Combine all BMU batches and time chunks
    generation = (
        generation
        .groupby(["settlement_date", "settlement_period"], as_index=False)
        .agg(
            england_wales_generation_mwh=("england_wales_generation_mwh", "sum"),
            b1610_rows_used=("b1610_rows_used", "sum"),
            bmu_count_used=("bmu_count_used", "sum"),
        )
        .sort_values(["settlement_date", "settlement_period"])
        .reset_index(drop=True)
    )

    # Settlement periods are half-hourly, so MWh × 2 = average MW
    generation["england_wales_generation_mw"] = (
        generation["england_wales_generation_mwh"] * 2
    )

    output_file = MAPPING_DIR / "ew_generation_by_sp.csv"
    generation.to_csv(output_file, index=False)

    print(f"Saved {output_file}")

    return generation


# ============================================================
# 4. FINAL RESIDUAL DEMAND TABLE
# ============================================================

def build_final_residual_demand(ew_demand, ew_generation):
    """
    Creates:
        demand data/final/ew_residual_demand_half_hourly.csv

    Formula:
        residual demand = England/Wales demand - England/Wales generation
    """

    final = ew_demand.merge(
        ew_generation,
        on=["settlement_date", "settlement_period"],
        how="left",
    )

    final["missing_generation_flag"] = final["england_wales_generation_mwh"].isna()

    final["ew_residual_demand_mw"] = (
        final["england_wales_demand_mw"]
        - final["england_wales_generation_mw"]
    )

    final["ew_residual_demand_clipped_mw"] = (
        final["ew_residual_demand_mw"].clip(lower=0)
    )

    final = (
        final
        .sort_values(["settlement_date", "settlement_period"])
        .reset_index(drop=True)
    )

    final_cols = [
        "settlement_date",
        "settlement_period",
        "timestamp",
        "england_wales_demand_mw",
        "england_wales_generation_mwh",
        "england_wales_generation_mw",
        "ew_residual_demand_mw",
        "ew_residual_demand_clipped_mw",
        "b1610_rows_used",
        "bmu_count_used",
        "missing_generation_flag",
        "gb_national_demand_mw",
        "gb_transmission_system_demand_mw",
        "scottish_transfer_mw",
    ]

    final_cols = [c for c in final_cols if c in final.columns]

    final = final[final_cols]

    output_file = FINAL_DIR / "ew_residual_demand_half_hourly.csv"
    final.to_csv(output_file, index=False)

    print(f"Saved final file: {output_file}")
    print(final.head())

    return final


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    make_dirs()

    print("\n1. Building England & Wales demand table")
    ew_demand = build_ew_demand_table()

    print("\n2. Building BMU region mapping")
    bmu_mapping, ew_mapping = build_bmu_region_mapping()

    print("\n3. Building England & Wales generation table from B1610")
    ew_generation = build_ew_generation_table(ew_mapping)

    print("\n4. Building final England & Wales residual demand table")
    build_final_residual_demand(ew_demand, ew_generation)

    print("\nDone.")


if __name__ == "__main__":
    main()