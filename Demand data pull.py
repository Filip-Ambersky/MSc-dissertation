import time
from pathlib import Path
from datetime import datetime, timedelta

import requests
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

YEARS = [2023, 2024, 2025]

BASE_DIR = Path("demand data")
RAW_DIR = BASE_DIR / "raw"
MAPPING_DIR = BASE_DIR / "mapping"
FINAL_DIR = BASE_DIR / "final"

B1610_CHUNK_DAYS = 7
REQUEST_SLEEP_SECONDS = 0.25

# Set this to True only if you want to save every raw B1610 chunk.
# Warning: this can create lots of large files.
SAVE_RAW_B1610_CHUNKS = False

NESO_DATASTORE_URL = "https://api.neso.energy/api/3/action/datastore_search"

NESO_HISTORIC_DEMAND_RESOURCES = {
    2023: "bf5ab335-9b40-4ea4-b93a-ab4af7bce003",
    2024: "f6d02c0f-957b-48cb-82ee-09003f2ba759",
    2025: "b2bde559-3455-4021-b179-dfe60c0337b0",
}

ELEXON_BASE_URL = "https://data.elexon.co.uk/bmrs/api/v1"
ELEXON_BMUNITS_URL = f"{ELEXON_BASE_URL}/reference/bmunits/all"
ELEXON_B1610_STREAM_URL = f"{ELEXON_BASE_URL}/datasets/B1610/stream"


# ============================================================
# HELPERS
# ============================================================

def make_dirs():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    if SAVE_RAW_B1610_CHUNKS:
        (RAW_DIR / "b1610_chunks").mkdir(parents=True, exist_ok=True)


def extract_records(payload):
    """
    Handles common API response shapes:
    - {"data": [...]}
    - {"result": {"records": [...]}}
    - [...]
    """
    if isinstance(payload, list):
        return payload

    if isinstance(payload, dict):
        if "data" in payload and isinstance(payload["data"], list):
            return payload["data"]

        if "items" in payload and isinstance(payload["items"], list):
            return payload["items"]

        if "results" in payload and isinstance(payload["results"], list):
            return payload["results"]

        if "result" in payload:
            result = payload["result"]
            if isinstance(result, dict) and "records" in result:
                return result["records"]

    return []


def request_json(url, params=None, max_retries=5):
    for attempt in range(max_retries):
        response = requests.get(url, params=params, timeout=120)

        if response.status_code == 429:
            wait = 5 * (attempt + 1)
            print(f"Rate limited. Waiting {wait}s...")
            time.sleep(wait)
            continue

        if response.status_code >= 500:
            wait = 5 * (attempt + 1)
            print(f"Server error {response.status_code}. Waiting {wait}s...")
            time.sleep(wait)
            continue

        response.raise_for_status()
        return response.json()

    raise RuntimeError(f"Failed after {max_retries} attempts: {url}")


def find_col(df, candidates, required=True):
    """
    Finds a column using case-insensitive matching.
    """
    lower_map = {c.lower(): c for c in df.columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    if required:
        raise ValueError(
            f"Could not find any of {candidates}. Available columns: {df.columns.tolist()}"
        )

    return None


def clean_key(series):
    return series.astype("string").str.strip().str.upper()


# ============================================================
# NESO DEMAND
# ============================================================

def fetch_neso_historic_demand_year(year):
    """
    Downloads one NESO Historic Demand Data year.
    These yearly files are small enough to request with a large limit.
    """
    out_file = RAW_DIR / f"raw_neso_historic_demand_{year}.csv"

    if out_file.exists():
        print(f"Loading cached NESO demand {year}")
        return pd.read_csv(out_file)

    resource_id = NESO_HISTORIC_DEMAND_RESOURCES[year]

    params = {
        "resource_id": resource_id,
        "limit": 50000,
    }

    print(f"Downloading NESO Historic Demand Data {year}")
    payload = request_json(NESO_DATASTORE_URL, params=params)
    records = extract_records(payload)

    df = pd.DataFrame(records)
    df.to_csv(out_file, index=False)

    return df


def build_ew_demand_table():
    frames = []

    for year in YEARS:
        raw = fetch_neso_historic_demand_year(year)

        raw.columns = [c.strip().upper() for c in raw.columns]

        required = [
            "SETTLEMENT_DATE",
            "SETTLEMENT_PERIOD",
            "ENGLAND_WALES_DEMAND",
        ]

        missing = [c for c in required if c not in raw.columns]
        if missing:
            raise ValueError(f"NESO demand file missing columns: {missing}")

        keep_cols = [
            "SETTLEMENT_DATE",
            "SETTLEMENT_PERIOD",
            "ENGLAND_WALES_DEMAND",
            "ND",
            "TSD",
            "SCOTTISH_TRANSFER",
        ]

        keep_cols = [c for c in keep_cols if c in raw.columns]

        df = raw[keep_cols].copy()

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

        if "ND" in df.columns:
            df["gb_national_demand_mw"] = pd.to_numeric(df["ND"], errors="coerce")

        if "TSD" in df.columns:
            df["gb_transmission_system_demand_mw"] = pd.to_numeric(df["TSD"], errors="coerce")

        if "SCOTTISH_TRANSFER" in df.columns:
            df["scottish_transfer_mw"] = pd.to_numeric(df["SCOTTISH_TRANSFER"], errors="coerce")

        df["timestamp"] = (
            pd.to_datetime(df["settlement_date"].astype(str))
            + pd.to_timedelta((df["settlement_period"] - 1) * 30, unit="min")
        )

        final_cols = [
            "settlement_date",
            "settlement_period",
            "timestamp",
            "england_wales_demand_mw",
            "gb_national_demand_mw",
            "gb_transmission_system_demand_mw",
            "scottish_transfer_mw",
        ]

        final_cols = [c for c in final_cols if c in df.columns]

        frames.append(df[final_cols])

    demand = pd.concat(frames, ignore_index=True)

    demand = (
        demand
        .dropna(subset=["settlement_date", "settlement_period", "england_wales_demand_mw"])
        .sort_values(["settlement_date", "settlement_period"])
        .reset_index(drop=True)
    )

    out_file = MAPPING_DIR / "ew_demand_by_sp.csv"
    demand.to_csv(out_file, index=False)

    print(f"Saved {out_file}")

    return demand


# ============================================================
# BMU MAPPING
# ============================================================

def fetch_bmunits_reference():
    out_file = RAW_DIR / "raw_elexon_bmunits.csv"

    if out_file.exists():
        print("Loading cached Elexon BM Units reference")
        return pd.read_csv(out_file)

    print("Downloading Elexon BM Units reference")
    payload = request_json(ELEXON_BMUNITS_URL, params={"format": "json"})
    records = extract_records(payload)

    df = pd.DataFrame(records)
    df.to_csv(out_file, index=False)

    return df


def build_bmu_region_mapping():
    raw = fetch_bmunits_reference()
    df = raw.copy()

    ngc_col = find_col(df, ["nationalGridBmUnit", "nGCBMUnitID"], required=False)
    elexon_col = find_col(df, ["elexonBmUnit", "bMUnitID", "bmUnit"], required=False)
    fuel_col = find_col(df, ["fuelType"], required=False)
    gsp_id_col = find_col(df, ["gspGroupId"], required=False)
    gsp_name_col = find_col(df, ["gspGroupName"], required=False)
    interconnector_col = find_col(df, ["interconnectorId"], required=False)
    bm_type_col = find_col(df, ["bmUnitType"], required=False)
    lead_party_col = find_col(df, ["leadPartyName"], required=False)
    gen_cap_col = find_col(df, ["generationCapacity"], required=False)
    demand_cap_col = find_col(df, ["demandCapacity"], required=False)
    prod_cons_col = find_col(df, ["productionOrConsumptionFlag"], required=False)

    if gsp_id_col is None:
        raise ValueError("BM Units reference does not contain gspGroupId.")

    mapping = pd.DataFrame()

    mapping["national_grid_bmu"] = clean_key(df[ngc_col]) if ngc_col else pd.NA
    mapping["elexon_bmu"] = clean_key(df[elexon_col]) if elexon_col else pd.NA
    mapping["gsp_group_id"] = clean_key(df[gsp_id_col])
    mapping["gsp_group_name"] = df[gsp_name_col] if gsp_name_col else pd.NA
    mapping["fuel_type"] = df[fuel_col] if fuel_col else pd.NA
    mapping["lead_party_name"] = df[lead_party_col] if lead_party_col else pd.NA
    mapping["bm_unit_type"] = df[bm_type_col] if bm_type_col else pd.NA
    mapping["generation_capacity"] = df[gen_cap_col] if gen_cap_col else pd.NA
    mapping["demand_capacity"] = df[demand_cap_col] if demand_cap_col else pd.NA
    mapping["production_or_consumption_flag"] = df[prod_cons_col] if prod_cons_col else pd.NA

    if interconnector_col:
        mapping["interconnector_id"] = df[interconnector_col]
    else:
        mapping["interconnector_id"] = pd.NA

    mapping["is_interconnector"] = (
        mapping["interconnector_id"].astype("string").notna()
        & (mapping["interconnector_id"].astype("string").str.strip() != "")
    )

    mapping["region"] = "England_Wales"
    mapping.loc[mapping["gsp_group_id"] == "_N", "region"] = "South_Scotland"
    mapping.loc[mapping["gsp_group_id"] == "_P", "region"] = "North_Scotland"

    # If GSP is missing, keep it out of the E&W aggregate unless manually reviewed.
    mapping.loc[
        mapping["gsp_group_id"].isna()
        | (mapping["gsp_group_id"].astype("string").str.strip() == ""),
        "region"
    ] = "Unknown"

    mapping["include_in_ew_generation"] = (
        (mapping["region"] == "England_Wales")
        & (~mapping["is_interconnector"])
    )

    mapping = mapping.drop_duplicates()

    out_file = MAPPING_DIR / "bmu_region_mapping.csv"
    mapping.to_csv(out_file, index=False)

    ew_file = MAPPING_DIR / "ew_bmu_mapping.csv"
    mapping[mapping["include_in_ew_generation"]].to_csv(ew_file, index=False)

    print(f"Saved {out_file}")
    print(f"Saved {ew_file}")

    return mapping


# ============================================================
# ELEXON B1610 GENERATION
# ============================================================

def iter_date_chunks(start_dt, end_dt, chunk_days):
    current = start_dt

    while current < end_dt:
        chunk_end = min(current + timedelta(days=chunk_days), end_dt)
        yield current, chunk_end
        current = chunk_end


def fetch_b1610_chunk(start_dt, end_dt):
    """
    B1610 stream.
    Uses from/to datetime and asks for settlement periods 1-50
    to handle clock-change days.
    """
    params = {
        "from": start_dt.strftime("%Y-%m-%dT%H:%MZ"),
        "to": end_dt.strftime("%Y-%m-%dT%H:%MZ"),
        "settlementPeriodFrom": 1,
        "settlementPeriodTo": 50,
        "format": "json",
    }

    payload = request_json(ELEXON_B1610_STREAM_URL, params=params)
    records = extract_records(payload)

    return pd.DataFrame(records)


def enrich_b1610_with_mapping(b1610_df, mapping):
    df = b1610_df.copy()

    settlement_date_col = find_col(df, ["settlementDate", "settlement_date"])
    settlement_period_col = find_col(df, ["settlementPeriod", "settlement_period"])
    quantity_col = find_col(df, ["quantity", "meteredVolume", "volume"])
    ngc_col = find_col(df, ["nGCBMUnitID", "nationalGridBmUnit", "ngcBmUnitId"], required=False)
    bmu_col = find_col(df, ["bMUnitID", "elexonBmUnit", "bmUnit"], required=False)

    if ngc_col is None and bmu_col is None:
        raise ValueError("B1610 data has no usable BMU ID column.")

    df["settlement_date"] = pd.to_datetime(
        df[settlement_date_col],
        errors="coerce"
    ).dt.date

    df["settlement_period"] = pd.to_numeric(
        df[settlement_period_col],
        errors="coerce"
    ).astype("Int64")

    df["quantity_mwh"] = pd.to_numeric(
        df[quantity_col],
        errors="coerce"
    )

    df["_ngc_key"] = clean_key(df[ngc_col]) if ngc_col else pd.NA
    df["_elexon_key"] = clean_key(df[bmu_col]) if bmu_col else pd.NA

    mapping_ngc = (
        mapping
        .dropna(subset=["national_grid_bmu"])
        .drop_duplicates(subset=["national_grid_bmu"])
        .rename(columns={"national_grid_bmu": "_ngc_key"})
    )

    merged = df.merge(
        mapping_ngc[
            [
                "_ngc_key",
                "region",
                "gsp_group_id",
                "gsp_group_name",
                "fuel_type",
                "include_in_ew_generation",
                "is_interconnector",
            ]
        ],
        on="_ngc_key",
        how="left",
    )

    # Fallback join using Elexon BMU ID where NGC BMU join failed.
    if "_elexon_key" in merged.columns:
        need_fallback = merged["region"].isna() & merged["_elexon_key"].notna()

        if need_fallback.any():
            mapping_elexon = (
                mapping
                .dropna(subset=["elexon_bmu"])
                .drop_duplicates(subset=["elexon_bmu"])
                .rename(columns={"elexon_bmu": "_elexon_key"})
            )

            fallback = merged.loc[need_fallback, ["_elexon_key"]].merge(
                mapping_elexon[
                    [
                        "_elexon_key",
                        "region",
                        "gsp_group_id",
                        "gsp_group_name",
                        "fuel_type",
                        "include_in_ew_generation",
                        "is_interconnector",
                    ]
                ],
                on="_elexon_key",
                how="left",
            )

            fill_cols = [
                "region",
                "gsp_group_id",
                "gsp_group_name",
                "fuel_type",
                "include_in_ew_generation",
                "is_interconnector",
            ]

            merged.loc[need_fallback, fill_cols] = fallback[fill_cols].values

    merged["include_in_ew_generation"] = merged["include_in_ew_generation"].fillna(False)

    return merged


def build_ew_generation_table(mapping):
    all_agg = []

    for year in YEARS:
        year_start = datetime(year, 1, 1)
        year_end = datetime(year + 1, 1, 1)

        print(f"Downloading and aggregating B1610 for {year}")

        for chunk_start, chunk_end in iter_date_chunks(year_start, year_end, B1610_CHUNK_DAYS):
            label = f"{chunk_start:%Y%m%d}_{chunk_end:%Y%m%d}"
            print(f"  B1610 chunk {label}")

            raw_chunk = fetch_b1610_chunk(chunk_start, chunk_end)

            if raw_chunk.empty:
                print("    empty chunk")
                continue

            if SAVE_RAW_B1610_CHUNKS:
                raw_chunk_file = RAW_DIR / "b1610_chunks" / f"raw_elexon_b1610_{label}.csv"
                raw_chunk.to_csv(raw_chunk_file, index=False)

            enriched = enrich_b1610_with_mapping(raw_chunk, mapping)

            ew = enriched[
                (enriched["include_in_ew_generation"])
                & (enriched["quantity_mwh"].notna())
            ].copy()

            # Only positive metered volumes are counted as generation.
            # Negative values represent import/consumption behaviour.
            ew["positive_generation_mwh"] = ew["quantity_mwh"].clip(lower=0)

            agg = (
                ew
                .groupby(["settlement_date", "settlement_period"], as_index=False)
                .agg(
                    england_wales_generation_mwh=("positive_generation_mwh", "sum"),
                    bmu_count_used=("positive_generation_mwh", lambda x: (x > 0).sum()),
                    bmu_rows_seen=("positive_generation_mwh", "size"),
                )
            )

            all_agg.append(agg)

            time.sleep(REQUEST_SLEEP_SECONDS)

    if not all_agg:
        raise RuntimeError("No B1610 generation data was downloaded/aggregated.")

    generation = pd.concat(all_agg, ignore_index=True)

    # In case chunks overlap or duplicate, aggregate again.
    generation = (
        generation
        .groupby(["settlement_date", "settlement_period"], as_index=False)
        .agg(
            england_wales_generation_mwh=("england_wales_generation_mwh", "sum"),
            bmu_count_used=("bmu_count_used", "sum"),
            bmu_rows_seen=("bmu_rows_seen", "sum"),
        )
        .sort_values(["settlement_date", "settlement_period"])
        .reset_index(drop=True)
    )

    generation["england_wales_generation_mw"] = generation["england_wales_generation_mwh"] * 2

    out_file = MAPPING_DIR / "ew_generation_by_sp.csv"
    generation.to_csv(out_file, index=False)

    print(f"Saved {out_file}")

    return generation


# ============================================================
# FINAL RESIDUAL DEMAND
# ============================================================

def build_final_residual_demand(ew_demand, ew_generation):
    final = ew_demand.merge(
        ew_generation,
        on=["settlement_date", "settlement_period"],
        how="left",
    )

    final["missing_generation_flag"] = final["england_wales_generation_mwh"].isna()

    # If generation is missing, leave residual as NaN to avoid false values.
    final["ew_residual_demand_mw"] = (
        final["england_wales_demand_mw"]
        - final["england_wales_generation_mw"]
    )

    final["ew_residual_demand_clipped_mw"] = final["ew_residual_demand_mw"].clip(lower=0)

    final = final.sort_values(["settlement_date", "settlement_period"]).reset_index(drop=True)

    final_cols = [
        "settlement_date",
        "settlement_period",
        "timestamp",
        "england_wales_demand_mw",
        "england_wales_generation_mwh",
        "england_wales_generation_mw",
        "ew_residual_demand_mw",
        "ew_residual_demand_clipped_mw",
        "bmu_count_used",
        "bmu_rows_seen",
        "missing_generation_flag",
        "gb_national_demand_mw",
        "gb_transmission_system_demand_mw",
        "scottish_transfer_mw",
    ]

    final_cols = [c for c in final_cols if c in final.columns]

    final = final[final_cols]

    out_file = FINAL_DIR / "ew_residual_demand_half_hourly.csv"
    final.to_csv(out_file, index=False)

    print(f"Saved final output: {out_file}")
    print(final.head())

    return final


# ============================================================
# MAIN
# ============================================================

def main():
    make_dirs()

    print("Step 1: Build England & Wales demand table")
    ew_demand = build_ew_demand_table()

    print("\nStep 2: Build BMU region mapping")
    bmu_mapping = build_bmu_region_mapping()

    print("\nStep 3: Build England & Wales generation table from B1610")
    ew_generation = build_ew_generation_table(bmu_mapping)

    print("\nStep 4: Build final residual demand table")
    build_final_residual_demand(ew_demand, ew_generation)

    print("\nDone.")


if __name__ == "__main__":
    main()