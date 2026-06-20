from __future__ import annotations

import html
import io
import json
import re
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import geopandas as gpd
import numpy as np
import pandas as pd
import requests
from shapely.geometry import LineString, Point
from shapely.ops import nearest_points, unary_union


# =============================================================================
# CONFIG
# =============================================================================

OUTPUT_DIR = Path("bmu_b5_b6_output")
OUTPUT_DIR.mkdir(exist_ok=True)

OSUKED_BASE = "https://osuked.github.io/Power-Station-Dictionary"

# Main OSUKED object-attribute table:
# dictionary_id, common_name, settlement_bmu_id, national_grid_bmu_id, EIC, REPD, CfD, etc.
OSUKED_OBJECT_ATTRS_URL = (
    f"{OSUKED_BASE}/object_attrs/dictionary_attributes.csv"
)

# NESO ArcGIS ETYS boundary layer.
# This is the boundary polyline layer with field Boundary_n.
NESO_BOUNDARY_LAYER_QUERY_URL = (
    "https://services-eu1.arcgis.com/89ur7pRPzrA8NWtj/"
    "ArcGIS/rest/services/CP30_NESO_Map_WFL/FeatureServer/0/query"
)

# The only three zones you asked for.
ZONE_NORTH_OF_B5 = "NORTH_OF_B5"
ZONE_BETWEEN_B5_B6 = "BETWEEN_B5_AND_B6"
ZONE_SOUTH_OF_B6 = "SOUTH_OF_B6"


# =============================================================================
# BASIC HELPERS
# =============================================================================

def clean_col(col: str) -> str:
    """
    Convert messy source columns to snake_case.

    Examples:
        "National Grid BMU ID" -> "national_grid_bmu_id"
        "Fuel Type" -> "fuel_type"
        "NGC_BMU_ID" -> "ngc_bmu_id"
    """
    col = str(col).strip()
    col = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", col)
    col = re.sub(r"[^0-9a-zA-Z]+", "_", col)
    return col.strip("_").lower()


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    return df.rename(columns={c: clean_col(c) for c in df.columns})


def request_get(url: str, params: dict[str, Any] | None = None) -> requests.Response:
    r = requests.get(
        url,
        params=params,
        timeout=90,
        headers={"User-Agent": "osuked-bmu-boundary-classifier/1.0"},
    )
    r.raise_for_status()
    return r


def read_csv_url(url: str) -> pd.DataFrame:
    r = request_get(url)
    return clean_columns(pd.read_csv(io.StringIO(r.text)))


def first_existing_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    candidates = [clean_col(c) for c in candidates]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def split_ids(value: Any) -> list[str]:
    """
    Split OSUKED ID cells like:
        "T_HOWAO-1, T_HOWAO-2, T_HOWAO-3"
    into a clean list.

    Returns [] for null, "-", blank.
    """
    if pd.isna(value):
        return []

    text = str(value).strip()
    if not text or text == "-" or text.lower() in {"nan", "none", "null"}:
        return []

    # OSUKED mostly uses comma-separated IDs.
    parts = re.split(r"\s*,\s*|\s*;\s*", text)

    out = []
    for p in parts:
        p = p.strip()
        if p and p != "-" and p.lower() not in {"nan", "none", "null"}:
            out.append(p)

    return out


def join_unique(values: pd.Series | list[Any]) -> Any:
    vals = []
    for v in values:
        if pd.isna(v):
            continue
        s = str(v).strip()
        if s and s != "-" and s.lower() not in {"nan", "none", "null"}:
            vals.append(s)

    vals = sorted(set(vals))
    return "; ".join(vals) if vals else pd.NA


def records_json(df: pd.DataFrame) -> str:
    df = df.where(pd.notna(df), None)
    return json.dumps(df.to_dict(orient="records"), ensure_ascii=False)


def discover_osuked_dataset_csvs(dataset_slug: str) -> list[str]:
    """
    Scrape the OSUKED dataset page and return CSV download links.

    Example:
        dataset_slug = "plant-locations"
        page = /datasets/plant-locations
    """
    page_url = f"{OSUKED_BASE}/datasets/{dataset_slug}"
    try:
        html_text = request_get(page_url).text
    except Exception as e:
        print(f"Could not open OSUKED dataset page {dataset_slug}: {e}")
        return []

    links = re.findall(r'href="([^"]+\.csv[^"]*)"', html_text, flags=re.IGNORECASE)
    links = [html.unescape(x) for x in links]
    links = [urljoin(page_url, x) for x in links]

    # Preserve order while removing duplicates.
    seen = set()
    out = []
    for link in links:
        if link not in seen:
            seen.add(link)
            out.append(link)

    return out


def read_first_available_csv(
    urls: list[str],
    required_any_cols: list[str] | None = None,
    label: str = "dataset",
) -> pd.DataFrame | None:
    """
    Try multiple CSV URLs and return the first one that works.

    If required_any_cols is given, require at least one of those columns.
    """
    required_any_cols_clean = (
        [clean_col(c) for c in required_any_cols]
        if required_any_cols
        else None
    )

    last_error = None

    for url in urls:
        try:
            df = read_csv_url(url)

            if required_any_cols_clean:
                if not any(c in df.columns for c in required_any_cols_clean):
                    continue

            print(f"Loaded {label}: {url}")
            return df

        except Exception as e:
            last_error = e

    print(f"Could not load {label}. Last error: {last_error}")
    return None


def read_all_available_csvs(
    urls: list[str],
    required_any_cols: list[str] | None = None,
    label: str = "dataset",
) -> pd.DataFrame | None:
    """
    Try all URLs and concatenate those that work.
    Useful where OSUKED has more than one resource under one dataset page.
    """
    frames = []
    required_any_cols_clean = (
        [clean_col(c) for c in required_any_cols]
        if required_any_cols
        else None
    )

    for url in urls:
        try:
            df = read_csv_url(url)

            if required_any_cols_clean:
                if not any(c in df.columns for c in required_any_cols_clean):
                    continue

            df["_source_url"] = url
            frames.append(df)
            print(f"Loaded {label}: {url}")

        except Exception as e:
            print(f"Skipped {label}: {url} because {e}")

    if not frames:
        return None

    return pd.concat(frames, ignore_index=True, sort=False)


# =============================================================================
# LOAD OSUKED BASE TABLE
# =============================================================================

def is_blank_value(x: Any) -> bool:
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s == "-" or s.lower() in {"nan", "none", "null"}


def choose_osuked_attr_value(row: pd.Series) -> Any:
    """
    OSUKED long table has both `id` and `value`.

    For identifier-like attributes, the useful thing is usually in `id`.
    For descriptive attributes, the useful thing is usually in `value`.

    This function chooses conservatively and falls back if one side is blank.
    """
    attr = clean_col(row.get("attribute", ""))
    id_type = clean_col(row.get("id_type", ""))

    raw_id = row.get("id", pd.NA)
    raw_value = row.get("value", pd.NA)

    name = f"{attr}_{id_type}"

    id_like_tokens = [
        "bmu",
        "eic",
        "cfd",
        "repd",
        "gppd",
        "esail",
        "wikidata",
        "wikipedia",
        "power_technology",
        "offshore",
        "wind_power",
        "jrc",
        "iaea",
        "crown_estate",
        "eutl",
        "id",
    ]

    looks_like_id = any(tok in name for tok in id_like_tokens)

    if looks_like_id:
        if not is_blank_value(raw_id):
            return str(raw_id).strip()
        if not is_blank_value(raw_value):
            return str(raw_value).strip()
    else:
        if not is_blank_value(raw_value):
            return str(raw_value).strip()
        if not is_blank_value(raw_id):
            return str(raw_id).strip()

    return pd.NA


def canonical_osuked_attribute_name(attribute: Any, id_type: Any = None) -> str:
    """
    Convert OSUKED long-table attributes into stable column names.

    Important:
    OSUKED object pages use:
        Related Settlement BMU ID
        Related National Grid BMU ID

    We force these into:
        settlement_bmu_id
        ngc_bmu_id
    """
    attr = clean_col(attribute)
    idt = clean_col(id_type) if id_type is not None else ""

    combined = f"{attr}_{idt}".strip("_")

    # Settlement BMU IDs, e.g. T_HOWAO-1
    if (
        "settlement_bmu" in combined
        or "settlement_bm_unit" in combined
        or "sett_bmu" in combined
        or "sett_bm_unit" in combined
        or "elexon_bmu" in combined
        or "elexon_bm_unit" in combined
    ):
        return "settlement_bmu_id"

    # National Grid / NGC BMU IDs, e.g. HOWAO-1
    if (
        "national_grid_bmu" in combined
        or "national_grid_bm_unit" in combined
        or "ngc_bmu" in combined
        or "ngc_bm_unit" in combined
    ):
        return "ngc_bmu_id"

    if "eic" in combined:
        return "eic_id"

    if "cfd" in combined:
        return "cfd_id"

    if "repd" in combined and "old" in combined:
        return "repd_id_old"

    if "repd" in combined and "new" in combined:
        return "repd_id_new"

    if "repd" in combined:
        return "repd_id"

    if "gppd" in combined:
        return "gppd_id"

    if "wikidata" in combined:
        return "wikidata_id"

    if "wikipedia" in combined:
        return "wikipedia_id"

    if "power_technology" in combined:
        return "power_technology_id"

    if "4c" in combined or "four_c" in combined:
        return "offshore_4c_id"

    if "wind_power" in combined or "windpower" in combined:
        return "wind_power_net_id"

    if "crown_estate" in combined:
        return "crown_estate_windfarm_id"

    if "esail" in combined:
        return "esail_id"

    if "jrc" in combined:
        return "jrc_id"

    if "iaea" in combined:
        return "iaea_id"

    if "common_name" in combined or combined == "name":
        return "common_name"

    return attr


def osuked_long_to_wide(raw: pd.DataFrame) -> pd.DataFrame:
    """
    Convert OSUKED object_attrs long table:

        dictionary_id | attribute | id | value | id_type | ...

    into:

        dictionary_id | common_name | settlement_bmu_id | national_grid_bmu_id | ...

    Multiple values for the same object/attribute are joined with '; '.
    """
    df = raw.copy()

    required = {"dictionary_id", "attribute"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(
            f"Cannot pivot OSUKED long table. Missing columns: {missing}. "
            f"Columns found: {list(df.columns)}"
        )

    if "id_type" not in df.columns:
        df["id_type"] = pd.NA

    if "id" not in df.columns:
        df["id"] = pd.NA

    if "value" not in df.columns:
        df["value"] = pd.NA

    df["dictionary_id"] = pd.to_numeric(df["dictionary_id"], errors="coerce")

    df["_attr_name"] = df.apply(
        lambda r: canonical_osuked_attribute_name(
            r.get("attribute", pd.NA),
            r.get("id_type", pd.NA),
        ),
        axis=1,
    )

    df["_attr_value"] = df.apply(choose_osuked_attr_value, axis=1)

    df = df[
        df["dictionary_id"].notna()
        & df["_attr_name"].notna()
        & df["_attr_value"].notna()
    ].copy()

    wide = (
        df.groupby(["dictionary_id", "_attr_name"], dropna=True)["_attr_value"]
        .apply(lambda s: join_unique(s))
        .unstack("_attr_name")
        .reset_index()
    )

    # Keep all raw OSUKED rows as JSON as well, so you do not lose anything.
    raw_json = (
        raw.assign(dictionary_id=pd.to_numeric(raw["dictionary_id"], errors="coerce"))
        .dropna(subset=["dictionary_id"])
        .groupby("dictionary_id", dropna=True)
        .apply(lambda g: records_json(g.drop(columns=["dictionary_id"], errors="ignore")))
        .reset_index(name="osuked_all_attributes_json")
    )

    wide = wide.merge(raw_json, on="dictionary_id", how="left")

    # Extra safety: if the canonical mapping missed a BMU column,
    # find any column that clearly looks like one.
    if "settlement_bmu_id" not in wide.columns:
        for c in wide.columns:
            cc = clean_col(c)
            if "settlement" in cc and "bmu" in cc:
                wide["settlement_bmu_id"] = wide[c]
                break

    if "national_grid_bmu_id" not in wide.columns:
        for c in wide.columns:
            cc = clean_col(c)
            if ("national_grid" in cc and "bmu" in cc) or ("ngc" in cc and "bmu" in cc):
                wide["national_grid_bmu_id"] = wide[c]
                break

    if "common_name" not in wide.columns:
        for c in wide.columns:
            cc = clean_col(c)
            if cc in {"name", "plant_name", "station_name", "power_station_name"}:
                wide["common_name"] = wide[c]
                break

    return wide

def load_osuked_objects() -> pd.DataFrame:
    """
    Load OSUKED dictionary object attributes.

    OSUKED can come as a long table:
        attribute | id | value | id_type | dictionary_id

    So this function converts it into one row per dictionary object.
    """
    raw = read_csv_url(OSUKED_OBJECT_ATTRS_URL)

    # Your error shows this exact long-table structure.
    is_long_table = {"attribute", "dictionary_id"}.issubset(raw.columns)

    if is_long_table:
        print("OSUKED object attributes are in long format. Pivoting to wide format...")
        objects = osuked_long_to_wide(raw)
    else:
        objects = raw.copy()

    objects = clean_columns(objects)

    if "dictionary_id" not in objects.columns:
        raise RuntimeError(
            f"OSUKED object attributes missing dictionary_id after loading. "
            f"Columns found: {list(objects.columns)}"
        )

    objects["dictionary_id"] = pd.to_numeric(objects["dictionary_id"], errors="coerce")

    bmu_cols = [c for c in objects.columns if "bmu" in c]
    print(f"BMU-related columns after OSUKED load: {bmu_cols}")

    return objects


def load_osuked_locations() -> pd.DataFrame:
    """
    Load OSUKED plant coordinates.
    """
    discovered = discover_osuked_dataset_csvs("plant-locations")

    fallback = [
        f"{OSUKED_BASE}/attribute_sources/plant-locations/plant-locations.csv",
    ]

    locs = read_first_available_csv(
        discovered + fallback,
        required_any_cols=["longitude", "latitude"],
        label="OSUKED plant locations",
    )

    if locs is None:
        raise RuntimeError("Could not load OSUKED plant locations.")

    required = {"dictionary_id", "longitude", "latitude"}
    missing = required - set(locs.columns)

    if missing:
        raise RuntimeError(
            f"OSUKED locations missing required columns: {missing}. "
            f"Columns found: {list(locs.columns)}"
        )

    locs["dictionary_id"] = pd.to_numeric(locs["dictionary_id"], errors="coerce")
    locs["longitude"] = pd.to_numeric(locs["longitude"], errors="coerce")
    locs["latitude"] = pd.to_numeric(locs["latitude"], errors="coerce")

    return locs


def make_bmu_rows(objects_with_locations: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Explode OSUKED plant/object rows into one row per BMU.

    Important:
    - The location comes from the physical plant / dictionary object.
    - If one plant has many BMUs, all those BMUs inherit the same plant coordinate.
    - Rows without coordinates are excluded from the main output and returned as skipped.
    """
    df = objects_with_locations.copy()

    # Flexible column names in case OSUKED changes slightly.
    dictionary_col = first_existing_col(df, ["dictionary_id"])
    common_name_col = first_existing_col(df, ["common_name"])
    ngc_col = first_existing_col(
        df,
        [
            "ngc_bmu_id",
            "national_grid_bmu_id",
            "national_grid_bm_unit",
            "national_grid_bm_unit_id",
            "ngc_bm_unit",
            "ngc_bm_unit_id",
        ],
    )

    settlement_col = first_existing_col(
        df,
        [
            "settlement_bmu_id",
            "sett_bmu_id",
            "settlement_bm_unit",
            "settlement_bm_unit_id",
            "sett_bm_unit",
            "sett_bm_unit_id",
            "elexon_bmu_id",
            "elexon_bm_unit",
        ],
    )

    print(f"Using NGC BMU column: {ngc_col}")
    print(f"Using Settlement BMU column: {settlement_col}")

    if dictionary_col is None:
        raise RuntimeError("No dictionary_id column found in OSUKED object table.")

    if ngc_col is None and settlement_col is None:
        raise RuntimeError(
            "No BMU ID columns found in OSUKED object table. "
            f"Columns found: {list(df.columns)}"
        )

    # Identify which dictionary objects have at least one BMU ID.
    has_bmu = pd.Series(False, index=df.index)

    if ngc_col:
        has_bmu = has_bmu | df[ngc_col].apply(lambda x: len(split_ids(x)) > 0)

    if settlement_col:
        has_bmu = has_bmu | df[settlement_col].apply(lambda x: len(split_ids(x)) > 0)

    has_location = (
        pd.to_numeric(df["longitude"], errors="coerce").notna()
        & pd.to_numeric(df["latitude"], errors="coerce").notna()
    )

    skipped = df[has_bmu & ~has_location].copy()
    df = df[has_bmu & has_location].copy()

    rows = []

    for _, row in df.iterrows():
        ngc_ids = split_ids(row[ngc_col]) if ngc_col else []
        settlement_ids = split_ids(row[settlement_col]) if settlement_col else []

        n = max(len(ngc_ids), len(settlement_ids), 1)

        for i in range(n):
            ngc_bmu_id = pd.NA
            settlement_bmu_id = pd.NA

            if len(ngc_ids) == n:
                ngc_bmu_id = ngc_ids[i]
            elif len(ngc_ids) == 1:
                ngc_bmu_id = ngc_ids[0]
            elif i < len(ngc_ids):
                ngc_bmu_id = ngc_ids[i]

            if len(settlement_ids) == n:
                settlement_bmu_id = settlement_ids[i]
            elif len(settlement_ids) == 1:
                settlement_bmu_id = settlement_ids[0]
            elif i < len(settlement_ids):
                settlement_bmu_id = settlement_ids[i]

            out = {
                "dictionary_id": row[dictionary_col],
                "common_name": row[common_name_col] if common_name_col else pd.NA,
                "ngc_bmu_id": ngc_bmu_id,
                "settlement_bmu_id": settlement_bmu_id,
                "longitude": row["longitude"],
                "latitude": row["latitude"],
            }

            # Keep every original OSUKED object attribute as a column too.
            # Prefix avoids clashing with the exploded single-BMU columns above.
            for c in df.columns:
                out[f"osuked_object_{c}"] = row[c]

            rows.append(out)

    bmus = pd.DataFrame(rows)

    # Guarantee: main output has no rows without location.
    bmus["longitude"] = pd.to_numeric(bmus["longitude"], errors="coerce")
    bmus["latitude"] = pd.to_numeric(bmus["latitude"], errors="coerce")

    bad = bmus[bmus["longitude"].isna() | bmus["latitude"].isna()]
    if len(bad):
        raise RuntimeError("Internal error: output BMU table contains missing locations.")

    return bmus, skipped


# =============================================================================
# LOAD OPTIONAL OSUKED SPECIFICATION DATASETS
# =============================================================================

def attach_bmu_dataset_json(
    bmus: pd.DataFrame,
    dataset_slug: str,
    output_json_col: str,
    bmu_key_candidates: list[str],
) -> pd.DataFrame:
    """
    Attach OSUKED BMU-level dataset as JSON.

    Fixed version:
    - Does NOT delete ngc_bmu_id from the main BMU table.
    - Uses temporary join columns to avoid pandas merge key collisions.
    """
    discovered = discover_osuked_dataset_csvs(dataset_slug)

    # Remove duplicate discovered URLs while preserving order.
    fallback = [
        f"{OSUKED_BASE}/attribute_sources/{dataset_slug}/{dataset_slug}.csv",
        f"{OSUKED_BASE}/attribute_sources/{dataset_slug}/{dataset_slug.replace('-', '_')}.csv",
    ]

    urls = []
    seen = set()
    for u in discovered + fallback:
        if u not in seen:
            seen.add(u)
            urls.append(u)

    src = read_all_available_csvs(
        urls,
        required_any_cols=bmu_key_candidates,
        label=f"OSUKED {dataset_slug}",
    )

    out = bmus.copy()

    if src is None:
        out[output_json_col] = pd.NA
        return out

    source_key_col = first_existing_col(src, bmu_key_candidates)
    if source_key_col is None:
        out[output_json_col] = pd.NA
        return out

    if "ngc_bmu_id" not in out.columns:
        raise RuntimeError(
            f"BMU table lost ngc_bmu_id before attaching {dataset_slug}. "
            f"Current columns include: {list(out.columns)[:80]}"
        )

    # Temporary source join key.
    src = src.copy()
    src["_source_bmu_join_key"] = src[source_key_col].astype("string").str.strip()

    src = src[
        src["_source_bmu_join_key"].notna()
        & (src["_source_bmu_join_key"].astype(str).str.strip() != "")
        & (src["_source_bmu_join_key"].astype(str).str.lower() != "nan")
    ].copy()

    # Create JSON records grouped by temporary source key.
    # Drop the source key from the JSON only, not from the BMU output table.
    grouped = (
        src.groupby("_source_bmu_join_key", dropna=True)
        .apply(
            lambda g: records_json(
                g.drop(columns=["_source_bmu_join_key"], errors="ignore")
            )
        )
        .reset_index(name=output_json_col)
    )

    # Temporary left join key.
    out["_left_bmu_join_key"] = out["ngc_bmu_id"].astype("string").str.strip()

    out = out.merge(
        grouped,
        left_on="_left_bmu_join_key",
        right_on="_source_bmu_join_key",
        how="left",
    )

    # Only remove temporary columns.
    out = out.drop(
        columns=["_left_bmu_join_key", "_source_bmu_join_key"],
        errors="ignore",
    )

    return out


def attach_fuel_type(bmus: pd.DataFrame) -> pd.DataFrame:
    """
    Attach simple fuel_type column plus full fuel-type source records as JSON.
    """
    discovered = discover_osuked_dataset_csvs("bmu-fuel-types")

    fallback = [
        f"{OSUKED_BASE}/attribute_sources/bmu-fuel-types/fuel_types.csv",
        f"{OSUKED_BASE}/attribute_sources/bmu-fuel-types/bmu-fuel-types.csv",
        f"{OSUKED_BASE}/attribute_sources/bmu-fuel-types/detailed-bmu-fuel-types.csv",
    ]

    urls = []
    seen = set()
    for u in discovered + fallback:
        if u not in seen:
            seen.add(u)
            urls.append(u)

    fuel = read_all_available_csvs(
        urls,
        required_any_cols=["ngc_bmu_id", "national_grid_bmu_id", "sett_bmu_id"],
        label="OSUKED BMU fuel types",
    )

    if fuel is None:
        bmus["fuel_type"] = pd.NA
        bmus["fuel_type_json"] = pd.NA
        return bmus

    ngc_key = first_existing_col(
        fuel,
        ["ngc_bmu_id", "national_grid_bmu_id", "national_grid_bm_unit"],
    )
    settlement_key = first_existing_col(
        fuel,
        ["sett_bmu_id", "settlement_bmu_id", "settlement_bm_unit"],
    )
    fuel_col = first_existing_col(fuel, ["fuel_type", "fuel type"])

    if fuel_col is None:
        bmus["fuel_type"] = pd.NA
    else:
        fuel[fuel_col] = fuel[fuel_col].astype("string").str.strip()

    frames = []

    if ngc_key is not None:
        tmp = fuel.copy()
        tmp["_join_bmu_id"] = tmp[ngc_key].astype("string").str.strip()
        frames.append(tmp)

    if settlement_key is not None:
        tmp = fuel.copy()
        tmp["_join_bmu_id"] = tmp[settlement_key].astype("string").str.strip()
        frames.append(tmp)

    if not frames:
        bmus["fuel_type"] = pd.NA
        bmus["fuel_type_json"] = pd.NA
        return bmus

    fuel_long = pd.concat(frames, ignore_index=True, sort=False)
    fuel_long = fuel_long[
        fuel_long["_join_bmu_id"].notna()
        & (fuel_long["_join_bmu_id"].astype(str).str.strip() != "")
    ]

    # Simple fuel type.
    if fuel_col:
        simple = (
            fuel_long.groupby("_join_bmu_id", dropna=True)[fuel_col]
            .apply(join_unique)
            .reset_index(name="fuel_type")
        )
    else:
        simple = pd.DataFrame(columns=["_join_bmu_id", "fuel_type"])

    # Full JSON source record.
    full = (
        fuel_long.groupby("_join_bmu_id", dropna=True)
        .apply(lambda g: records_json(g.drop(columns=["_join_bmu_id"], errors="ignore")))
        .reset_index(name="fuel_type_json")
    )

    fuel_grouped = simple.merge(full, on="_join_bmu_id", how="outer")

    out = bmus.copy()

    out = out.merge(
        fuel_grouped.add_prefix("ngc_"),
        left_on="ngc_bmu_id",
        right_on="ngc__join_bmu_id",
        how="left",
    )

    out = out.merge(
        fuel_grouped.add_prefix("settlement_"),
        left_on="settlement_bmu_id",
        right_on="settlement__join_bmu_id",
        how="left",
    )

    out["fuel_type"] = out["ngc_fuel_type"].combine_first(out["settlement_fuel_type"])
    out["fuel_type_json"] = out["ngc_fuel_type_json"].combine_first(
        out["settlement_fuel_type_json"]
    )

    out = out.drop(
        columns=[
            "ngc__join_bmu_id",
            "ngc_fuel_type",
            "ngc_fuel_type_json",
            "settlement__join_bmu_id",
            "settlement_fuel_type",
            "settlement_fuel_type_json",
        ],
        errors="ignore",
    )

    return out


def attach_plant_dataset_by_dictionary_id(
    bmus: pd.DataFrame,
    dataset_slug: str,
    output_json_col: str,
) -> pd.DataFrame:
    """
    Attach plant-level OSUKED dataset using dictionary_id.
    """
    discovered = discover_osuked_dataset_csvs(dataset_slug)

    fallback = [
        f"{OSUKED_BASE}/attribute_sources/{dataset_slug}/{dataset_slug}.csv",
        f"{OSUKED_BASE}/attribute_sources/{dataset_slug}/{dataset_slug.replace('-', '_')}.csv",
    ]

    src = read_all_available_csvs(
        discovered + fallback,
        required_any_cols=["dictionary_id"],
        label=f"OSUKED {dataset_slug}",
    )

    if src is None or "dictionary_id" not in src.columns:
        bmus[output_json_col] = pd.NA
        return bmus

    src["dictionary_id"] = pd.to_numeric(src["dictionary_id"], errors="coerce")

    grouped = (
        src.groupby("dictionary_id", dropna=True)
        .apply(lambda g: records_json(g.drop(columns=["dictionary_id"], errors="ignore")))
        .reset_index(name=output_json_col)
    )

    return bmus.merge(grouped, on="dictionary_id", how="left")


def attach_external_id_dataset_json(
    bmus: pd.DataFrame,
    dataset_slug: str,
    object_id_col: str,
    source_id_candidates: list[str],
    output_json_col: str,
) -> pd.DataFrame:
    """
    Attach source rows using an external ID list from the OSUKED dictionary.

    Example:
        object_id_col = "osuked_object_cfd_id"
        source_id_candidates = ["cfd_id"]
    """
    if object_id_col not in bmus.columns:
        bmus[output_json_col] = pd.NA
        return bmus

    discovered = discover_osuked_dataset_csvs(dataset_slug)

    fallback = [
        f"{OSUKED_BASE}/attribute_sources/{dataset_slug}/{dataset_slug}.csv",
        f"{OSUKED_BASE}/attribute_sources/{dataset_slug}/{dataset_slug.replace('-', '_')}.csv",
    ]

    src = read_all_available_csvs(
        discovered + fallback,
        required_any_cols=source_id_candidates,
        label=f"OSUKED {dataset_slug}",
    )

    if src is None:
        bmus[output_json_col] = pd.NA
        return bmus

    source_id_col = first_existing_col(src, source_id_candidates)
    if source_id_col is None:
        bmus[output_json_col] = pd.NA
        return bmus

    # Explode OSUKED object ID lists.
    bridge_rows = []
    for idx, value in bmus[object_id_col].items():
        for one_id in split_ids(value):
            bridge_rows.append({"_row_id": idx, "_external_id": one_id})

    if not bridge_rows:
        bmus[output_json_col] = pd.NA
        return bmus

    bridge = pd.DataFrame(bridge_rows)
    src["_external_id"] = src[source_id_col].astype("string").str.strip()

    joined = bridge.merge(src, on="_external_id", how="inner")

    if joined.empty:
        bmus[output_json_col] = pd.NA
        return bmus

    grouped = (
        joined.groupby("_row_id", dropna=True)
        .apply(lambda g: records_json(g.drop(columns=["_row_id"], errors="ignore")))
        .reset_index(name=output_json_col)
    )

    out = bmus.copy()
    out["_row_id"] = out.index

    out = out.merge(grouped, on="_row_id", how="left")
    out = out.drop(columns=["_row_id"], errors="ignore")

    return out


def add_capacity_guess(bmus: pd.DataFrame) -> pd.DataFrame:
    """
    Try to extract a usable capacity number from attached JSON sources.

    This is deliberately conservative. The full JSON sources are still preserved.
    """
    capacity_candidates = []

    possible_json_cols = [
        "gppd_json",
        "repd_new_json",
        "repd_old_json",
        "cfd_status_json",
        "wind_farms_json",
    ]

    capacity_name_patterns = [
        "capacity_mw",
        "installed_capacity_mw",
        "installed_capacity_mwelec",
        "installed_capacity_mwelec_",
        "installed_capacity",
        "cfd_capacity_mw",
        "max_contract_capacity",
        "max_contract_capacity_mw",
        "capacity",
    ]

    best_values = []
    best_sources = []

    for _, row in bmus.iterrows():
        best_val = pd.NA
        best_source = pd.NA

        for json_col in possible_json_cols:
            if json_col not in bmus.columns or pd.isna(row.get(json_col)):
                continue

            try:
                records = json.loads(row[json_col])
            except Exception:
                continue

            for rec in records:
                for k, v in rec.items():
                    ck = clean_col(k)
                    if ck not in capacity_name_patterns:
                        continue

                    num = pd.to_numeric(v, errors="coerce")
                    if pd.notna(num):
                        best_val = float(num)
                        best_source = f"{json_col}.{k}"
                        break

                if pd.notna(best_val):
                    break

            if pd.notna(best_val):
                break

        best_values.append(best_val)
        best_sources.append(best_source)

    bmus["capacity_mw_best_effort"] = best_values
    bmus["capacity_mw_source"] = best_sources

    return bmus


# =============================================================================
# LOAD NESO B5/B6 BOUNDARIES
# =============================================================================

def load_neso_boundaries_b5_b6() -> tuple[Any, Any]:
    """
    Load NESO boundary polylines from ArcGIS and return B5 and B6 geometries
    in British National Grid metres, EPSG:27700.
    """
    params = {
        "where": "1=1",
        "outFields": "*",
        "returnGeometry": "true",
        "outSR": "4326",
        "f": "geojson",
    }

    r = request_get(NESO_BOUNDARY_LAYER_QUERY_URL, params=params)
    gdf = gpd.read_file(io.BytesIO(r.content))

    if gdf.empty:
        raise RuntimeError("NESO boundary layer returned no geometry.")

    gdf = clean_columns(gdf)

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")

    gdf = gdf.to_crs("EPSG:27700")

    boundary_col = first_existing_col(gdf, ["boundary_n", "boundary"])

    if boundary_col is None:
        raise RuntimeError(
            f"Could not find Boundary_n field in NESO boundary layer. "
            f"Columns found: {list(gdf.columns)}"
        )

    gdf["_boundary_clean"] = gdf[boundary_col].astype(str).str.upper().str.strip()

    def get_boundary(label: str) -> Any:
        mask = gdf["_boundary_clean"].eq(label)

        if not mask.any():
            # Fallback in case labels are like "B5 something".
            mask = gdf["_boundary_clean"].str.contains(rf"\b{label}\b", regex=True)

        if not mask.any():
            raise RuntimeError(
                f"Could not find {label} in NESO boundary layer. "
                f"Available labels include: {sorted(gdf['_boundary_clean'].dropna().unique())[:50]}"
            )

        return unary_union(gdf.loc[mask, "geometry"].to_list())

    b5 = get_boundary("B5")
    b6 = get_boundary("B6")

    return b5, b6


# =============================================================================
# BOUNDARY CLASSIFICATION
# =============================================================================

def extract_y_values(geom: Any) -> list[float]:
    """
    Extract northing values from a Shapely intersection geometry.
    """
    if geom.is_empty:
        return []

    if geom.geom_type == "Point":
        return [float(geom.y)]

    if geom.geom_type == "MultiPoint":
        return [float(p.y) for p in geom.geoms]

    if geom.geom_type in {"LineString", "LinearRing"}:
        coords = list(geom.coords)
        if not coords:
            return []
        return [float(np.mean([y for _, y in coords]))]

    if geom.geom_type in {"MultiLineString", "GeometryCollection"}:
        ys = []
        for part in geom.geoms:
            ys.extend(extract_y_values(part))
        return ys

    return []


def boundary_y_at_same_easting(boundary_line: Any, x: float, y: float) -> tuple[float, str]:
    """
    Get the boundary northing at the same easting as the point.

    Method:
    - Draw a vertical north-south line through the BMU point.
    - Intersect it with the boundary polyline.
    - Use the closest intersection in northing.
    - If no intersection exists, fall back to nearest point on boundary.

    This works because B5 and B6 are broadly west-east boundaries.
    """
    minx, miny, maxx, maxy = boundary_line.bounds
    pad = 1_000_000

    vertical = LineString([(x, miny - pad), (x, maxy + pad)])
    inter = boundary_line.intersection(vertical)

    y_candidates = extract_y_values(inter)

    if y_candidates:
        chosen_y = min(y_candidates, key=lambda yy: abs(yy - y))
        return chosen_y, "vertical_intersection"

    nearest = nearest_points(Point(x, y), boundary_line)[1]
    return float(nearest.y), "nearest_boundary_fallback"


def classify_points_against_b5_b6(bmus: pd.DataFrame) -> pd.DataFrame:
    """
    Add B5/B6 zone classification to each BMU row.
    """
    b5_line, b6_line = load_neso_boundaries_b5_b6()

    gdf = gpd.GeoDataFrame(
        bmus.copy(),
        geometry=gpd.points_from_xy(bmus["longitude"], bmus["latitude"]),
        crs="EPSG:4326",
    ).to_crs("EPSG:27700")

    zones = []
    point_eastings = []
    point_northings = []
    b5_ys = []
    b6_ys = []
    b5_distances = []
    b6_distances = []
    methods = []
    warnings = []

    for _, row in gdf.iterrows():
        p: Point = row.geometry
        x = float(p.x)
        y = float(p.y)

        b5_y, b5_method = boundary_y_at_same_easting(b5_line, x, y)
        b6_y, b6_method = boundary_y_at_same_easting(b6_line, x, y)

        if y > b5_y:
            zone = ZONE_NORTH_OF_B5
        elif y < b6_y:
            zone = ZONE_SOUTH_OF_B6
        else:
            zone = ZONE_BETWEEN_B5_B6

        warning = ""
        if b5_y < b6_y:
            warning = (
                "B5 boundary northing is below B6 at this easting; "
                "check this point manually."
            )

        zones.append(zone)
        point_eastings.append(x)
        point_northings.append(y)
        b5_ys.append(b5_y)
        b6_ys.append(b6_y)
        b5_distances.append(float(p.distance(b5_line)))
        b6_distances.append(float(p.distance(b6_line)))
        methods.append(f"B5:{b5_method}; B6:{b6_method}")
        warnings.append(warning)

    out = bmus.copy()

    out["b5_b6_zone"] = zones
    out["point_easting_bng_m"] = point_eastings
    out["point_northing_bng_m"] = point_northings
    out["b5_boundary_northing_at_point_easting_m"] = b5_ys
    out["b6_boundary_northing_at_point_easting_m"] = b6_ys
    out["distance_to_b5_m"] = b5_distances
    out["distance_to_b6_m"] = b6_distances
    out["boundary_method"] = methods
    out["boundary_warning"] = warnings

    return out


# =============================================================================
# MAIN
# =============================================================================
def numeric_col_or_na(df: pd.DataFrame, col: str) -> pd.Series:
    if col in df.columns:
        return pd.to_numeric(df[col], errors="coerce")
    return pd.Series(pd.NA, index=df.index, dtype="float64")


def join_objects_with_locations(objects: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    """
    Robustly join OSUKED wide object table to plant-location table.

    Prevents pandas from creating longitude_x / longitude_y and latitude_x / latitude_y
    in a way that later breaks make_bmu_rows().
    """
    objects = objects.copy()
    locations = locations.copy()

    objects["dictionary_id"] = pd.to_numeric(objects["dictionary_id"], errors="coerce")
    locations["dictionary_id"] = pd.to_numeric(locations["dictionary_id"], errors="coerce")

    # Preserve any coordinate columns already present in the object table.
    if "longitude" in objects.columns:
        objects = objects.rename(columns={"longitude": "object_longitude"})
    if "latitude" in objects.columns:
        objects = objects.rename(columns={"latitude": "object_latitude"})

    # Rename location-table coordinates before merging.
    locations_small = locations[["dictionary_id", "longitude", "latitude"]].copy()
    locations_small = locations_small.rename(
        columns={
            "longitude": "location_longitude",
            "latitude": "location_latitude",
        }
    )

    merged = objects.merge(
        locations_small,
        on="dictionary_id",
        how="left",
    )

    # Prefer the official plant-location dataset; fall back to object-table coordinates.
    merged["longitude"] = numeric_col_or_na(merged, "location_longitude").combine_first(
        numeric_col_or_na(merged, "object_longitude")
    )

    merged["latitude"] = numeric_col_or_na(merged, "location_latitude").combine_first(
        numeric_col_or_na(merged, "object_latitude")
    )

    print("Coordinate columns after merge:")
    print([c for c in merged.columns if "long" in c.lower() or "lat" in c.lower()])
    print(f"Objects with longitude/latitude: {(merged['longitude'].notna() & merged['latitude'].notna()).sum()}")

    return merged


def make_clean_bmu_export(classified: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a clean modelling CSV with only useful columns.

    Keeps:
    - settlement BMU ID, e.g. T_HOWAO-1
    - NGC BMU ID, e.g. HOWAO-1
    - site name
    - fuel type
    - capacity
    - location
    - B5/B6 zone
    - key external IDs
    """
    df = classified.copy()

    # Normalise fuel type casing.
    if "fuel_type" in df.columns:
        df["fuel_type"] = df["fuel_type"].astype("string").str.upper().str.strip()

    clean_cols = [
        "b5_b6_zone",
        "dictionary_id",
        "common_name",
        "settlement_bmu_id",
        "ngc_bmu_id",
        "fuel_type",
        "capacity_mw_best_effort",
        "capacity_mw_source",
        "longitude",
        "latitude",
        "distance_to_b5_m",
        "distance_to_b6_m",
        "boundary_method",
        "boundary_warning",

        # Key OSUKED IDs
        "osuked_object_eic_id",
        "osuked_object_cfd_id",
        "osuked_object_repd_id_old",
        "osuked_object_repd_id_new",
        "osuked_object_gppd_id",
        "osuked_object_wikidata_id",
        "osuked_object_wikipedia_id",
        "osuked_object_power_technology_id",
        "osuked_object_offshore_4c_id",
        "osuked_object_wind_power_net_id",
        "osuked_object_crown_estate_windfarm_id",
    ]

    clean_cols = [c for c in clean_cols if c in df.columns]

    clean = df[clean_cols].copy()

    # Rename to nicer final names.
    clean = clean.rename(
        columns={
            "b5_b6_zone": "zone_b5_b6",
            "dictionary_id": "osuked_dictionary_id",
            "common_name": "plant_name",
            "capacity_mw_best_effort": "capacity_mw",
            "capacity_mw_source": "capacity_source",
            "osuked_object_eic_id": "eic_id",
            "osuked_object_cfd_id": "cfd_id",
            "osuked_object_repd_id_old": "repd_id_old",
            "osuked_object_repd_id_new": "repd_id_new",
            "osuked_object_gppd_id": "gppd_id",
            "osuked_object_wikidata_id": "wikidata_id",
            "osuked_object_wikipedia_id": "wikipedia_id",
            "osuked_object_power_technology_id": "power_technology_id",
            "osuked_object_offshore_4c_id": "offshore_4c_id",
            "osuked_object_wind_power_net_id": "wind_power_net_id",
            "osuked_object_crown_estate_windfarm_id": "crown_estate_windfarm_id",
        }
    )

    # Put settlement ID first because that is what Elexon/BMRS usually uses.
    final_order = [
        "settlement_bmu_id",
        "ngc_bmu_id",
        "plant_name",
        "fuel_type",
        "capacity_mw",
        "zone_b5_b6",
        "longitude",
        "latitude",
        "osuked_dictionary_id",
        "eic_id",
        "cfd_id",
        "repd_id_old",
        "repd_id_new",
        "gppd_id",
        "wikidata_id",
        "wikipedia_id",
        "power_technology_id",
        "offshore_4c_id",
        "wind_power_net_id",
        "crown_estate_windfarm_id",
        "distance_to_b5_m",
        "distance_to_b6_m",
        "boundary_method",
        "boundary_warning",
        "capacity_source",
    ]

    final_order = [c for c in final_order if c in clean.columns]
    clean = clean[final_order]

    return clean

def main() -> None:
    print("Loading OSUKED object attributes...")
    objects = load_osuked_objects()

    print("Loading OSUKED plant locations...")
    locations = load_osuked_locations()

    print("Joining OSUKED objects to locations...")
    objects_with_locations = join_objects_with_locations(objects, locations)

    print("Exploding OSUKED objects into one row per BMU...")
    bmus, skipped_no_location = make_bmu_rows(objects_with_locations)

    print(f"Classifiable BMU rows with location: {len(bmus)}")
    print(f"Skipped BMU-bearing OSUKED objects with no location: {len(skipped_no_location)}")

    # Export skipped before classification.
    skipped_path = OUTPUT_DIR / "skipped_osuked_objects_with_bmu_but_no_location.csv"
    skipped_no_location.to_csv(skipped_path, index=False)

    print("Attaching fuel type...")
    bmus = attach_fuel_type(bmus)

    print("Attaching BMU annual output history...")
    bmus = attach_bmu_dataset_json(
        bmus,
        dataset_slug="annual-output",
        output_json_col="annual_output_json",
        bmu_key_candidates=["ngc_bmu_id", "national_grid_bmu_id"],
    )

    print("Attaching BMU capture price history...")
    bmus = attach_bmu_dataset_json(
        bmus,
        dataset_slug="capture-prices",
        output_json_col="capture_prices_json",
        bmu_key_candidates=["ngc_bmu_id", "national_grid_bmu_id"],
    )

    print("Attaching BMU load factor history...")
    bmus = attach_bmu_dataset_json(
        bmus,
        dataset_slug="load-factors",
        output_json_col="load_factors_json",
        bmu_key_candidates=["ngc_bmu_id", "national_grid_bmu_id"],
    )

    print("Attaching plant-level wind farm data where available...")
    bmus = attach_plant_dataset_by_dictionary_id(
        bmus,
        dataset_slug="gb-wind-farm-database",
        output_json_col="wind_farms_json",
    )

    print("Attaching plant-level carbon intensity where available...")
    bmus = attach_plant_dataset_by_dictionary_id(
        bmus,
        dataset_slug="carbon-intensity",
        output_json_col="carbon_intensity_json",
    )

    print("Attaching Global Power Plant Database rows where available...")
    bmus = attach_external_id_dataset_json(
        bmus,
        dataset_slug="global-power-plant-database",
        object_id_col="osuked_object_gppd_id",
        source_id_candidates=["gppd_idnr", "gppd_id", "idnr"],
        output_json_col="gppd_json",
    )

    print("Attaching REPD new-ID rows where available...")
    bmus = attach_external_id_dataset_json(
        bmus,
        dataset_slug="renewable-energy-planning-database",
        object_id_col="osuked_object_repd_id_new",
        source_id_candidates=["ref_id", "repd_id", "id"],
        output_json_col="repd_new_json",
    )

    print("Attaching REPD old-ID rows where available...")
    bmus = attach_external_id_dataset_json(
        bmus,
        dataset_slug="renewable-energy-planning-database",
        object_id_col="osuked_object_repd_id_old",
        source_id_candidates=["old_ref_id", "old_repd_id", "old_ref"],
        output_json_col="repd_old_json",
    )

    print("Attaching CfD status rows where available...")
    bmus = attach_external_id_dataset_json(
        bmus,
        dataset_slug="cfd-contract-portfolio-status",
        object_id_col="osuked_object_cfd_id",
        source_id_candidates=["cfd_id"],
        output_json_col="cfd_status_json",
    )

    print("Attaching CfD strike-price rows where available...")
    bmus = attach_external_id_dataset_json(
        bmus,
        dataset_slug="cfd-strike-prices",
        object_id_col="osuked_object_cfd_id",
        source_id_candidates=["cfd_id"],
        output_json_col="cfd_strike_prices_json",
    )

    print("Adding best-effort capacity column...")
    bmus = add_capacity_guess(bmus)

    print("Loading NESO B5/B6 boundary geometry and classifying points...")
    classified = classify_points_against_b5_b6(bmus)

    # Put important columns first.
    first_cols = [
        "b5_b6_zone",
        "dictionary_id",
        "common_name",
        "ngc_bmu_id",
        "settlement_bmu_id",
        "fuel_type",
        "capacity_mw_best_effort",
        "capacity_mw_source",
        "longitude",
        "latitude",
        "point_easting_bng_m",
        "point_northing_bng_m",
        "distance_to_b5_m",
        "distance_to_b6_m",
        "boundary_method",
        "boundary_warning",
        "osuked_object_gppd_id",
        "osuked_object_esail_id",
        "osuked_object_eic_id",
        "osuked_object_cfd_id",
        "osuked_object_repd_id_old",
        "osuked_object_repd_id_new",
        "osuked_object_wikidata_id",
        "osuked_object_wikipedia_id",
        "osuked_object_power_technology_id",
        "osuked_object_4c_offshore_id",
        "osuked_object_wind_power_net_id",
        "fuel_type_json",
        "annual_output_json",
        "capture_prices_json",
        "load_factors_json",
        "gppd_json",
        "repd_new_json",
        "repd_old_json",
        "cfd_status_json",
        "cfd_strike_prices_json",
        "wind_farms_json",
        "carbon_intensity_json",
    ]

    first_cols = [c for c in first_cols if c in classified.columns]
    other_cols = [c for c in classified.columns if c not in first_cols]
    classified = classified[first_cols + other_cols]

    csv_path = OUTPUT_DIR / "osuked_bmus_classified_b5_b6_FULL.csv"
    jsonl_path = OUTPUT_DIR / "osuked_bmus_classified_b5_b6_FULL.jsonl"

    classified.to_csv(csv_path, index=False)
    classified.to_json(jsonl_path, orient="records", lines=True, force_ascii=False)

    clean = make_clean_bmu_export(classified)

    clean_csv_path = OUTPUT_DIR / "osuked_bmus_classified_b5_b6_CLEAN.csv"
    clean.to_csv(clean_csv_path, index=False)

    print("\nDONE")
    print(f"Full CSV:      {csv_path}")
    print(f"Full JSONL:    {jsonl_path}")
    print(f"Clean CSV:     {clean_csv_path}")
    print(f"Skipped CSV:   {skipped_path}")

    print("\nZone counts:")
    print(clean["zone_b5_b6"].value_counts(dropna=False))

    print("\nSettlement ID check:")
    print(clean[["settlement_bmu_id", "ngc_bmu_id", "plant_name", "zone_b5_b6"]].head(20))


if __name__ == "__main__":
    main()