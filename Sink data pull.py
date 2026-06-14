import requests
import pandas as pd
from pathlib import Path


# ============================================================
# SETTINGS
# ============================================================

OUT_DIR = Path("sink prediction")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# NESO Day Ahead Constraint Flows and Limits resource ID
DAY_AHEAD_CONSTRAINT_RESOURCE_ID = "38a18ec1-9e40-465d-93fb-301e80fd1352"

# B6 / Scotland-England boundary in this NESO dataset
BOUNDARY_CODE = "SCOTEX"

START_DATE = "2023-01-01"
END_DATE = "2025-12-31"


# ============================================================
# API PULL
# ============================================================

def fetch_b6_day_ahead_flow_limit(start_date, end_date):
    """
    Pull SCOTEX/B6 day-ahead flow and limit from NESO CKAN API.
    """

    sql = f"""
        SELECT *
        FROM "{DAY_AHEAD_CONSTRAINT_RESOURCE_ID}"
        WHERE "Constraint Group" = '{BOUNDARY_CODE}'
        AND "Date (GMT/BST)" >= '{start_date}'
        AND "Date (GMT/BST)" <= '{end_date} 23:59:59'
    """

    url = "https://api.neso.energy/api/3/action/datastore_search_sql"

    response = requests.get(url, params={"sql": sql}, timeout=180)
    response.raise_for_status()

    payload = response.json()

    if not payload.get("success", False):
        raise RuntimeError(payload)

    records = payload["result"]["records"]

    return pd.DataFrame(records)


# ============================================================
# CLEANING + SINK PREDICTION
# ============================================================

def clean_b6_sink_prediction(raw):
    """
    Convert raw NESO SCOTEX data into model-ready sink-prediction data.
    """

    df = raw.copy()

    print("Raw columns:")
    print(df.columns.tolist())

    df = df.rename(
        columns={
            "Date (GMT/BST)": "timestamp",
            "Constraint Group": "constraint_group",
            "Flow (MW)": "b6_day_ahead_flow_mw",
            "Limit (MW)": "b6_day_ahead_limit_mw",
        }
    )

    required = [
        "timestamp",
        "constraint_group",
        "b6_day_ahead_flow_mw",
        "b6_day_ahead_limit_mw",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing expected columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    df["b6_day_ahead_flow_mw"] = pd.to_numeric(
        df["b6_day_ahead_flow_mw"],
        errors="coerce"
    )

    df["b6_day_ahead_limit_mw"] = pd.to_numeric(
        df["b6_day_ahead_limit_mw"],
        errors="coerce"
    )

    df = df.dropna(
        subset=[
            "timestamp",
            "b6_day_ahead_flow_mw",
            "b6_day_ahead_limit_mw",
        ]
    )

    # Core sink prediction:
    # extra electricity that can still be sent south through B6
    df["b6_headroom_mw"] = (
        df["b6_day_ahead_limit_mw"]
        - df["b6_day_ahead_flow_mw"]
    )

    df["predicted_southward_sink_mw"] = (
        df["b6_headroom_mw"].clip(lower=0)
    )

    # Useful flags
    df["b6_is_over_forecast_limit"] = df["b6_headroom_mw"] < 0

    df["b6_utilisation_pct"] = (
        df["b6_day_ahead_flow_mw"]
        / df["b6_day_ahead_limit_mw"]
        * 100
    )

    df["settlement_date"] = df["timestamp"].dt.date

    # Approximate settlement period from timestamp
    df["settlement_period"] = (
        df["timestamp"].dt.hour * 2
        + (df["timestamp"].dt.minute // 30)
        + 1
    )

    keep_cols = [
        "timestamp",
        "settlement_date",
        "settlement_period",
        "constraint_group",
        "b6_day_ahead_flow_mw",
        "b6_day_ahead_limit_mw",
        "b6_headroom_mw",
        "predicted_southward_sink_mw",
        "b6_utilisation_pct",
        "b6_is_over_forecast_limit",
    ]

    df = (
        df[keep_cols]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )

    return df


# ============================================================
# SIMPLE DAY-AHEAD SCENARIOS
# ============================================================

def create_simple_sink_scenarios(base_df):
    """
    Creates simple low/base/high sink scenarios from B6 headroom.

    Interpretation:
    - pessimistic: less export room than expected
    - base: raw NESO day-ahead headroom
    - optimistic: more export room than expected
    """

    df = base_df.copy()

    scenarios = []

    for scenario_name, multiplier in [
        ("pessimistic_low_sink", 0.75),
        ("base_sink", 1.00),
        ("optimistic_high_sink", 1.25),
    ]:
        temp = df.copy()
        temp["scenario"] = scenario_name
        temp["scenario_probability"] = {
            "pessimistic_low_sink": 0.25,
            "base_sink": 0.50,
            "optimistic_high_sink": 0.25,
        }[scenario_name]

        temp["scenario_predicted_southward_sink_mw"] = (
            temp["predicted_southward_sink_mw"] * multiplier
        )

        # Cannot exceed the physical day-ahead B6 limit
        temp["scenario_predicted_southward_sink_mw"] = temp[
            "scenario_predicted_southward_sink_mw"
        ].clip(
            lower=0,
            upper=temp["b6_day_ahead_limit_mw"]
        )

        scenarios.append(temp)

    return pd.concat(scenarios, ignore_index=True)


# ============================================================
# RUN
# ============================================================

raw_b6 = fetch_b6_day_ahead_flow_limit(
    start_date=START_DATE,
    end_date=END_DATE,
)

raw_b6.to_csv(
    OUT_DIR / "raw_neso_b6_scotex_day_ahead_flow_limit.csv",
    index=False,
)

sink_base = clean_b6_sink_prediction(raw_b6)

sink_base.to_csv(
    OUT_DIR / "b6_sink_prediction_base_half_hourly.csv",
    index=False,
)

sink_scenarios = create_simple_sink_scenarios(sink_base)

sink_scenarios.to_csv(
    OUT_DIR / "b6_sink_prediction_scenarios_half_hourly.csv",
    index=False,
)

print("Saved:")
print(OUT_DIR / "raw_neso_b6_scotex_day_ahead_flow_limit.csv")
print(OUT_DIR / "b6_sink_prediction_base_half_hourly.csv")
print(OUT_DIR / "b6_sink_prediction_scenarios_half_hourly.csv")

print("\nBase sink prediction sample:")
print(sink_base.head())

print("\nScenario sample:")
print(sink_scenarios.head())