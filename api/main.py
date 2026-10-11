from pathlib import Path
import sys
import pickle

import numpy as np
import pandas as pd
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from src.temporal_signal import calculate_temporal_consistency
from src.consensus_engine import calculate_consensus_risk

app = FastAPI(
    title="GARUDA AI API",
    description="AI-Driven Crime Pattern Analysis & Public Safety Intelligence System",
    version="1.5.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = BASE_DIR / "models" / "garuda_xgboost_e4.pkl"
CHICAGO_DATA_PATH = BASE_DIR / "data" / "chicago_crime_2025_2026.csv"
NCRB_DATA_PATH = BASE_DIR / "data" / "NCRB_Table_1C.1.csv"
UCI_DATA_PATH = BASE_DIR / "data" / "communities_crime.csv"
UCI_RAW_PATH = BASE_DIR / "data" / "communities.data"
UCI_CONTEXT_PATH = BASE_DIR / "results" / "uci_context_features.csv"

with open(MODEL_PATH, "rb") as f:
    GARUDA_ARTIFACT = pickle.load(f)

MODEL = GARUDA_ARTIFACT["model"]
ANOMALY_MODEL = GARUDA_ARTIFACT["anomaly_model"]
SPATIAL_MODEL = GARUDA_ARTIFACT["spatial_model"]
FEATURES = GARUDA_ARTIFACT["features"]

CHICAGO_DF = None
NCRB_DF = None
UCI_DF = None
UCI_CONTEXT_DF = None


def load_chicago_data():
    global CHICAGO_DF

    if CHICAGO_DF is not None:
        return CHICAGO_DF

    if not CHICAGO_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Chicago dataset not found: {CHICAGO_DATA_PATH}"
        )

    df = pd.read_csv(CHICAGO_DATA_PATH)

    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    for column in ["latitude", "longitude"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    CHICAGO_DF = df
    return CHICAGO_DF


def load_ncrb_data():
    global NCRB_DF

    if NCRB_DF is not None:
        return NCRB_DF

    if not NCRB_DATA_PATH.exists():
        raise FileNotFoundError(
            f"NCRB dataset not found: {NCRB_DATA_PATH}"
        )

    df = pd.read_csv(NCRB_DATA_PATH)

    for column in df.columns:
        if column not in ["State/UT", "Category"]:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce"
            )

    NCRB_DF = df
    return NCRB_DF


def load_uci_data():
    global UCI_DF

    if UCI_DF is not None:
        return UCI_DF

    if UCI_DATA_PATH.exists():
        df = pd.read_csv(UCI_DATA_PATH)
    elif UCI_RAW_PATH.exists():
        # Fallback for the original UCI Communities and Crime file.
        # The raw file is headerless; expose it as indexed attributes unless
        # a prepared communities_crime.csv is present.
        df = pd.read_csv(
            UCI_RAW_PATH,
            header=None,
            na_values="?"
        )
        df.columns = [f"feature_{i}" for i in range(df.shape[1])]
    else:
        raise FileNotFoundError(
            "UCI Communities & Crime dataset not found. Expected "
            f"{UCI_DATA_PATH} or {UCI_RAW_PATH}"
        )

    UCI_DF = df
    return UCI_DF


def load_uci_context_data():
    """Load the final UCI community-context feature artifact."""
    global UCI_CONTEXT_DF

    if UCI_CONTEXT_DF is not None:
        return UCI_CONTEXT_DF

    if not UCI_CONTEXT_PATH.exists():
        raise FileNotFoundError(
            f"UCI context artifact not found: {UCI_CONTEXT_PATH}"
        )

    df = pd.read_csv(UCI_CONTEXT_PATH)

    required_columns = [
        "feature_column", "feature_name_clean", "category",
        "missing_percent", "valid_n", "spearman_rho", "p_value",
        "adjusted_p_value", "fdr_significant", "context_role",
    ]

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            "UCI context artifact is missing required columns: "
            + ", ".join(missing)
        )

    UCI_CONTEXT_DF = df
    return UCI_CONTEXT_DF


class CrimeRequest(BaseModel):
    daily_crime_count: float = Field(..., ge=0)
    violent_crime_count: float = Field(..., ge=0)
    crime_count_7d: float = Field(..., ge=0)
    crime_count_30d: float = Field(..., ge=0)
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)


def clean_record(record):
    cleaned = {}

    for key, value in record.items():
        if pd.isna(value):
            cleaned[key] = None
        elif isinstance(value, pd.Timestamp):
            cleaned[key] = value.isoformat()
        elif isinstance(value, np.integer):
            cleaned[key] = int(value)
        elif isinstance(value, np.floating):
            cleaned[key] = float(value)
        else:
            cleaned[key] = value

    return cleaned


def calculate_spatial_signal(latitude, longitude):
    if SPATIAL_MODEL is None:
        return 0

    core_samples = getattr(
        SPATIAL_MODEL,
        "components_",
        None
    )

    if core_samples is None or len(core_samples) == 0:
        return 0

    point = np.array(
        [[latitude, longitude]],
        dtype=float
    )

    distances = np.sqrt(
        np.sum((core_samples - point) ** 2, axis=1)
    )

    return int(
        np.any(
            distances <= float(SPATIAL_MODEL.eps)
        )
    )


@app.get("/")
def root():
    return {
        "project": "GARUDA AI",
        "status": "running",
        "message": "GARUDA AI backend is operational"
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": type(MODEL).__name__,
        "anomaly_model": (
            type(ANOMALY_MODEL).__name__
            if ANOMALY_MODEL is not None else None
        ),
        "spatial_model": (
            type(SPATIAL_MODEL).__name__
            if SPATIAL_MODEL is not None else None
        ),
        "features": FEATURES,
        "data_sources": {
            "chicago_local_dataset":
                CHICAGO_DATA_PATH.exists(),
            "ncrb_local_dataset":
                NCRB_DATA_PATH.exists(),
            "uci_local_dataset":
                UCI_DATA_PATH.exists() or UCI_RAW_PATH.exists(),
            "uci_context_artifact":
                UCI_CONTEXT_PATH.exists()
        },
        "signals": {
            "ml_prediction": True,
            "temporal_consistency": True,
            "anomaly_detection":
                ANOMALY_MODEL is not None,
            "spatial_verification":
                SPATIAL_MODEL is not None,
            "consensus_layer": True
        }
    }


@app.get("/chicago/summary")
def chicago_summary():
    df = load_chicago_data()

    response = {
        "status": "success",
        "source": "Local Chicago crime dataset",
        "dataset": "chicago_crime_2025_2026.csv",
        "rows": int(len(df)),
        "columns": list(df.columns)
    }

    if "date" in df.columns:
        dates = df["date"].dropna()

        if len(dates):
            response["date_range"] = {
                "start": dates.min().isoformat(),
                "end": dates.max().isoformat()
            }

    if "beat" in df.columns:
        response["unique_beats"] = int(
            df["beat"].dropna().nunique()
        )

    return response


@app.get("/chicago/analysis")
def chicago_analysis():
    """Aggregate descriptive statistics from the complete loaded Chicago snapshot."""
    df = load_chicago_data()
    result = {"status": "success", "source": "Local Chicago crime dataset", "total_records": int(len(df))}

    if "primary_type" in df.columns:
        categories = (df["primary_type"].dropna().astype(str).str.strip()
                      .replace("", pd.NA).dropna().value_counts().head(12))
        result["crime_categories"] = [
            {"category": str(name), "count": int(count)}
            for name, count in categories.items()
        ]
    else:
        result["crime_categories"] = []

    if "beat" in df.columns:
        beats = df["beat"].dropna().astype(str).value_counts().head(10)
        result["top_beats"] = [
            {"beat": str(name), "count": int(count)}
            for name, count in beats.items()
        ]
    else:
        result["top_beats"] = []

    if "date" in df.columns:
        dates = pd.to_datetime(df["date"], errors="coerce").dropna()
        result["valid_dated_records"] = int(len(dates))
        result["date_range"] = {
            "start": dates.min().isoformat() if len(dates) else None,
            "end": dates.max().isoformat() if len(dates) else None,
        }
    else:
        result["valid_dated_records"] = 0
        result["date_range"] = {"start": None, "end": None}

    return result


@app.get("/chicago/crimes")
def chicago_crimes(
    limit: int = Query(500, ge=1, le=5000),
    crime_type: str | None = Query(None),
    beat: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None)
):
    df = load_chicago_data()
    filtered = df.copy()

    if crime_type and "primary_type" in filtered.columns:
        filtered = filtered[
            filtered["primary_type"]
            .astype(str)
            .str.upper()
            == crime_type.upper()
        ]

    if beat and "beat" in filtered.columns:
        filtered = filtered[
            filtered["beat"].astype(str) == str(beat)
        ]

    if start_date and "date" in filtered.columns:
        start = pd.to_datetime(
            start_date,
            errors="coerce"
        )

        if not pd.isna(start):
            filtered = filtered[
                filtered["date"] >= start
            ]

    if end_date and "date" in filtered.columns:
        end = pd.to_datetime(
            end_date,
            errors="coerce"
        )

        if not pd.isna(end):
            filtered = filtered[
                filtered["date"] <= end
            ]

    if "date" in filtered.columns:
        filtered = filtered.sort_values(
            "date",
            ascending=False
        )

    total = int(len(filtered))

    records = [
        clean_record(record)
        for record in filtered
        .head(limit)
        .to_dict(orient="records")
    ]

    return {
        "status": "success",
        "source": "Local Chicago crime dataset",
        "total_matching_records": total,
        "returned_records": len(records),
        "records": records
    }


@app.get("/chicago/hotspots")
def chicago_hotspots(
    limit: int = Query(100, ge=1, le=500),
    crime_type: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None)
):
    df = load_chicago_data()

    required = [
        "beat",
        "latitude",
        "longitude"
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        return {
            "status": "error",
            "message": "Required spatial columns are missing",
            "missing_columns": missing
        }

    filtered = df.copy()

    if crime_type and "primary_type" in filtered.columns:
        filtered = filtered[
            filtered["primary_type"]
            .astype(str)
            .str.upper()
            == crime_type.upper()
        ]

    if start_date and "date" in filtered.columns:
        start = pd.to_datetime(
            start_date,
            errors="coerce"
        )

        if not pd.isna(start):
            filtered = filtered[
                filtered["date"] >= start
            ]

    if end_date and "date" in filtered.columns:
        end = pd.to_datetime(
            end_date,
            errors="coerce"
        )

        if not pd.isna(end):
            filtered = filtered[
                filtered["date"] <= end
            ]

    filtered = filtered.dropna(
        subset=required
    )

    grouped = (
        filtered
        .groupby("beat")
        .agg(
            crime_count=("beat", "size"),
            latitude=("latitude", "median"),
            longitude=("longitude", "median")
        )
        .reset_index()
        .sort_values(
            "crime_count",
            ascending=False
        )
    )

    hotspots = [
        {
            "beat": str(row["beat"]),
            "crime_count": int(row["crime_count"]),
            "latitude": round(
                float(row["latitude"]),
                6
            ),
            "longitude": round(
                float(row["longitude"]),
                6
            )
        }
        for _, row in grouped.head(limit).iterrows()
    ]

    return {
        "status": "success",
        "source": "Observed Chicago crime records",
        "method": "Beat-level spatial aggregation",
        "total_hotspots": int(len(grouped)),
        "returned_hotspots": len(hotspots),
        "hotspots": hotspots
    }


@app.get("/chicago/temporal")
def chicago_temporal(
    granularity: str = Query("daily"),
    crime_type: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None)
):
    if granularity not in {
        "daily",
        "weekly",
        "monthly"
    }:
        return {
            "status": "error",
            "message":
                "granularity must be daily, weekly, or monthly"
        }

    df = load_chicago_data()

    if "date" not in df.columns:
        return {
            "status": "error",
            "message": "date column is missing"
        }

    filtered = df.dropna(
        subset=["date"]
    ).copy()

    if crime_type and "primary_type" in filtered.columns:
        filtered = filtered[
            filtered["primary_type"]
            .astype(str)
            .str.upper()
            == crime_type.upper()
        ]

    if start_date:
        start = pd.to_datetime(
            start_date,
            errors="coerce"
        )

        if not pd.isna(start):
            filtered = filtered[
                filtered["date"] >= start
            ]

    if end_date:
        end = pd.to_datetime(
            end_date,
            errors="coerce"
        )

        if not pd.isna(end):
            filtered = filtered[
                filtered["date"] <= end
            ]

    if granularity == "daily":
        filtered["period"] = (
            filtered["date"]
            .dt.strftime("%Y-%m-%d")
        )

    elif granularity == "weekly":
        filtered["period"] = (
            filtered["date"]
            .dt.to_period("W")
            .apply(
                lambda x:
                x.start_time.strftime("%Y-%m-%d")
            )
        )

    else:
        filtered["period"] = (
            filtered["date"]
            .dt.strftime("%Y-%m")
        )

    grouped = (
        filtered
        .groupby("period")
        .size()
        .reset_index(name="crime_count")
        .sort_values("period")
    )

    counts = grouped["crime_count"].astype(float)

    if len(counts) >= 2:
        mean_value = float(counts.mean())
        std_value = float(
            counts.std(ddof=0)
        )

        if std_value > 0:
            grouped["normalized_activity"] = (
                (counts - mean_value)
                / std_value
            )
        else:
            grouped["normalized_activity"] = 0.0

    else:
        grouped["normalized_activity"] = 0.0

    records = [
        {
            "period": str(row["period"]),
            "crime_count": int(row["crime_count"]),
            "normalized_activity": round(
                float(row["normalized_activity"]),
                6
            )
        }
        for _, row in grouped.iterrows()
    ]

    return {
        "status": "success",
        "source": "Observed Chicago crime records",
        "granularity": granularity,
        "total_periods": len(records),
        "records": records
    }


@app.get("/ncrb/summary")
def ncrb_summary():
    df = load_ncrb_data()

    response = {
        "status": "success",
        "source": (
            "Local NCRB Crime in India dataset"
        ),
        "dataset": "NCRB_Table_1C.1.csv",
        "rows": int(len(df)),
        "columns": list(df.columns),
        "granularity": "Annual",
        "geographic_level": "State/UT"
    }

    if "State/UT" in df.columns:
        response["states_or_ut_count"] = int(
            df["State/UT"]
            .dropna()
            .nunique()
        )

    if "Category" in df.columns:
        response["categories"] = [
            str(value)
            for value in df["Category"]
            .dropna()
            .unique()
        ]

    return response


@app.get("/ncrb/states")
def ncrb_states(
    limit: int = Query(50, ge=1, le=100),
    category: str | None = Query(None)
):
    df = load_ncrb_data()

    if "State/UT" not in df.columns:
        return {
            "status": "error",
            "message": "State/UT column is missing"
        }

    filtered = df.copy()

    if category and "Category" in filtered.columns:
        filtered = filtered[
            filtered["Category"]
            .astype(str)
            .str.upper()
            == category.upper()
        ]

    records = [
        clean_record(record)
        for record in filtered
        .head(limit)
        .to_dict(orient="records")
    ]

    return {
        "status": "success",
        "source": (
            "Local NCRB Crime in India dataset"
        ),
        "granularity": "Annual",
        "records": records
    }


@app.get("/ncrb/trend")
def ncrb_trend(
    state: str | None = Query(None),
    category: str | None = Query(None)
):
    df = load_ncrb_data()

    required_years = [
        "2020",
        "2021",
        "2022"
    ]

    missing = [
        year
        for year in required_years
        if year not in df.columns
    ]

    if missing:
        return {
            "status": "error",
            "message": "Required year columns are missing",
            "missing_columns": missing
        }

    filtered = df.copy()

    if state and "State/UT" in filtered.columns:
        filtered = filtered[
            filtered["State/UT"]
            .astype(str)
            .str.upper()
            == state.upper()
        ]

    if category and "Category" in filtered.columns:
        filtered = filtered[
            filtered["Category"]
            .astype(str)
            .str.upper()
            == category.upper()
        ]

    records = []

    for year in required_years:
        value = pd.to_numeric(
            filtered[year],
            errors="coerce"
        ).sum()

        records.append({
            "year": int(year),
            "value": (
                None
                if pd.isna(value)
                else float(value)
            )
        })

    return {
        "status": "success",
        "source": (
            "Local NCRB Crime in India dataset"
        ),
        "state": state,
        "category": category,
        "records": records
    }


@app.get("/uci/summary")
def uci_summary():
    df = load_uci_data()

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    return {
        "status": "success",
        "source": "Local UCI Communities & Crime dataset",
        "dataset": (
            UCI_DATA_PATH.name
            if UCI_DATA_PATH.exists()
            else UCI_RAW_PATH.name
        ),
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "numeric_features": len(numeric_columns),
        "missing_values": int(df.isna().sum().sum()),
        "granularity": "Community-level",
        "purpose": (
            "Community and socioeconomic crime-related attributes "
            "for descriptive and comparative analysis"
        )
    }


@app.get("/uci/records")
def uci_records(
    limit: int = Query(100, ge=1, le=500),
    offset: int = Query(0, ge=0)
):
    df = load_uci_data()

    total = int(len(df))
    records = [
        clean_record(record)
        for record in df.iloc[offset:offset + limit]
        .to_dict(orient="records")
    ]

    return {
        "status": "success",
        "source": "Local UCI Communities & Crime dataset",
        "total_records": total,
        "offset": offset,
        "returned_records": len(records),
        "records": records
    }


@app.get("/uci/features")
def uci_features():
    df = load_uci_data()

    features = []

    for column in df.columns:
        series = df[column]
        numeric = pd.to_numeric(series, errors="coerce")

        features.append({
            "feature": str(column),
            "dtype": str(series.dtype),
            "numeric": bool(numeric.notna().sum() > 0),
            "missing": int(series.isna().sum()),
            "unique_values": int(series.nunique(dropna=True))
        })

    return {
        "status": "success",
        "source": "Local UCI Communities & Crime dataset",
        "feature_count": len(features),
        "features": features
    }


@app.get("/uci/context")
def uci_context(
    category: str | None = Query(None),
    limit: int = Query(20, ge=1, le=20),
):
    """Return selected UCI community-context evidence."""
    df = load_uci_context_data().copy()

    if category:
        df = df[df["category"].astype(str).str.upper() == category.upper()]

    records = [
        clean_record(record)
        for record in df.head(limit).to_dict(orient="records")
    ]

    return {
        "status": "success",
        "source": "UCI Communities & Crime",
        "dataset": "communities_crime.csv",
        "granularity": "Community-level",
        "target": "ViolentCrimesPerPop",
        "interpretation": (
            "Historical descriptive associations between selected "
            "community-level attributes and violent-crime rate. "
            "These associations are not causal and are not used "
            "as direct Chicago crime predictions."
        ),
        "total_context_features": int(len(df)),
        "returned_features": len(records),
        "records": records,
    }


@app.get("/intelligence/profile")
def intelligence_profile(
    state: str | None = Query(None),
    ncrb_category: str | None = Query(None),
    chicago_hotspot_limit: int = Query(10, ge=1, le=25),
):
    """Cross-dataset GARUDA intelligence profile."""
    chicago = load_chicago_data()
    ncrb = load_ncrb_data()
    uci_context_df = load_uci_context_data()

    chicago_dates = (
        chicago["date"].dropna()
        if "date" in chicago.columns
        else pd.Series(dtype="datetime64[ns]")
    )

    chicago_summary = {
        "source": "Local Chicago crime dataset",
        "dataset": CHICAGO_DATA_PATH.name,
        "granularity": "Incident-level",
        "records": int(len(chicago)),
    }

    if len(chicago_dates):
        chicago_summary["date_range"] = {
            "start": chicago_dates.min().isoformat(),
            "end": chicago_dates.max().isoformat(),
        }

    if "beat" in chicago.columns:
        chicago_summary["unique_beats"] = int(
            chicago["beat"].dropna().nunique()
        )

    if {"beat", "latitude", "longitude"}.issubset(chicago.columns):
        spatial_df = chicago.dropna(
            subset=["beat", "latitude", "longitude"]
        )
        grouped = (
            spatial_df.groupby("beat")
            .agg(
                crime_count=("beat", "size"),
                latitude=("latitude", "median"),
                longitude=("longitude", "median"),
            )
            .reset_index()
            .sort_values("crime_count", ascending=False)
            .head(chicago_hotspot_limit)
        )
        chicago_hotspots = [
            {
                "beat": str(row["beat"]),
                "crime_count": int(row["crime_count"]),
                "latitude": round(float(row["latitude"]), 6),
                "longitude": round(float(row["longitude"]), 6),
            }
            for _, row in grouped.iterrows()
        ]
    else:
        chicago_hotspots = []

    filtered_ncrb = ncrb.copy()
    if state and "State/UT" in filtered_ncrb.columns:
        filtered_ncrb = filtered_ncrb[
            filtered_ncrb["State/UT"].astype(str).str.upper() == state.upper()
        ]
    if ncrb_category and "Category" in filtered_ncrb.columns:
        filtered_ncrb = filtered_ncrb[
            filtered_ncrb["Category"].astype(str).str.upper()
            == ncrb_category.upper()
        ]

    ncrb_records = []
    for year in ["2020", "2021", "2022"]:
        if year in filtered_ncrb.columns:
            value = pd.to_numeric(filtered_ncrb[year], errors="coerce").sum()
            ncrb_records.append({
                "year": int(year),
                "value": None if pd.isna(value) else float(value),
            })

    uci_records = [
        clean_record(record)
        for record in uci_context_df.to_dict(orient="records")
    ]

    return {
        "status": "success",
        "profile": {
            "name": "GARUDA Cross-Dataset Intelligence Profile",
            "purpose": (
                "Structured contextual evidence from Chicago crime, "
                "NCRB annual State/UT statistics, and UCI community-level "
                "crime-related attributes."
            ),
            "integration_rule": (
                "Datasets are kept separate by source, geographic "
                "granularity and time period; they are not concatenated "
                "into a single prediction dataset."
            ),
        },
        "chicago": {
            "role": "Operational crime intelligence",
            "summary": chicago_summary,
            "top_observed_hotspots": chicago_hotspots,
        },
        "ncrb": {
            "role": "Annual Indian State/UT crime context",
            "state_filter": state,
            "category_filter": ncrb_category,
            "records": ncrb_records,
        },
        "uci": {
            "role": "Historical community-level context",
            "target": "ViolentCrimesPerPop",
            "records": uci_records,
        },
    }


@app.post("/predict")
def predict_crime_risk(
    request: CrimeRequest
):
    temporal_consistency = (
        calculate_temporal_consistency(
            request.daily_crime_count,
            request.crime_count_7d,
            request.crime_count_30d
        )
    )

    anomaly_features = pd.DataFrame(
        [[
            request.daily_crime_count,
            request.violent_crime_count,
            request.crime_count_7d,
            request.crime_count_30d
        ]],
        columns=[
            "daily_crime_count",
            "violent_crime_count",
            "crime_count_7d",
            "crime_count_30d"
        ]
    )

    anomaly_score = 0.0
    anomaly_signal = 0

    if ANOMALY_MODEL is not None:
        anomaly_score = float(
            -ANOMALY_MODEL
            .score_samples(
                anomaly_features
            )[0]
        )

        anomaly_signal = int(
            ANOMALY_MODEL
            .predict(
                anomaly_features
            )[0]
            == -1
        )

    spatial_signal = calculate_spatial_signal(
        request.latitude,
        request.longitude
    )

    input_data = pd.DataFrame(
        [[
            request.daily_crime_count,
            request.violent_crime_count,
            request.crime_count_7d,
            request.crime_count_30d,
            temporal_consistency,
            anomaly_score,
            spatial_signal
        ]],
        columns=[
            "daily_crime_count",
            "violent_crime_count",
            "crime_count_7d",
            "crime_count_30d",
            "temporal_consistency",
            "anomaly_score",
            "spatial_signal"
        ]
    )

    ml_probability = float(
        MODEL.predict_proba(
            input_data
        )[0][1]
    )

    consensus_result = (
        calculate_consensus_risk(
            ml_probability=ml_probability,
            temporal_consistency=temporal_consistency,
            anomaly_signal=anomaly_signal,
            spatial_signal=spatial_signal
        )
    )

    return {
        "status": "prediction_successful",
        "signals": {
            "ml_probability":
                round(ml_probability, 6),
            "temporal_consistency":
                round(
                    temporal_consistency,
                    6
                ),
            "anomaly_score":
                round(
                    anomaly_score,
                    6
                ),
            "anomaly_signal":
                anomaly_signal,
            "spatial_signal":
                spatial_signal
        },
        "consensus": {
            "consensus_score":
                round(
                    consensus_result[
                        "consensus_score"
                    ],
                    6
                ),
            "risk_level":
                consensus_result["risk_level"]
        },
        "location": {
            "latitude":
                request.latitude,
            "longitude":
                request.longitude
        },
        "model": "XGBoost E4",
        "verification": {
            "ml": True,
            "temporal": True,
            "anomaly":
                ANOMALY_MODEL is not None,
            "spatial":
                SPATIAL_MODEL is not None
        }
    }
