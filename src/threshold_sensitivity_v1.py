"""
GARUDA AI — Baseline V1 Threshold Sensitivity Experiment
=========================================================
Uses:
- strict day-level temporal holdout
- spatial eps = 0.02
- spatial min_samples = 5
- ML/Temporal/Anomaly/Spatial weights = 0.40/0.20/0.20/0.20

Outputs:
- results/garuda_v1_threshold_sensitivity.csv
- results/garuda_v1_threshold_sensitivity.png
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    cohen_kappa_score,
    confusion_matrix,
)

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if str(DASHBOARD_DIR) not in sys.path:
    sys.path.insert(0, str(DASHBOARD_DIR))

from model_comparison_8_models import (
    load_and_engineer_data,
    prepare_spatial_coordinates,
    add_anomaly_signal,
    add_spatial_signal,
    create_xgboost_model,
)

RANDOM_STATE = 42
SPATIAL_EPS = 0.020
SPATIAL_MIN_SAMPLES = 5
ANOMALY_CONTAMINATION = 0.20

BASE_FEATURES = [
    "daily_crime_count",
    "violent_crime_count",
    "crime_count_7d",
    "crime_count_30d",
]

# Thresholds include the frozen 0.66 value and a useful range around it.
THRESHOLDS = [
    0.30, 0.35, 0.40, 0.45, 0.50,
    0.55, 0.60, 0.65, 0.66, 0.70
]


def strict_day_split(data):
    data = data.sort_values(["crime_date", "beat"]).reset_index(drop=True)

    unique_dates = sorted(data["crime_date"].dt.normalize().unique())
    split_idx = int(len(unique_dates) * 0.80)

    # Prevent a date from appearing in both train and test.
    split_idx = max(1, min(split_idx, len(unique_dates) - 1))
    cutoff_date = pd.Timestamp(unique_dates[split_idx])

    train = data[data["crime_date"] < cutoff_date].copy()
    test = data[data["crime_date"] >= cutoff_date].copy()

    if set(train["crime_date"].dt.normalize()).intersection(
        set(test["crime_date"].dt.normalize())
    ):
        raise RuntimeError("Temporal leakage: train/test dates overlap.")

    return train, test, cutoff_date


def prepare_common_cohort_without_row_split(df):
    required = [
        "crime_date", "beat",
        "daily_crime_count", "violent_crime_count",
        "crime_count_7d", "crime_count_30d",
        "temporal_consistency", "next_day_crime_count", "high_risk",
        "avg_latitude", "avg_longitude",
    ]

    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    data = df[required].copy()

    numeric = [
        "daily_crime_count", "violent_crime_count",
        "crime_count_7d", "crime_count_30d",
        "temporal_consistency", "next_day_crime_count",
        "high_risk", "avg_latitude", "avg_longitude",
    ]

    for c in numeric:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    data["crime_date"] = pd.to_datetime(data["crime_date"], errors="coerce")

    before = len(data)
    data = data.dropna(
        subset=[
            "crime_date",
            "daily_crime_count",
            "violent_crime_count",
            "crime_count_7d",
            "crime_count_30d",
            "temporal_consistency",
            "next_day_crime_count",
            "high_risk",
        ]
    ).copy()

    data["high_risk"] = data["high_risk"].astype(int)

    return data.sort_values(["crime_date", "beat"]).reset_index(drop=True), before


def main():
    print("=" * 90)
    print("GARUDA AI — BASELINE V1 THRESHOLD SENSITIVITY")
    print("=" * 90)

    df = load_and_engineer_data()
    data, before = prepare_common_cohort_without_row_split(df)

    print(f"\nRows before filtering : {before:,}")
    print(f"Rows after filtering  : {len(data):,}")

    train_data, test_data, cutoff = strict_day_split(data)

    print("\nSTRICT DAY-LEVEL HOLDOUT")
    print(f"Cutoff date           : {cutoff.date()}")
    print(f"Training rows          : {len(train_data):,}")
    print(f"Testing rows           : {len(test_data):,}")
    print(
        f"Training period       : "
        f"{train_data['crime_date'].min().date()} → "
        f"{train_data['crime_date'].max().date()}"
    )
    print(
        f"Testing period        : "
        f"{test_data['crime_date'].min().date()} → "
        f"{test_data['crime_date'].max().date()}"
    )

    # 1. Anomaly component
    train_g, test_g, _ = add_anomaly_signal(
        train_data.copy(),
        test_data.copy()
    )

    # 2. Spatial component
    train_g, test_g = prepare_spatial_coordinates(train_g, test_g)

    train_g, test_g, _, spatial_stats = add_spatial_signal(
        train_g,
        test_g
    )

    # 3. XGBoost ML component — identical GARUDA V1 logic
    X_train = train_g[BASE_FEATURES]
    X_test = test_g[BASE_FEATURES]
    y_train = train_g["high_risk"].to_numpy()
    y_test = test_g["high_risk"].to_numpy()

    neg = int(np.sum(y_train == 0))
    pos = int(np.sum(y_train == 1))
    scale_pos_weight = neg / pos if pos else 1.0

    model = create_xgboost_model(
        scale_pos_weight=scale_pos_weight
    )
    model.fit(X_train, y_train)

    ml_probability = model.predict_proba(X_test)[:, 1]
    temporal_signal = (
        test_g["temporal_consistency"]
        .astype(float).clip(0.0, 1.0).to_numpy()
    )
    anomaly_signal = (
        test_g["anomaly_signal"]
        .astype(float).clip(0.0, 1.0).to_numpy()
    )
    spatial_signal = (
        test_g["spatial_signal"]
        .astype(float).clip(0.0, 1.0).to_numpy()
    )

    ml_probability = np.nan_to_num(
        ml_probability, nan=0.0, posinf=1.0, neginf=0.0
    )
    temporal_signal = np.nan_to_num(
        temporal_signal, nan=0.0, posinf=1.0, neginf=0.0
    )
    anomaly_signal = np.nan_to_num(
        anomaly_signal, nan=0.0, posinf=1.0, neginf=0.0
    )
    spatial_signal = np.nan_to_num(
        spatial_signal, nan=0.0, posinf=1.0, neginf=0.0
    )

    consensus = np.clip(
        0.40 * ml_probability
        + 0.20 * temporal_signal
        + 0.20 * anomaly_signal
        + 0.20 * spatial_signal,
        0.0,
        1.0
    )

    rows = []

    for threshold in THRESHOLDS:
        pred = (consensus >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).ravel()

        rows.append({
            "Threshold": threshold,
            "Accuracy": accuracy_score(y_test, pred),
            "Precision": precision_score(
                y_test, pred, zero_division=0
            ),
            "Sensitivity": recall_score(
                y_test, pred, zero_division=0
            ),
            "Specificity": tn / (tn + fp) if (tn + fp) else 0.0,
            "F1": f1_score(
                y_test, pred, zero_division=0
            ),
            "MCC": matthews_corrcoef(y_test, pred),
            "Kappa": cohen_kappa_score(y_test, pred),
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp),
            "High_Risk_Count": int(pred.sum()),
            "High_Risk_Rate": float(pred.mean()),
        })

    results = pd.DataFrame(rows)

    csv_path = RESULTS_DIR / "garuda_v1_threshold_sensitivity.csv"
    png_path = RESULTS_DIR / "garuda_v1_threshold_sensitivity.png"

    results.to_csv(csv_path, index=False)

    # Research figure: multiple metric trajectories against threshold.
    fig, ax = plt.subplots(figsize=(10, 6))
    for metric in ["Precision", "Sensitivity", "Specificity", "F1", "MCC"]:
        ax.plot(
            results["Threshold"],
            results[metric],
            marker="o",
            label=metric
        )

    ax.axvline(
        0.66,
        linestyle="--",
        linewidth=1.5,
        label="Frozen threshold = 0.66"
    )

    ax.set_xlabel("GARUDA High-Risk Threshold")
    ax.set_ylabel("Metric Value")
    ax.set_title("GARUDA V1 Threshold Sensitivity")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("\n" + "=" * 90)
    print("THRESHOLD RESULTS")
    print("=" * 90)
    print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    row66 = results[np.isclose(results["Threshold"], 0.66)].iloc[0]

    print("\nFROZEN THRESHOLD = 0.66")
    for col in [
        "Accuracy", "Precision", "Sensitivity",
        "Specificity", "F1", "MCC",
        "TN", "FP", "FN", "TP", "High_Risk_Count"
    ]:
        print(f"{col:18s}: {row66[col]}")

    print("\nSPATIAL CONFIGURATION")
    print(f"eps                 : {SPATIAL_EPS}")
    print(f"min_samples         : {SPATIAL_MIN_SAMPLES}")
    print(f"scale_pos_weight    : {scale_pos_weight:.12f}")

    print("\nSAVED:")
    print(csv_path)
    print(png_path)


if __name__ == "__main__":
    main()
