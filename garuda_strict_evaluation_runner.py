"""
GARUDA AI — shared strict target-date evaluation runner.

Place this file in the GARUDA AI project root (next to dashboard/ and src/).
Run with the project's Python environment:
    python garuda_strict_evaluation_runner.py

It does not overwrite historical result files. New outputs go to:
    results/strict_v1/

Protocol:
- Target-date holdout starts 2025-03-08 (no target date shared across splits).
- Existing feature engineering and model/component implementations are reused.
- Model/threshold/ablation/repeated-seed experiments share the same cohort and split.
- GARUDA consensus weights remain 0.40/0.20/0.20/0.20; default threshold remains 0.66.
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    matthews_corrcoef, cohen_kappa_score, log_loss
)

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "strict_v1"
OUT.mkdir(parents=True, exist_ok=True)

DASHBOARD_DIR = ROOT / "dashboard"
SRC_DIR = ROOT / "src"
for folder in (DASHBOARD_DIR, SRC_DIR):
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))

import model_comparison_8_models as base
import ablation_experiment as ablation

# The archived ablation module's spatial-density implementation can emit NaN
# spatial_signal values on this cohort. Reuse the validated baseline spatial
# preparation and signal functions so all spatial ablations share one definition.
ablation.prepare_spatial_coordinates = base.prepare_spatial_coordinates
ablation.add_spatial_signal = base.add_spatial_signal

CUTOFF = pd.Timestamp("2025-03-08")
DEFAULT_THRESHOLD = 0.66
WEIGHTS = {"ml": 0.40, "temporal": 0.20, "anomaly": 0.20, "spatial": 0.20}
SEEDS = [42, 52, 62, 72, 82]
ABLATIONS = [
    ("E1 — ML Only", False, False, False),
    ("E2 — ML + Temporal", True, False, False),
    ("E3 — ML + Anomaly", False, True, False),
    ("E4 — ML + Spatial", False, False, True),
    ("E5 — ML + Temporal + Anomaly", True, True, False),
    ("E6 — ML + Temporal + Spatial", True, False, True),
    ("E7 — ML + Anomaly + Spatial", False, True, True),
    ("E8 — ML + Temporal + Anomaly + Spatial", True, True, True),
]
METRIC_NAMES = [
    "Accuracy", "Precision", "Sensitivity", "Specificity", "F1 Score",
    "ROC-AUC", "PR-AUC", "MCC", "Cohen's Kappa", "Log Loss"
]


def build_strict_cohort(df: pd.DataFrame):
    required = [
        "crime_date", "beat", "daily_crime_count", "violent_crime_count",
        "crime_count_7d", "crime_count_30d", "temporal_consistency",
        "next_day_crime_count", "high_risk", "avg_latitude", "avg_longitude",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for strict evaluation: {missing}")

    data = df[required].copy()
    data["crime_date"] = pd.to_datetime(data["crime_date"], errors="coerce")
    numeric = [c for c in required if c not in ("crime_date", "beat")]
    for c in numeric:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    before = len(data)
    # Unlike the historical row-split helper, this requires valid spatial
    # coordinates because every experiment is compared on the same cohort.
    data = data.dropna(subset=required).copy()
    data["high_risk"] = data["high_risk"].astype(int)
    if not set(data["high_risk"].unique()).issubset({0, 1}):
        raise ValueError("Target high_risk must be binary 0/1.")
    data = data.sort_values(["crime_date", "beat"]).reset_index(drop=True)

    # crime_date is the feature/observation date t; the label describes t+1.
    # Split on the LABEL TARGET DATE, not the feature date.
    data["target_date"] = data["crime_date"] + pd.Timedelta(days=1)
    train = data.loc[data["target_date"] < CUTOFF].copy().reset_index(drop=True)
    test = data.loc[data["target_date"] >= CUTOFF].copy().reset_index(drop=True)
    if train.empty or test.empty:
        raise ValueError("Strict target-date split produced an empty partition.")
    if train["target_date"].max() >= test["target_date"].min():
        raise AssertionError("Chronological leakage: training target date overlaps test boundary.")
    overlap = set(train["target_date"].dt.normalize()) & set(test["target_date"].dt.normalize())
    if overlap:
        raise AssertionError(f"Target dates overlap across split: {sorted(overlap)[:5]}")
    if train["high_risk"].nunique() < 2 or test["high_risk"].nunique() < 2:
        raise ValueError("Both partitions must contain both target classes.")

    audit = {
        "protocol": "strict_target_date_holdout",
        "target_date_cutoff_inclusive_test": str(CUTOFF.date()),
        "target_definition": "features observed on crime_date=t predict high_risk for target_date=t+1 day",
        "rows_before_filtering": int(before),
        "rows_after_filtering": int(len(data)),
        "rows_removed": int(before - len(data)),
        "train_rows": int(len(train)),
        "test_rows": int(len(test)),
        "train_feature_date_start": str(train["crime_date"].min().date()),
        "train_feature_date_end": str(train["crime_date"].max().date()),
        "test_feature_date_start": str(test["crime_date"].min().date()),
        "test_feature_date_end": str(test["crime_date"].max().date()),
        "train_target_date_start": str(train["target_date"].min().date()),
        "train_target_date_end": str(train["target_date"].max().date()),
        "test_target_date_start": str(test["target_date"].min().date()),
        "test_target_date_end": str(test["target_date"].max().date()),
        "train_target_distribution": {str(k): int(v) for k, v in train["high_risk"].value_counts().sort_index().items()},
        "test_target_distribution": {str(k): int(v) for k, v in test["high_risk"].value_counts().sort_index().items()},
        "shared_target_dates": len(overlap),
        "coordinate_missing_rows_in_engineered_data": int(
            df[["avg_latitude", "avg_longitude"]].apply(
                lambda col: pd.to_numeric(col, errors="coerce")
            ).isna().any(axis=1).sum()
        ) if "avg_latitude" in df and "avg_longitude" in df else None,
        "feature_columns": [
            "daily_crime_count", "violent_crime_count", "crime_count_7d",
            "crime_count_30d", "temporal_consistency",
        ],
        "weights": WEIGHTS,
        "default_consensus_threshold": DEFAULT_THRESHOLD,
        "seeds": SEEDS,
        "note": (
            "The source loader attaches beat-level median coordinates from raw rows across the "
            "dataset. Coordinates are treated as static geographic metadata; this assumption "
            "must be documented and checked before claiming strict inductive spatial evaluation."
        ),
    }
    if audit["shared_target_dates"] != 0:
        raise AssertionError("Target-date overlap must be zero.")
    return train, test, audit


def metric_pack(y_true, y_pred, probability):
    probability = np.clip(np.asarray(probability, dtype=float), 1e-7, 1 - 1e-7)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    return {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Sensitivity": float(recall_score(y_true, y_pred, zero_division=0)),
        "Specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "F1 Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, probability)),
        "PR-AUC": float(average_precision_score(y_true, probability)),
        "MCC": float(matthews_corrcoef(y_true, y_pred)),
        "Cohen's Kappa": float(cohen_kappa_score(y_true, y_pred)),
        "Log Loss": float(log_loss(y_true, probability, labels=[0, 1])),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        "Predicted_Positive": int(np.sum(np.asarray(y_pred) == 1)),
    }


def get_consensus_scores(train, test, seed=42):
    """Fit GARUDA components on training rows; return test consensus and signals."""
    train_g = train.copy()
    test_g = test.copy()

    old_state = base.RANDOM_STATE
    base.RANDOM_STATE = seed
    try:
        train_g, test_g, _ = base.add_anomaly_signal(train_g, test_g)
        train_g, test_g = base.prepare_spatial_coordinates(train_g, test_g)
        train_g, test_g, _, _ = base.add_spatial_signal(train_g, test_g)

        X_train = train_g[base.BASE_FEATURES]
        y_train = train_g["high_risk"]
        X_test = test_g[base.BASE_FEATURES]
        n0, n1 = int((y_train == 0).sum()), int((y_train == 1).sum())
        scale = n0 / n1 if n1 else 1.0
        model = base.create_xgboost_model(scale_pos_weight=scale)
        model.fit(X_train, y_train)
        ml = model.predict_proba(X_test)[:, 1]
        temporal = test_g["temporal_consistency"].astype(float).clip(0, 1).to_numpy()
        anomaly = test_g["anomaly_signal"].astype(float).clip(0, 1).to_numpy()
        spatial = test_g["spatial_signal"].astype(float).clip(0, 1).to_numpy()
        consensus = (
            WEIGHTS["ml"] * ml
            + WEIGHTS["temporal"] * temporal
            + WEIGHTS["anomaly"] * anomaly
            + WEIGHTS["spatial"] * spatial
        )
        return np.clip(consensus, 0, 1), {
            "ml_probability": ml, "temporal_signal": temporal,
            "anomaly_signal": anomaly, "spatial_signal": spatial,
        }
    finally:
        base.RANDOM_STATE = old_state


def save_df(df, name):
    path = OUT / name
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return path


def main():
    print("=" * 88)
    print("GARUDA AI — STRICT SHARED EVALUATION RUNNER")
    print("=" * 88)
    df = base.load_and_engineer_data()
    train, test, audit = build_strict_cohort(df)
    (OUT / "strict_split_audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print("\nSPLIT AUDIT")
    print(json.dumps(audit, indent=2))
    y_train, y_test = train["high_risk"], test["high_risk"]

    # 1. Eight-model comparison + GARUDA hybrid, on the exact same strict split.
    print("\n[1/4] Model comparison")
    X_train, X_test = train[base.BASE_FEATURES], test[base.BASE_FEATURES]
    models = base.train_baseline_models(X_train, y_train)
    model_rows = []
    for name, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        pred = model.predict(X_test)
        row = {"Model": name, **metric_pack(y_test, pred, prob)}
        model_rows.append(row)

    consensus, signals = get_consensus_scores(train, test, seed=42)
    garuda_pred = (consensus >= DEFAULT_THRESHOLD).astype(int)
    model_rows.append({"Model": "GARUDA Hybrid", **metric_pack(y_test, garuda_pred, consensus)})
    save_df(pd.DataFrame(model_rows), "strict_model_comparison.csv")

    # 2. Threshold sensitivity: fixed model/components and same test set.
    print("\n[2/4] Threshold sensitivity")
    threshold_rows = []
    for threshold in [0.30, 0.33, 0.40, 0.45, 0.50, 0.55, 0.60, 0.66, 0.70, 0.75, 0.80]:
        pred = (consensus >= threshold).astype(int)
        threshold_rows.append({"Threshold": threshold, **metric_pack(y_test, pred, consensus)})
    save_df(pd.DataFrame(threshold_rows), "strict_threshold_sensitivity.csv")

    # 3. E1–E8 ablation on the same split. Existing experiment code is reused.
    print("\n[3/4] E1–E8 ablation")
    ablation_rows = []
    old_ablation_state = ablation.RANDOM_STATE
    try:
        for name, temporal, anomaly, spatial in ABLATIONS:
            result = ablation.run_experiment(
                name, train, test, use_temporal=temporal,
                use_anomaly=anomaly, use_spatial=spatial, random_state=42
            )
            row = {"Experiment": name, **result["metrics"]}
            ablation_rows.append(row)
    finally:
        ablation.RANDOM_STATE = old_ablation_state
    save_df(pd.DataFrame(ablation_rows), "strict_ablation_e1_e8.csv")

    # 4. Repeated seeds for all eight ablations, same immutable cohort.
    print("\n[4/4] Repeated-seed ablation validation")
    seed_rows = []
    for seed in SEEDS:
        print(f"\nRepeated-seed run: {seed}")
        for name, temporal, anomaly, spatial in ABLATIONS:
            result = ablation.run_experiment(
                name, train, test, use_temporal=temporal,
                use_anomaly=anomaly, use_spatial=spatial, random_state=seed
            )
            seed_rows.append({"Seed": seed, "Experiment": name, **result["metrics"]})
    raw = pd.DataFrame(seed_rows)
    save_df(raw, "strict_repeated_seed_raw.csv")
    summary_rows = []
    for name, *_ in ABLATIONS:
        subset = raw.loc[raw["Experiment"] == name]
        row = {"Experiment": name, "Seed_Count": int(subset["Seed"].nunique())}
        for metric in METRIC_NAMES:
            vals = pd.to_numeric(subset[metric], errors="coerce")
            row[f"{metric} Mean"] = float(vals.mean())
            row[f"{metric} Std"] = float(vals.std(ddof=1))
        summary_rows.append(row)
    save_df(pd.DataFrame(summary_rows), "strict_repeated_seed_summary.csv")

    metadata = {
        "protocol": "strict_target_date_holdout",
        "cutoff": str(CUTOFF.date()),
        "weights": WEIGHTS,
        "default_threshold": DEFAULT_THRESHOLD,
        "seeds": SEEDS,
        "outputs": [
            "strict_split_audit.json", "strict_model_comparison.csv",
            "strict_threshold_sensitivity.csv", "strict_ablation_e1_e8.csv",
            "strict_repeated_seed_raw.csv", "strict_repeated_seed_summary.csv",
        ],
        "historical_outputs_overwritten": False,
    }
    (OUT / "strict_run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print("\nCompleted. New files are in results/strict_v1/. Historical result files were not overwritten.")


if __name__ == "__main__":
    main()
