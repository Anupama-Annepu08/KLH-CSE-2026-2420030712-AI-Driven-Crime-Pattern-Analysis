"""
GARUDA AI - Explainable AI Engine
----------------------------------

SHAP-based explainability for the validated GARUDA XGBoost model.

This module provides:

1. Global SHAP feature importance
2. Local SHAP explanations
3. Top positive contributors
4. Top negative contributors
5. Human-readable feature explanations
6. CSV/JSON output for dashboard integration

The XAI layer does NOT retrain the model.
It explains an already-trained XGBoost model.
"""

from pathlib import Path
import json
import warnings

import numpy as np
import pandas as pd
import shap

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"
XAI_DIR = RESULTS_DIR / "xai"

XAI_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# GARUDA FEATURE DEFINITIONS
# ============================================================

BASE_FEATURES = [
    "daily_crime_count",
    "violent_crime_count",
    "crime_count_7d",
    "crime_count_30d",
]

TEMPORAL_FEATURE = "temporal_consistency"

ANOMALY_FEATURES = [
    "anomaly_score",
    "anomaly_signal",
]

SPATIAL_FEATURES = [
    "spatial_signal",
]


# ============================================================
# FEATURE DESCRIPTIONS
# ============================================================

FEATURE_DESCRIPTIONS = {

    "daily_crime_count":
        "Crime activity observed for the current day.",

    "violent_crime_count":
        "Count of violent crimes associated with the observation period.",

    "crime_count_7d":
        "Crime activity accumulated over the previous 7 days.",

    "crime_count_30d":
        "Crime activity accumulated over the previous 30 days.",

    "temporal_consistency":
        "Measure of consistency in recent temporal crime activity.",

    "anomaly_score":
        "Continuous Isolation Forest score representing unusual crime activity.",

    "anomaly_signal":
        "Binary indication that the observation was identified as anomalous.",

    "spatial_signal":
        "Spatial evidence derived from GARUDA's spatial analysis."
}


# ============================================================
# FEATURE NAME FORMATTER
# ============================================================

def format_feature_name(feature):

    names = {

        "daily_crime_count":
            "Daily Crime Count",

        "violent_crime_count":
            "Violent Crime Count",

        "crime_count_7d":
            "7-Day Crime Count",

        "crime_count_30d":
            "30-Day Crime Count",

        "temporal_consistency":
            "Temporal Consistency",

        "anomaly_score":
            "Anomaly Score",

        "anomaly_signal":
            "Anomaly Signal",

        "spatial_signal":
            "Spatial Signal"
    }

    return names.get(
        feature,
        feature.replace("_", " ").title()
    )


# ============================================================
# VALIDATE FEATURE MATRIX
# ============================================================

def validate_feature_matrix(
    X,
    feature_names
):

    missing = [
        feature
        for feature in feature_names
        if feature not in X.columns
    ]

    if missing:

        raise ValueError(
            "Missing XAI features: "
            f"{missing}"
        )

    data = X[
        feature_names
    ].copy()

    for column in feature_names:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    if data.isna().any().any():

        missing_values = (
            data.isna()
            .sum()
            .to_dict()
        )

        raise ValueError(
            "XAI feature matrix contains "
            f"NaN values: {missing_values}"
        )

    if not np.isfinite(
        data.to_numpy(dtype=float)
    ).all():

        raise ValueError(
            "XAI feature matrix contains "
            "non-finite values."
        )

    return data


# ============================================================
# GET MODEL FEATURE NAMES
# ============================================================

def get_model_feature_names(model):

    if hasattr(
        model,
        "feature_names_in_"
    ):

        return list(
            model.feature_names_in_
        )

    if hasattr(
        model,
        "get_booster"
    ):

        booster = model.get_booster()

        names = booster.feature_names

        if names is not None:

            return list(names)

    raise ValueError(
        "Unable to determine feature names "
        "from the supplied XGBoost model."
    )


# ============================================================
# CREATE SHAP EXPLAINER
# ============================================================

def create_shap_explainer(model):

    if not hasattr(
        model,
        "get_booster"
    ):

        raise TypeError(
            "GARUDA XAI currently expects "
            "an XGBoost classifier."
        )

    return shap.TreeExplainer(
        model
    )


# ============================================================
# COMPUTE SHAP VALUES
# ============================================================

def compute_shap_values(
    model,
    X
):

    feature_names = (
        get_model_feature_names(model)
    )

    X_validated = validate_feature_matrix(
        X,
        feature_names
    )

    explainer = create_shap_explainer(
        model
    )

    shap_values = explainer.shap_values(
        X_validated
    )

    # Binary XGBoost classifiers normally
    # return a 2-D matrix here.
    if isinstance(
        shap_values,
        list
    ):

        if len(shap_values) == 2:

            shap_values = shap_values[1]

        else:

            shap_values = shap_values[0]

    shap_values = np.asarray(
        shap_values
    )

    if shap_values.ndim != 2:

        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{shap_values.shape}"
        )

    if shap_values.shape[
        1
    ] != len(feature_names):

        raise ValueError(
            "SHAP feature dimension does not "
            "match model feature count."
        )

    return (
        explainer,
        X_validated,
        shap_values
    )


# ============================================================
# GLOBAL SHAP IMPORTANCE
# ============================================================

def global_shap_importance(
    X,
    shap_values
):

    importance = np.mean(
        np.abs(shap_values),
        axis=0
    )

    result = pd.DataFrame({

        "Feature":
            X.columns,

        "Mean_Absolute_SHAP":
            importance

    })

    result[
        "Feature_Display"
    ] = result[
        "Feature"
    ].apply(
        format_feature_name
    )

    result[
        "Description"
    ] = result[
        "Feature"
    ].map(
        FEATURE_DESCRIPTIONS
    )

    result = result.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    ).reset_index(
        drop=True
    )

    result[
        "Rank"
    ] = np.arange(
        1,
        len(result) + 1
    )

    return result


# ============================================================
# LOCAL SHAP EXPLANATION
# ============================================================

def explain_instance(
    X_row,
    shap_row,
    expected_value
):

    records = []

    for feature, value, contribution in zip(
        X_row.index,
        X_row.values,
        shap_row
    ):

        records.append({

            "Feature":
                feature,

            "Feature_Display":
                format_feature_name(
                    feature
                ),

            "Value":
                float(value),

            "SHAP_Value":
                float(contribution),

            "Absolute_SHAP":
                float(abs(contribution)),

            "Direction":
                (
                    "increases risk"
                    if contribution > 0
                    else
                    "decreases risk"
                    if contribution < 0
                    else
                    "neutral"
                ),

            "Description":
                FEATURE_DESCRIPTIONS.get(
                    feature,
                    ""
                )
        })

    result = pd.DataFrame(
        records
    )

    result = result.sort_values(
        "Absolute_SHAP",
        ascending=False
    ).reset_index(
        drop=True
    )

    result[
        "Rank"
    ] = np.arange(
        1,
        len(result) + 1
    )

    result[
        "Expected_Value"
    ] = float(
        expected_value
    )

    return result


# ============================================================
# TOP CONTRIBUTORS
# ============================================================

def get_top_contributors(
    explanation,
    top_n=5
):

    positive = explanation[
        explanation["SHAP_Value"] > 0
    ].head(top_n).copy()

    negative = explanation[
        explanation["SHAP_Value"] < 0
    ].head(top_n).copy()

    return (
        positive,
        negative
    )


# ============================================================
# HUMAN-READABLE EXPLANATION
# ============================================================

def generate_human_explanation(
    explanation,
    prediction_probability=None,
    top_n=3
):

    positive, negative = (
        get_top_contributors(
            explanation,
            top_n=top_n
        )
    )

    parts = []

    if prediction_probability is not None:

        parts.append(
            "GARUDA predicted a "
            f"{prediction_probability:.1%} "
            "probability of high risk."
        )

    if not positive.empty:

        positive_features = []

        for _, row in positive.iterrows():

            positive_features.append(
                f"{row['Feature_Display']} "
                f"(SHAP +{row['SHAP_Value']:.4f})"
            )

        parts.append(
            "Factors increasing the prediction: "
            + ", ".join(
                positive_features
            )
            + "."
        )

    if not negative.empty:

        negative_features = []

        for _, row in negative.iterrows():

            negative_features.append(
                f"{row['Feature_Display']} "
                f"(SHAP {row['SHAP_Value']:.4f})"
            )

        parts.append(
            "Factors reducing the prediction: "
            + ", ".join(
                negative_features
            )
            + "."
        )

    return " ".join(
        parts
    )


# ============================================================
# RUN GLOBAL XAI
# ============================================================

def run_global_xai(
    model,
    X_test,
    save=True
):

    print(
        "\n"
        + "=" * 80
    )

    print(
        "GARUDA XAI - GLOBAL SHAP ANALYSIS"
    )

    print(
        "=" * 80
    )

    (
        explainer,
        X_validated,
        shap_values
    ) = compute_shap_values(
        model,
        X_test
    )

    importance = global_shap_importance(
        X_validated,
        shap_values
    )

    print(
        "\nGlobal SHAP Feature Importance:"
    )

    print(
        importance[
            [
                "Rank",
                "Feature_Display",
                "Mean_Absolute_SHAP"
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}"
        )
    )

    if save:

        csv_path = (
            XAI_DIR
            / "garuda_shap_global_importance.csv"
        )

        importance.to_csv(
            csv_path,
            index=False
        )

        print(
            f"\nSaved global SHAP results:"
            f"\n{csv_path}"
        )

    return {
        "explainer": explainer,
        "X": X_validated,
        "shap_values": shap_values,
        "importance": importance
    }


# ============================================================
# RUN LOCAL XAI
# ============================================================

def run_local_xai(
    model,
    X_test,
    row_index=0,
    save=True
):

    (
        explainer,
        X_validated,
        shap_values
    ) = compute_shap_values(
        model,
        X_test
    )

    if row_index < 0:

        row_index = (
            len(X_validated)
            + row_index
        )

    if row_index >= len(
        X_validated
    ):

        raise IndexError(
            f"row_index {row_index} "
            "is outside the test dataset."
        )

    expected_value = (
        explainer.expected_value
    )

    if isinstance(
        expected_value,
        np.ndarray
    ):

        expected_value = (
            expected_value[-1]
        )

    X_row = X_validated.iloc[
        row_index
    ]

    shap_row = shap_values[
        row_index
    ]

    explanation = explain_instance(
        X_row,
        shap_row,
        expected_value
    )

    probability = float(
        model.predict_proba(
            X_validated.iloc[
                [row_index]
            ]
        )[:, 1][0]
    )

    text = generate_human_explanation(
        explanation,
        prediction_probability=probability
    )

    print(
        "\n"
        + "=" * 80
    )

    print(
        "GARUDA XAI - LOCAL SHAP EXPLANATION"
    )

    print(
        "=" * 80
    )

    print(
        f"\nTest observation: {row_index}"
    )

    print(
        f"Predicted high-risk probability: "
        f"{probability:.4f}"
    )

    print(
        "\nFeature contributions:"
    )

    print(
        explanation[
            [
                "Rank",
                "Feature_Display",
                "Value",
                "SHAP_Value",
                "Direction"
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
            f"{x:.6f}"
        )
    )

    print(
        "\nHuman-readable explanation:"
    )

    print(
        text
    )

    if save:

        csv_path = (
            XAI_DIR
            / f"garuda_shap_local_{row_index}.csv"
        )

        explanation.to_csv(
            csv_path,
            index=False
        )

        json_path = (
            XAI_DIR
            / f"garuda_shap_local_{row_index}.json"
        )

        payload = {

            "row_index":
                int(row_index),

            "prediction_probability":
                probability,

            "expected_value":
                float(expected_value),

            "explanation":
                text,

            "features":
                explanation.to_dict(
                    orient="records"
                )
        }

        with open(
            json_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                payload,
                file,
                indent=4
            )

        print(
            f"\nSaved local SHAP CSV:"
            f"\n{csv_path}"
        )

        print(
            f"\nSaved local SHAP JSON:"
            f"\n{json_path}"
        )

    return {
        "explainer": explainer,
        "X": X_validated,
        "shap_values": shap_values,
        "explanation": explanation,
        "prediction_probability":
            probability,
        "human_explanation":
            text
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "GARUDA XAI engine loaded successfully."
    )

    print(
        "This module expects an already-trained "
        "XGBoost GARUDA model and matching "
        "feature matrix."
    )