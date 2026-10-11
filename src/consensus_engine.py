import pandas as pd


# ============================================================
# GARUDA CONSENSUS RISK ENGINE
# ============================================================

ML_WEIGHT = 0.40
TEMPORAL_WEIGHT = 0.20
ANOMALY_WEIGHT = 0.20
SPATIAL_WEIGHT = 0.20


def calculate_consensus_risk(
    ml_probability,
    temporal_consistency,
    anomaly_signal,
    spatial_signal
):
    """
    Calculate GARUDA Consensus Risk Score.

    CRS =
        0.40 * ML Risk
        0.20 * Temporal Consistency
        0.20 * Anomaly Signal
        0.20 * Spatial Signal

    All inputs are expected to be normalized to [0, 1].
    """

    ml_probability = float(ml_probability)
    temporal_consistency = float(temporal_consistency)
    anomaly_signal = float(anomaly_signal)
    spatial_signal = float(spatial_signal)

    consensus_score = (
        ML_WEIGHT * ml_probability
        + TEMPORAL_WEIGHT * temporal_consistency
        + ANOMALY_WEIGHT * anomaly_signal
        + SPATIAL_WEIGHT * spatial_signal
    )

    consensus_score = max(
        0.0,
        min(1.0, consensus_score)
    )

    if consensus_score < 0.33:
        risk_level = "LOW"

    elif consensus_score < 0.66:
        risk_level = "MEDIUM"

    else:
        risk_level = "HIGH"

    return {
        "consensus_score": consensus_score,
        "risk_level": risk_level,
        "ml_probability": ml_probability,
        "temporal_consistency": temporal_consistency,
        "anomaly_signal": anomaly_signal,
        "spatial_signal": spatial_signal
    }


def calculate_consensus_dataframe(df):
    """
    Calculate consensus risk for every row in a DataFrame.

    Required columns:
        ml_probability
        temporal_consistency
        anomaly_signal
        spatial_signal
    """

    required_columns = [
        "ml_probability",
        "temporal_consistency",
        "anomaly_signal",
        "spatial_signal"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing consensus columns: {missing_columns}"
        )

    result = df.copy()

    result["consensus_score"] = (
        ML_WEIGHT * result["ml_probability"]
        + TEMPORAL_WEIGHT * result["temporal_consistency"]
        + ANOMALY_WEIGHT * result["anomaly_signal"]
        + SPATIAL_WEIGHT * result["spatial_signal"]
    )

    result["consensus_score"] = (
        result["consensus_score"]
        .clip(0, 1)
    )

    result["risk_level"] = pd.Series(
    "MEDIUM",
    index=result.index,
    dtype="object"
   )

    result.loc[
    result["consensus_score"] < 0.33,
    "risk_level"
    ] = "LOW"

    result.loc[
    result["consensus_score"] >= 0.66,
    "risk_level"
    ] = "HIGH"

    return result


def get_consensus_explanation(result):
    """
    Generate a simple human-readable explanation
    of the consensus decision.
    """

    explanations = []

    if result["ml_probability"] >= 0.5:
        explanations.append(
            "ML model indicates elevated risk"
        )

    if result["temporal_consistency"] >= 0.5:
        explanations.append(
            "recent crime activity shows temporal consistency"
        )

    if result["anomaly_signal"] == 1:
        explanations.append(
            "anomalous activity was detected"
        )

    if result["spatial_signal"] == 1:
        explanations.append(
            "the location has spatial support from DBSCAN"
        )

    if not explanations:
        explanations.append(
            "available signals provide limited evidence of elevated risk"
        )

    return "; ".join(explanations)


def print_consensus_result(result):
    """
    Print a GARUDA consensus decision.
    """

    print("\n" + "=" * 60)
    print("GARUDA CONSENSUS RISK ASSESSMENT")
    print("=" * 60)

    print(
        f"ML Risk Probability : "
        f"{result['ml_probability']:.4f}"
    )

    print(
        f"Temporal Consistency: "
        f"{result['temporal_consistency']:.4f}"
    )

    print(
        f"Anomaly Signal      : "
        f"{result['anomaly_signal']}"
    )

    print(
        f"Spatial Signal      : "
        f"{result['spatial_signal']}"
    )

    print("-" * 60)

    print(
        f"Consensus Risk Score: "
        f"{result['consensus_score']:.4f}"
    )

    print(
        f"Risk Level          : "
        f"{result['risk_level']}"
    )

    print("-" * 60)

    print(
        "Explanation:",
        get_consensus_explanation(result)
    )

    print("=" * 60)