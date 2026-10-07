from pathlib import Path

from src.data_loader import load_csv
from src.feature_engineering import (
    create_temporal_features,
    create_crime_features,
    create_historical_features,
)
from src.model_training import train_classification_model
from src.consensus_engine import calculate_consensus_risk


def main():

    # --------------------------------------------------------
    # Load and prepare data
    # --------------------------------------------------------

    df = load_csv(
        Path("data/chicago_crime_2025_2026.csv")
    )

    df = create_temporal_features(
        df,
        "date"
    )

    df = create_crime_features(df)

    df = create_historical_features(df)

    # --------------------------------------------------------
    # Train E4 XGBoost
    # --------------------------------------------------------

    features = [
        "daily_crime_count",
        "violent_crime_count",
        "crime_count_7d",
        "crime_count_30d",
        "temporal_consistency",
        "anomaly_score",
        "spatial_signal",
    ]

    result = train_classification_model(
        df=df,
        features=features,
        model_name="XGBoost",
        target="high_risk",
    )

    # --------------------------------------------------------
    # Extract actual test signals
    # --------------------------------------------------------

    test_data = result["test_data"]
    probabilities = result["y_probability"]

    print("\n" + "=" * 70)
    print("GARUDA CONSENSUS RISK ENGINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Calculate consensus for every test row
    # --------------------------------------------------------

    consensus_results = []

    for index, probability in zip(
        test_data.index,
        probabilities
    ):

        row = test_data.loc[index]

        consensus = calculate_consensus_risk(
            ml_probability=probability,
            temporal_consistency=row[
                "temporal_consistency"
            ],
            anomaly_signal=row[
                "anomaly_signal"
            ],
            spatial_signal=row[
                "spatial_signal"
            ],
        )

        consensus_results.append(
            consensus
        )

    # --------------------------------------------------------
    # Display first 10 decisions
    # --------------------------------------------------------

    print("\nFirst 10 consensus decisions:")
    print("-" * 70)

    for i, result in enumerate(
        consensus_results[:10],
        start=1
    ):

        print(
            f"{i:02d}. "
            f"ML={result['ml_probability']:.3f} | "
            f"T={result['temporal_consistency']:.3f} | "
            f"A={result['anomaly_signal']:.0f} | "
            f"S={result['spatial_signal']:.0f} | "
            f"CRS={result['consensus_score']:.3f} | "
            f"Risk={result['risk_level']}"
        )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    risk_levels = [
        result["risk_level"]
        for result in consensus_results
    ]

    print("\n" + "-" * 70)
    print("CONSENSUS SUMMARY")
    print("-" * 70)

    print(
        "Total test cases:",
        len(consensus_results)
    )

    print(
        "LOW:",
        risk_levels.count("LOW")
    )

    print(
        "MEDIUM:",
        risk_levels.count("MEDIUM")
    )

    print(
        "HIGH:",
        risk_levels.count("HIGH")
    )

    print("=" * 70)


if __name__ == "__main__":
    main()