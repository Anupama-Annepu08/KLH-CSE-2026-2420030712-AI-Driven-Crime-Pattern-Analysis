import pandas as pd
from src.eda import (
    dataset_summary,
    feature_information,
    numerical_features,
    categorical_features
)

DATA_PATH = "data/chicago_crime_2025_2026.csv"


def main():
    print("=" * 90)
    print("GARUDA AI — EXPLORATORY DATA ANALYSIS")
    print("=" * 90)

    print("\nLoading Chicago crime dataset...")
    df = pd.read_csv(DATA_PATH)

    # ---------------------------------------------------------
    # DATASET SUMMARY
    # ---------------------------------------------------------
    summary = dataset_summary(df)

    print("\n" + "=" * 90)
    print("1. DATASET SUMMARY")
    print("=" * 90)

    print(f"Rows             : {summary['rows']:,}")
    print(f"Columns          : {summary['columns']:,}")
    print(f"Missing values   : {summary['missing_values']:,}")
    print(f"Duplicates       : {summary['duplicates']:,}")

    # ---------------------------------------------------------
    # FEATURE INFORMATION
    # ---------------------------------------------------------
    info = feature_information(df)

    print("\n" + "=" * 90)
    print("2. FEATURE INFORMATION")
    print("=" * 90)

    print(info.to_string(index=False))

    # ---------------------------------------------------------
    # NUMERICAL FEATURES
    # ---------------------------------------------------------
    numerical = numerical_features(df)

    print("\n" + "=" * 90)
    print("3. NUMERICAL FEATURES")
    print("=" * 90)

    for feature in numerical:
        print(f"  - {feature}")

    print(f"\nTotal numerical features: {len(numerical)}")

    # ---------------------------------------------------------
    # CATEGORICAL FEATURES
    # ---------------------------------------------------------
    categorical = categorical_features(df)

    print("\n" + "=" * 90)
    print("4. CATEGORICAL FEATURES")
    print("=" * 90)

    for feature in categorical:
        print(f"  - {feature}")

    print(f"\nTotal categorical features: {len(categorical)}")

    # ---------------------------------------------------------
    # DATA TYPES
    # ---------------------------------------------------------
    print("\n" + "=" * 90)
    print("5. DATA TYPES")
    print("=" * 90)

    print(df.dtypes.to_string())

    # ---------------------------------------------------------
    # BASIC STATISTICS
    # ---------------------------------------------------------
    print("\n" + "=" * 90)
    print("6. BASIC NUMERICAL STATISTICS")
    print("=" * 90)

    if numerical:
        print(df[numerical].describe().T.to_string())
    else:
        print("No numerical features found.")

    # ---------------------------------------------------------
    # DATASET PREVIEW
    # ---------------------------------------------------------
    print("\n" + "=" * 90)
    print("7. DATASET PREVIEW")
    print("=" * 90)

    print(df.head(10).to_string(index=False))

    # ---------------------------------------------------------
    # TARGET / CRIME TYPE INFORMATION
    # ---------------------------------------------------------
    print("\n" + "=" * 90)
    print("8. AVAILABLE COLUMNS")
    print("=" * 90)

    for i, column in enumerate(df.columns, start=1):
        print(f"{i:2}. {column}")

    print("\n" + "=" * 90)
    print("EDA SUMMARY COMPLETED")
    print("=" * 90)


if __name__ == "__main__":
    main()