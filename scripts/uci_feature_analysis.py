import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path


DATA_PATH = Path("data/communities_crime.csv")
NAMES_PATH = Path("data/communities.names")
RESULTS_DIR = Path("results")

TARGET = "feature_128"
TARGET_NAME = "ViolentCrimesPerPop"

MISSING_THRESHOLD = 0.20


def benjamini_hochberg(p_values):
    """
    Benjamini-Hochberg FDR correction.

    Returns adjusted q-values in the same order
    as the input p-values.
    """

    p_values = np.asarray(
        p_values,
        dtype=float
    )

    n = len(p_values)

    if n == 0:
        return []

    order = np.argsort(p_values)

    sorted_p = p_values[order]

    adjusted = (
        sorted_p
        * n
        / np.arange(1, n + 1)
    )

    # Enforce monotonicity from largest
    # p-value backwards.
    adjusted = np.minimum.accumulate(
        adjusted[::-1]
    )[::-1]

    # Keep adjusted p-values within [0, 1].
    adjusted = np.clip(
        adjusted,
        0.0,
        1.0
    )

    result = np.empty(n)

    result[order] = adjusted

    return result.tolist()


def load_feature_names():
    """
    Extract the 128 actual UCI attribute names
    from communities.names.
    """

    text = NAMES_PATH.read_text(
        errors="ignore"
    )

    start = text.find(
        "Attribute Information:"
    )

    end = text.find(
        "Distribution of the Goal Variable"
    )

    if start == -1 or end == -1:
        raise ValueError(
            "Could not locate the attribute-information "
            "section in communities.names."
        )

    block = text[start:end]

    names = []

    for line in block.splitlines():

        line = line.strip()

        if not line.startswith("-- "):
            continue

        content = line[3:].strip()

        if not content:
            continue

        name = content.split()[0]

        names.append(name)

    if len(names) != 128:
        raise ValueError(
            f"Expected 128 UCI attributes, "
            f"found {len(names)}."
        )

    return names


def build_mapping(columns, names):

    return {
        column: name
        for column, name in zip(
            columns,
            names
        )
    }


def main():

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    df = pd.read_csv(DATA_PATH)

    if TARGET not in df.columns:
        raise ValueError(
            f"Target column '{TARGET}' "
            f"was not found."
        )

    feature_names = load_feature_names()

    mapping = build_mapping(
        df.columns,
        feature_names
    )

    numeric_columns = df.select_dtypes(
        include="number"
    ).columns.tolist()

    predictor_columns = [
        column
        for column in numeric_columns
        if column != TARGET
    ]

    rows = []

    for column in predictor_columns:

        feature = df[column]

        target = df[TARGET]

        missing_pct = float(
            feature.isna().mean()
        )

        valid = pd.concat(
            [feature, target],
            axis=1
        ).dropna()

        valid_n = len(valid)

        if valid_n < 3:

            rho = None
            p_value = None

        else:

            rho, p_value = spearmanr(
                valid[column],
                valid[TARGET]
            )

            rho = float(rho)

            p_value = float(
                p_value
            )

        rows.append(
            {
                "feature_column": column,

                "feature_name":
                    mapping[column],

                "missing_percent":
                    round(
                        missing_pct * 100,
                        4
                    ),

                "valid_n":
                    valid_n,

                "spearman_rho":
                    rho,

                "p_value":
                    p_value,

                "primary_analysis_eligible":
                    missing_pct
                    <= MISSING_THRESHOLD,
            }
        )

    results = pd.DataFrame(rows)

    # --------------------------------------------------
    # Benjamini-Hochberg FDR correction
    # --------------------------------------------------

    eligible = results[
        results[
            "primary_analysis_eligible"
        ]
        & results["p_value"].notna()
    ].copy()

    if not eligible.empty:

        adjusted_p = benjamini_hochberg(
            eligible["p_value"].tolist()
        )

        results.loc[
            eligible.index,
            "adjusted_p_value"
        ] = adjusted_p

        results.loc[
            eligible.index,
            "fdr_significant"
        ] = (
            results.loc[
                eligible.index,
                "adjusted_p_value"
            ] < 0.05
        )

    # --------------------------------------------------
    # Sort by absolute Spearman correlation
    # --------------------------------------------------

    results = results.sort_values(
        by="spearman_rho",
        key=lambda s: s.abs(),
        ascending=False,
        na_position="last"
    )

    # --------------------------------------------------
    # Save complete analysis
    # --------------------------------------------------

    output_path = (
        RESULTS_DIR
        / "uci_feature_analysis.csv"
    )

    results.to_csv(
        output_path,
        index=False
    )

    # --------------------------------------------------
    # Save primary-analysis subset
    # --------------------------------------------------

    primary = results[
        results[
            "primary_analysis_eligible"
        ]
    ].copy()

    primary = primary.sort_values(
        by="spearman_rho",
        key=lambda s: s.abs(),
        ascending=False
    )

    primary_path = (
        RESULTS_DIR
        / "uci_primary_feature_analysis.csv"
    )

    primary.to_csv(
        primary_path,
        index=False
    )

    # --------------------------------------------------
    # Terminal summary
    # --------------------------------------------------

    print("=" * 70)

    print(
        "GARUDA UCI FEATURE ANALYSIS"
    )

    print("=" * 70)

    print(
        f"Dataset shape: {df.shape}"
    )

    print(
        f"Target: "
        f"{TARGET_NAME} ({TARGET})"
    )

    print(
        f"Missingness threshold: "
        f"{MISSING_THRESHOLD * 100:.1f}%"
    )

    print(
        f"\nPrimary-analysis features: "
        f"{len(primary)}"
    )

    print(
        f"Excluded for missingness: "
        f"{len(results) - len(primary)}"
    )

    print(
        "\nTop 20 features by "
        "absolute Spearman correlation:"
    )

    print("-" * 70)

    display_columns = [
        "feature_name",
        "missing_percent",
        "valid_n",
        "spearman_rho",
        "p_value",
        "adjusted_p_value",
        "fdr_significant",
    ]

    print(
        primary[
            display_columns
        ]
        .head(20)
        .to_string(index=False)
    )

    print("\nSaved:")

    print(output_path)

    print(primary_path)


if __name__ == "__main__":
    main()