import pandas as pd
from pathlib import Path


INPUT_PATH = Path(
    "results/uci_feature_analysis.csv"
)

OUTPUT_PATH = Path(
    "results/uci_context_features.csv"
)


# Top 20 features from the completed
# primary UCI correlation analysis.
SELECTED_FEATURES = [
    "PctKids2Par",
    "PctIlleg",
    "PctFam2Par",
    "NumIlleg",
    "racePctWhite",
    "PctYoungKids2Par",
    "NumUnderPov",
    "PctTeen2Par",
    "pctWInvInc",
    "PctPersDenseHous",
    "FemalePctDiv",
    "TotalPctDiv",
    "pctWPubAsst",
    "PctPopUnderPov",
    "racepctblack",
    "PctPersOwnOccup",
    "MalePctDivorce",
    "PctHousNoPhone",
    "PctUnemployed",
    "PctHousLess3BR",
]


CATEGORY_MAP = {
    "PctKids2Par": "Family / Community",
    "PctIlleg": "Family / Community",
    "PctFam2Par": "Family / Community",
    "NumIlleg": "Family / Community",
    "PctYoungKids2Par": "Family / Community",
    "PctTeen2Par": "Family / Community",
    "FemalePctDiv": "Family / Community",
    "TotalPctDiv": "Family / Community",
    "MalePctDivorce": "Family / Community",

    "NumUnderPov": "Economic",
    "pctWInvInc": "Economic",
    "pctWPubAsst": "Economic",
    "PctPopUnderPov": "Economic",

    "racePctWhite": "Demographics",
    "racepctblack": "Demographics",

    "PctPersDenseHous": "Housing",
    "PctPersOwnOccup": "Housing",
    "PctHousNoPhone": "Housing",
    "PctHousLess3BR": "Housing",

    "PctUnemployed": "Employment",
}


def main():

    df = pd.read_csv(INPUT_PATH)

    # Remove the trailing colon used in
    # the UCI attribute definitions.
    df["feature_name_clean"] = (
        df["feature_name"]
        .str.rstrip(":")
    )

    context = df[
        df["feature_name_clean"].isin(
            SELECTED_FEATURES
        )
    ].copy()

    # Safety check.
    missing = sorted(
        set(SELECTED_FEATURES)
        - set(context["feature_name_clean"])
    )

    if missing:
        print(
            "ERROR: Selected features not found:"
        )

        for feature in missing:
            print(f"  - {feature}")

        raise ValueError(
            "Some selected UCI features "
            "were not found."
        )

    context["category"] = (
        context["feature_name_clean"]
        .map(CATEGORY_MAP)
    )

    context["context_role"] = (
        "Historical community-level "
        "association with ViolentCrimesPerPop"
    )

    context = context[
        [
            "feature_column",
            "feature_name_clean",
            "category",
            "missing_percent",
            "valid_n",
            "spearman_rho",
            "p_value",
            "adjusted_p_value",
            "fdr_significant",
            "context_role",
        ]
    ]

    context = context.sort_values(
        by="spearman_rho",
        key=lambda s: s.abs(),
        ascending=False,
    )

    context.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("=" * 70)
    print("GARUDA UCI COMMUNITY CONTEXT")
    print("=" * 70)

    print(
        f"\nSelected features: {len(context)}"
    )

    print("\nCategories:")

    print(
        context["category"]
        .value_counts()
        .to_string()
    )

    print("\nSelected UCI context features:")

    print(
        context[
            [
                "feature_name_clean",
                "category",
                "spearman_rho",
                "adjusted_p_value",
            ]
        ].to_string(index=False)
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()