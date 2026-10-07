import pandas as pd
from pathlib import Path


INPUT_PATH = Path(
    "results/uci_feature_analysis.csv"
)

OUTPUT_PATH = Path(
    "results/uci_feature_categories.csv"
)


CATEGORY_MAP = {

    # -----------------------------
    # Demographics
    # -----------------------------
    "population": "Demographics",
    "householdsize": "Demographics",
    "racepctblack": "Demographics",
    "racePctWhite": "Demographics",
    "racePctAsian": "Demographics",
    "racePctHisp": "Demographics",
    "agePct12t21": "Demographics",
    "agePct12t29": "Demographics",
    "agePct16t24": "Demographics",
    "agePct65up": "Demographics",
    "numbUrban": "Demographics",
    "pctUrban": "Demographics",
    "LandArea": "Demographics",
    "PopDens": "Demographics",
    "PctUsePubTrans": "Demographics",

    # -----------------------------
    # Economic
    # -----------------------------
    "medIncome": "Economic",
    "pctWFarmSelf": "Economic",
    "pctWInvInc": "Economic",
    "pctWSocSec": "Economic",
    "pctWPubAsst": "Economic",
    "pctWRetire": "Economic",
    "medFamInc": "Economic",
    "perCapInc": "Economic",
    "whitePerCap": "Economic",
    "blackPerCap": "Economic",
    "indianPerCap": "Economic",
    "AsianPerCap": "Economic",
    "OtherPerCap": "Economic",
    "HispPerCap": "Economic",
    "NumUnderPov": "Economic",
    "PctPopUnderPov": "Economic",

    # -----------------------------
    # Employment
    # -----------------------------
    "pctWWage": "Employment",
    "PctUnemployed": "Employment",
    "PctEmploy": "Employment",
    "PctEmplManu": "Employment",
    "PctEmplProfServ": "Employment",
    "PctOccupManu": "Employment",
    "PctOccupMgmtProf": "Employment",

    # -----------------------------
    # Education
    # -----------------------------
    "PctLess9thGrade": "Education",
    "PctNotHSGrad": "Education",
    "PctBSorMore": "Education",

    # -----------------------------
    # Family / Community
    # -----------------------------
    "PctKids2Par": "Family / Community",
    "PctYoungKids2Par": "Family / Community",
    "PctTeen2Par": "Family / Community",
    "PctFam2Par": "Family / Community",
    "PctIlleg": "Family / Community",
    "NumIlleg": "Family / Community",
    "FemalePctDiv": "Family / Community",
    "MalePctDivorce": "Family / Community",
    "TotalPctDiv": "Family / Community",

    # -----------------------------
    # Housing
    # -----------------------------
    "PctPersOwnOccup": "Housing",
    "PctPersDenseHous": "Housing",
    "PctHousLess3BR": "Housing",
    "PctHousNoPhone": "Housing",
    "PctHousPlumb": "Housing",

    # -----------------------------
    # Crime-related
    # -----------------------------
    "ViolentCrimesPerPop": "Crime-related",
}


def main():

    df = pd.read_csv(INPUT_PATH)

    # --------------------------------------------------
    # IMPORTANT:
    # Keep ONLY the primary-analysis variables.
    # This removes variables with >20% missingness.
    # --------------------------------------------------

    df = df[
        df["primary_analysis_eligible"] == True
    ].copy()

    # Remove non-predictive identifier fields.
    non_predictive = {
        "state",
        "county",
        "community",
        "communityname",
        "fold",
    }

    df = df[
        ~df["feature_name"]
        .str.rstrip(":")
        .isin(non_predictive)
    ].copy()

    # --------------------------------------------------
    # Apply category mapping
    # --------------------------------------------------

    df["feature_name_clean"] = (
        df["feature_name"]
        .str.rstrip(":")
    )

    df["category"] = (
        df["feature_name_clean"]
        .map(CATEGORY_MAP)
    )

    # --------------------------------------------------
    # Safety check:
    # Do NOT silently classify unknown variables.
    # --------------------------------------------------

    unmapped = df[
        df["category"].isna()
    ].copy()

    if not unmapped.empty:

        print("=" * 70)
        print(
            "ERROR: UNMAPPED PRIMARY-ANALYSIS FEATURES"
        )
        print("=" * 70)

        print(
            unmapped[
                [
                    "feature_column",
                    "feature_name"
                ]
            ].to_string(index=False)
        )

        raise ValueError(
            f"{len(unmapped)} primary-analysis "
            f"features are not classified."
        )

    # --------------------------------------------------
    # Final columns
    # --------------------------------------------------

    columns = [
        "feature_column",
        "feature_name_clean",
        "category",
        "missing_percent",
        "valid_n",
        "spearman_rho",
        "p_value",
        "adjusted_p_value",
        "fdr_significant",
    ]

    df = df[columns]

    # Strongest absolute correlations first.
    df = df.sort_values(
        by="spearman_rho",
        key=lambda s: s.abs(),
        ascending=False,
    )

    # --------------------------------------------------
    # Save
    # --------------------------------------------------

    df.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("=" * 70)
    print(
        "GARUDA UCI FEATURE CATEGORIZATION"
    )
    print("=" * 70)

    print(
        f"\nPrimary-analysis variables: {len(df)}"
    )

    print("\nCategory counts:")

    print(
        df["category"]
        .value_counts()
        .to_string()
    )

    print("\nStrongest features:")

    print(
        df[
            [
                "feature_name_clean",
                "category",
                "spearman_rho",
                "adjusted_p_value",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )

    print(
        f"\nSaved: {OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()