import os
import pickle
import warnings

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

from sklearn.inspection import permutation_importance

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r".\models\garuda_xgboost_e4.pkl"

DATA_PATH = r".\data\chicago_crime_2025_2026.csv"

OUTPUT_DIR = r".\results\xai"

RANDOM_STATE = 42

MAX_SHAP_SAMPLES = 2000


# ============================================================
# DISPLAY
# ============================================================

def section(title):

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# LOAD MODEL
# ============================================================

section("GARUDA AI - XAI / SHAP ANALYSIS")

print("\nLoading GARUDA E4 model...")

with open(MODEL_PATH, "rb") as f:

    artifact = pickle.load(f)


print("Model artifact loaded successfully.")

print("\nArtifact keys:")
print(list(artifact.keys()))


model = artifact["model"]

anomaly_model = artifact.get("anomaly_model")

spatial_model = artifact.get("spatial_model")

features = artifact["features"]

spatial_stats = artifact.get("spatial_stats")

configuration = artifact.get("configuration")


print("\nGARUDA model features:")

for feature in features:

    print(" -", feature)


# ============================================================
# IMPORT PROJECT DATA ENGINEERING
# ============================================================

from ablation_experiment import (
    load_and_engineer_data,
    prepare_common_cohort,
    add_anomaly_signal,
    prepare_spatial_coordinates,
    add_spatial_signal,
)


# ============================================================
# LOAD AND ENGINEER DATA
# ============================================================

section("LOADING AND ENGINEERING DATA")

print("\nCalling the project's existing load_and_engineer_data() function...")

df = load_and_engineer_data()

print("\nEngineered dataset shape:", df.shape)


# ============================================================
# COMMON COHORT
# ============================================================

section("CREATING COMMON EVALUATION COHORT")

train_data, test_data = prepare_common_cohort(df)

print("\nTraining rows:", len(train_data))

print("Testing rows :", len(test_data))


# ============================================================
# TEMPORAL FEATURE
# ============================================================

section("VERIFYING TEMPORAL FEATURE")

if "temporal_consistency" not in train_data.columns:

    raise ValueError(
        "temporal_consistency is missing from engineered data."
    )

print("✓ temporal_consistency available.")


# ============================================================
# ANOMALY SIGNAL
# ============================================================

section("ADDING ANOMALY SIGNAL")

train_data, test_data, fitted_anomaly_model = add_anomaly_signal(
    train_data,
    test_data,
    random_state=RANDOM_STATE
)

print("✓ anomaly_score generated.")


# ============================================================
# SPATIAL SIGNAL
# ============================================================

section("ADDING SPATIAL SIGNAL")

train_data, test_data = prepare_spatial_coordinates(
    train_data,
    test_data
)

(
    train_data,
    test_data,
    fitted_spatial_model,
    fitted_spatial_stats
) = add_spatial_signal(
    train_data,
    test_data
)

print("✓ spatial_signal generated.")


# ============================================================
# VERIFY E4 FEATURES
# ============================================================

section("VERIFYING E4 FEATURE MATRIX")

missing_features = [
    feature
    for feature in features
    if feature not in test_data.columns
]

if missing_features:

    raise ValueError(
        f"Missing E4 features: {missing_features}"
    )


X_train = train_data[features].copy()

X_test = test_data[features].copy()

y_train = train_data["high_risk"].copy()

y_test = test_data["high_risk"].copy()


print("\nE4 features:")

for feature in features:

    print(" -", feature)


print("\nFinal X_train shape:", X_train.shape)

print("Final X_test shape :", X_test.shape)


# ============================================================
# NUMERIC CLEANUP
# ============================================================

X_train = X_train.apply(
    pd.to_numeric,
    errors="coerce"
)

X_test = X_test.apply(
    pd.to_numeric,
    errors="coerce"
)


if X_train.isna().any().any():

    raise ValueError(
        "NaN values found in X_train."
    )


if X_test.isna().any().any():

    raise ValueError(
        "NaN values found in X_test."
    )


# ============================================================
# MODEL COMPATIBILITY
# ============================================================

section("MODEL COMPATIBILITY CHECK")

print("\nModel type:")

print(type(model))


print("\nModel feature count:")

print(model.n_features_in_)


if model.n_features_in_ != len(features):

    raise ValueError(
        "Model feature count does not match E4 feature list."
    )


print("\n✓ Saved E4 model is compatible.")


# ============================================================
# PREDICTIONS
# ============================================================

section("GENERATING GARUDA PREDICTIONS")

y_probability = model.predict_proba(
    X_test
)[:, 1]


y_pred = (
    y_probability >= 0.5
).astype(int)


print("\nProbability range:")

print(
    f"{y_probability.min():.4f} - "
    f"{y_probability.max():.4f}"
)


print(
    "\nPredicted high-risk rows:",
    int(y_pred.sum())
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SHAP
# ============================================================

section("SHAP EXPLAINABILITY")

print(
    "\nSHAP version:",
    shap.__version__
)

print(
    "XGBoost version:"
)

import xgboost

print(
    xgboost.__version__
)


# ============================================================
# SHAP COMPATIBILITY WORKAROUND
# ============================================================

print(
    "\nCreating SHAP TreeExplainer..."
)


try:

    explainer = shap.TreeExplainer(
        model
    )

    print(
        "✓ TreeExplainer created successfully."
    )

    shap_mode = "tree"

except Exception as e:

    print(
        "\nTreeExplainer failed:"
    )

    print(
        repr(e)
    )

    print(
        "\nUsing SHAP-independent XGBoost feature importance "
        "fallback for global explainability."
    )

    explainer = None

    shap_mode = "fallback"


# ============================================================
# SHAP VALUES
# ============================================================

if explainer is not None:

    sample_size = min(
        MAX_SHAP_SAMPLES,
        len(X_test)
    )

    rng = np.random.RandomState(
        RANDOM_STATE
    )

    sample_indices = rng.choice(
        len(X_test),
        size=sample_size,
        replace=False
    )

    X_sample = X_test.iloc[
        sample_indices
    ].copy()


    print(
        f"\nCalculating SHAP values for "
        f"{sample_size:,} test samples..."
    )


    try:

        shap_values = explainer(
            X_sample
        )

        values = shap_values.values


        # Handle possible multiclass / output dimensions.

        if values.ndim == 3:

            values = values[:, :, -1]


        mean_abs_shap = np.abs(
            values
        ).mean(
            axis=0
        )


        shap_importance = pd.DataFrame({

            "feature": X_sample.columns,

            "mean_abs_shap": mean_abs_shap

        }).sort_values(
            "mean_abs_shap",
            ascending=False
        )


        shap_importance.to_csv(
            os.path.join(
                OUTPUT_DIR,
                "garuda_shap_feature_importance.csv"
            ),
            index=False
        )


        print(
            "\n✓ SHAP values calculated."
        )


        print(
            "\nSHAP feature importance:"
        )

        print(
            shap_importance.to_string(
                index=False
            )
        )


        # ----------------------------------------------------
        # SHAP BAR PLOT
        # ----------------------------------------------------

        plt.figure(
            figsize=(10, 6)
        )


        shap.summary_plot(
            values,
            X_sample,
            plot_type="bar",
            show=False
        )


        plt.title(
            "GARUDA AI - SHAP Feature Importance"
        )


        plt.tight_layout()


        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "garuda_shap_feature_importance.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )


        plt.close()


        # ----------------------------------------------------
        # SHAP SUMMARY PLOT
        # ----------------------------------------------------

        plt.figure(
            figsize=(10, 7)
        )


        shap.summary_plot(
            values,
            X_sample,
            show=False
        )


        plt.title(
            "GARUDA AI - SHAP Summary Plot"
        )


        plt.tight_layout()


        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "garuda_shap_summary.png"
            ),
            dpi=300,
            bbox_inches="tight"
        )


        plt.close()


        print(
            "\n✓ SHAP plots saved."
        )


    except Exception as e:

        print(
            "\nSHAP calculation failed:"
        )

        print(
            repr(e)
        )

        shap_mode = "fallback"


# ============================================================
# FALLBACK FEATURE IMPORTANCE
# ============================================================

if shap_mode == "fallback":

    section(
        "GLOBAL FEATURE IMPORTANCE FALLBACK"
    )


    booster_importance = model.feature_importances_


    importance_df = pd.DataFrame({

        "feature": features,

        "importance": booster_importance

    }).sort_values(
        "importance",
        ascending=False
    )


    importance_df.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "garuda_xgboost_feature_importance.csv"
        ),
        index=False
    )


    print(
        "\nXGBoost feature importance:"
    )

    print(
        importance_df.to_string(
            index=False
        )
    )


    plt.figure(
        figsize=(10, 6)
    )


    sorted_df = importance_df.sort_values(
        "importance"
    )


    plt.barh(
        sorted_df["feature"],
        sorted_df["importance"]
    )


    plt.xlabel(
        "XGBoost Feature Importance"
    )


    plt.ylabel(
        "Feature"
    )


    plt.title(
        "GARUDA AI - XGBoost Feature Importance"
    )


    plt.tight_layout()


    plt.savefig(
        os.path.join(
            OUTPUT_DIR,
            "garuda_xgboost_feature_importance.png"
        ),
        dpi=300,
        bbox_inches="tight"
    )


    plt.close()


    print(
        "\n✓ Fallback feature-importance plot saved."
    )


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

section(
    "PERMUTATION IMPORTANCE"
)


print(
    "\nCalculating permutation importance..."
)


sample_size = min(
    2000,
    len(X_test)
)


rng = np.random.RandomState(
    RANDOM_STATE
)


sample_indices = rng.choice(
    len(X_test),
    size=sample_size,
    replace=False
)


X_perm = X_test.iloc[
    sample_indices
].copy()


y_perm = y_test.iloc[
    sample_indices
].copy()


perm_result = permutation_importance(
    model,
    X_perm,
    y_perm,
    scoring="roc_auc",
    n_repeats=10,
    random_state=RANDOM_STATE,
    n_jobs=-1
)


permutation_df = pd.DataFrame({

    "feature": features,

    "mean_importance":
        perm_result.importances_mean,

    "std_importance":
        perm_result.importances_std

}).sort_values(
    "mean_importance",
    ascending=False
)


permutation_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "garuda_permutation_importance.csv"
    ),
    index=False
)


print(
    "\nPermutation importance:"
)

print(
    permutation_df.to_string(
        index=False
    )
)


# ============================================================
# PERMUTATION PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)


sorted_perm = permutation_df.sort_values(
    "mean_importance"
)


plt.barh(
    sorted_perm["feature"],
    sorted_perm["mean_importance"]
)


plt.xlabel(
    "Mean decrease in ROC-AUC"
)


plt.ylabel(
    "Feature"
)


plt.title(
    "GARUDA AI - Permutation Feature Importance"
)


plt.tight_layout()


plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "garuda_permutation_importance.png"
    ),
    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\n✓ Permutation plot saved."
)


# ============================================================
# LOCAL EXPLANATIONS
# ============================================================

section(
    "LOCAL HIGH-RISK EXPLANATIONS"
)


prediction_df = test_data.copy()


prediction_df["garuda_probability"] = (
    y_probability
)


prediction_df["garuda_prediction"] = (
    y_pred
)


high_risk_indices = np.argsort(
    y_probability
)[::-1][:20]


local_rows = []


for rank, idx in enumerate(
    high_risk_indices,
    start=1
):

    row = X_test.iloc[
        idx
    ]


    local_record = {

        "rank": rank,

        "original_index": int(
            idx
        ),

        "garuda_probability": float(
            y_probability[idx]
        )

    }


    for feature in features:

        local_record[
            feature
        ] = float(
            row[feature]
        )


    local_rows.append(
        local_record
    )


local_df = pd.DataFrame(
    local_rows
)


local_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "garuda_top_20_high_risk_cases.csv"
    ),
    index=False
)


print(
    "\nTop 20 high-risk cases:"
)


print(
    local_df[
        [
            "rank",
            "garuda_probability"
        ]
    ].to_string(
        index=False
    )
)


# ============================================================
# XAI SUMMARY
# ============================================================

section(
    "XAI SUMMARY"
)


summary = {

    "model":
        "GARUDA E4 XGBoost",

    "shap_version":
        shap.__version__,

    "xgboost_version":
        xgboost.__version__,

    "shap_mode":
        shap_mode,

    "test_samples":
        int(len(X_test)),

    "features":
        features,

    "predicted_high_risk":
        int(y_pred.sum()),

    "mean_probability":
        float(y_probability.mean()),

    "max_probability":
        float(y_probability.max()),

    "min_probability":
        float(y_probability.min())

}


summary_df = pd.DataFrame(
    [summary]
)


summary_df.to_json(
    os.path.join(
        OUTPUT_DIR,
        "garuda_xai_summary.json"
    ),
    orient="records",
    indent=4
)


print(
    "\nXAI analysis completed."
)


print(
    "\nOutput directory:"
)

print(
    os.path.abspath(
        OUTPUT_DIR
    )
)


print(
    "\nGenerated files:"
)


for filename in sorted(
    os.listdir(
        OUTPUT_DIR
    )
):

    print(
        " -",
        filename
    )


print(
    "\n" + "=" * 80
)

print(
    "GARUDA AI XAI ANALYSIS COMPLETE"
)

print(
    "=" * 80
)