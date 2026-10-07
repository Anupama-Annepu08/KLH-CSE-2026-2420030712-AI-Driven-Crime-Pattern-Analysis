import json
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt


# ============================================================
# GARUDA AI — EXPLAINABLE AI MODULE
# ============================================================

RANDOM_STATE = 42

XAI_RESULTS_DIR = (
    Path(__file__).resolve().parent.parent
    / "results"
    / "xai"
)

XAI_RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SHAP EXPLAINER
# ============================================================

def create_shap_explainer(model):
    """
    Create a SHAP TreeExplainer for the trained XGBoost model.
    """

    return shap.TreeExplainer(model)


# ============================================================
# SHAP VALUES
# ============================================================

def calculate_shap_values(
    model,
    X
):
    """
    Calculate SHAP values for the supplied feature matrix.
    """

    explainer = create_shap_explainer(
        model
    )

    shap_values = explainer.shap_values(
        X
    )

    return (
        explainer,
        shap_values
    )


# ============================================================
# GLOBAL FEATURE IMPORTANCE
# ============================================================

def calculate_global_feature_importance(
    shap_values,
    X
):
    """
    Calculate global feature importance using
    mean absolute SHAP values.
    """

    values = np.asarray(
        shap_values
    )

    # Handle binary classification output
    # if SHAP returns an additional class dimension.
    if values.ndim == 3:
        values = values[:, :, -1]

    importance = np.mean(
        np.abs(values),
        axis=0
    )

    result = pd.DataFrame({
        "Feature": X.columns,
        "Mean_Absolute_SHAP": importance
    })

    result = result.sort_values(
        "Mean_Absolute_SHAP",
        ascending=False
    ).reset_index(
        drop=True
    )

    return result


# ============================================================
# SAVE GLOBAL IMPORTANCE
# ============================================================

def save_global_feature_importance(
    importance_df
):
    """
    Save global SHAP feature importance.
    """

    path = (
        XAI_RESULTS_DIR
        / "global_shap_feature_importance.csv"
    )

    importance_df.to_csv(
        path,
        index=False
    )

    return path


# ============================================================
# SHAP SUMMARY PLOT
# ============================================================

def save_shap_summary_plot(
    shap_values,
    X
):
    """
    Generate and save SHAP summary plot.
    """

    values = np.asarray(
        shap_values
    )

    if values.ndim == 3:
        values = values[:, :, -1]

    plt.figure()

    shap.summary_plot(
        values,
        X,
        show=False
    )

    plt.tight_layout()

    path = (
        XAI_RESULTS_DIR
        / "shap_summary_plot.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return path


# ============================================================
# SHAP BAR PLOT
# ============================================================

def save_shap_bar_plot(
    shap_values,
    X
):
    """
    Generate a global SHAP bar plot.
    """

    values = np.asarray(
        shap_values
    )

    if values.ndim == 3:
        values = values[:, :, -1]

    plt.figure()

    shap.summary_plot(
        values,
        X,
        plot_type="bar",
        show=False
    )

    plt.tight_layout()

    path = (
        XAI_RESULTS_DIR
        / "shap_feature_importance.png"
    )

    plt.savefig(
        path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    return path


# ============================================================
# LOCAL EXPLANATION
# ============================================================

def explain_prediction(
    model,
    X,
    row_index=0
):
    """
    Generate a local SHAP explanation for
    one prediction.
    """

    if row_index < 0 or row_index >= len(X):
        raise IndexError(
            "row_index is outside the supplied dataset."
        )

    explainer = create_shap_explainer(
        model
    )

    row = X.iloc[
        [row_index]
    ]

    shap_values = explainer.shap_values(
        row
    )

    values = np.asarray(
        shap_values
    )

    if values.ndim == 3:
        values = values[:, :, -1]

    values = values[0]

    explanation = pd.DataFrame({
        "Feature": row.columns,
        "Feature_Value": row.iloc[0].values,
        "SHAP_Value": values,
        "Absolute_SHAP": np.abs(values)
    })

    explanation = explanation.sort_values(
        "Absolute_SHAP",
        ascending=False
    ).reset_index(
        drop=True
    )

    return explanation


# ============================================================
# SAVE LOCAL EXPLANATION
# ============================================================

def save_local_explanation(
    explanation,
    row_index
):
    """
    Save a local prediction explanation.
    """

    path = (
        XAI_RESULTS_DIR
        / f"local_explanation_{row_index}.csv"
    )

    explanation.to_csv(
        path,
        index=False
    )

    return path


# ============================================================
# XAI SUMMARY
# ============================================================

def create_xai_summary(
    model,
    X,
    row_index=0
):
    """
    Run the initial GARUDA XAI pipeline.

    This function currently provides:

    1. Global SHAP importance
    2. SHAP summary plot
    3. SHAP feature importance plot
    4. Local explanation
    """

    print("\n")
    print("=" * 80)
    print("GARUDA AI — EXPLAINABLE AI ANALYSIS")
    print("=" * 80)

    print(
        f"\nSamples analysed: {len(X):,}"
    )

    print(
        f"Features analysed: {X.shape[1]}"
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    print(
        "\nCalculating SHAP values..."
    )

    (
        explainer,
        shap_values
    ) = calculate_shap_values(
        model,
        X
    )

    print(
        "✓ SHAP values calculated."
    )

    # --------------------------------------------------------
    # GLOBAL IMPORTANCE
    # --------------------------------------------------------

    importance = (
        calculate_global_feature_importance(
            shap_values,
            X
        )
    )

    importance_path = (
        save_global_feature_importance(
            importance
        )
    )

    print(
        f"\nGlobal importance saved:\n"
        f"{importance_path}"
    )

    # --------------------------------------------------------
    # SUMMARY PLOT
    # --------------------------------------------------------

    summary_path = (
        save_shap_summary_plot(
            shap_values,
            X
        )
    )

    print(
        f"\nSHAP summary plot saved:\n"
        f"{summary_path}"
    )

    # --------------------------------------------------------
    # BAR PLOT
    # --------------------------------------------------------

    bar_path = (
        save_shap_bar_plot(
            shap_values,
            X
        )
    )

    print(
        f"\nSHAP feature importance plot saved:\n"
        f"{bar_path}"
    )

    # --------------------------------------------------------
    # LOCAL EXPLANATION
    # --------------------------------------------------------

    local_explanation = (
        explain_prediction(
            model,
            X,
            row_index=row_index
        )
    )

    local_path = (
        save_local_explanation(
            local_explanation,
            row_index
        )
    )

    print(
        f"\nLocal explanation saved:\n"
        f"{local_path}"
    )

    # --------------------------------------------------------
    # DISPLAY
    # --------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("TOP SHAP FEATURES")
    print("=" * 80)

    print(
        importance.head(10).to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 80)
    print("LOCAL EXPLANATION")
    print("=" * 80)

    print(
        local_explanation.head(10).to_string(
            index=False
        )
    )

    print("\n")
    print("=" * 80)
    print("GARUDA XAI ANALYSIS COMPLETED")
    print("=" * 80)

    return {
        "explainer": explainer,
        "shap_values": shap_values,
        "global_importance": importance,
        "local_explanation": local_explanation,
        "importance_path": str(
            importance_path
        ),
        "summary_plot_path": str(
            summary_path
        ),
        "bar_plot_path": str(
            bar_path
        ),
        "local_explanation_path": str(
            local_path
        )
    }


# ============================================================
# MODULE TEST
# ============================================================

if __name__ == "__main__":

    print(
        "GARUDA XAI module loaded successfully."
    )

    print(
        "\nAvailable functions:"
    )

    print(
        "  create_shap_explainer()"
    )

    print(
        "  calculate_shap_values()"
    )

    print(
        "  calculate_global_feature_importance()"
    )

    print(
        "  explain_prediction()"
    )

    print(
        "  create_xai_summary()"
    )
