import pandas as pd
import plotly.express as px


def create_confusion_matrix_df(matrix):
    return pd.DataFrame(
        matrix,
        index=["Actual Low Risk", "Actual High Risk"],
        columns=["Predicted Low Risk", "Predicted High Risk"]
    )


def create_roc_curve_df(fpr, tpr):
    return pd.DataFrame(
        {
            "False Positive Rate": fpr,
            "True Positive Rate": tpr
        }
    )


def create_model_comparison(results):
    rows = []

    for model_name, result in results.items():

        row = {
            "Model": model_name,
            "Accuracy": result["metrics"]["Accuracy"],
            "Precision": result["metrics"]["Precision"],
            "Recall": result["metrics"]["Recall"],
            "F1 Score": result["metrics"]["F1 Score"],
            "ROC-AUC": result["metrics"]["ROC-AUC"]
        }

        rows.append(row)

    return (
        pd.DataFrame(rows)
        .sort_values(
            "F1 Score",
            ascending=False
        )
        .reset_index(drop=True)
    )


def get_best_model(results):
    comparison = create_model_comparison(results)

    if comparison.empty:
        return None

    return comparison.iloc[0]["Model"]


def plot_confusion_matrix(matrix):
    matrix_df = create_confusion_matrix_df(
        matrix
    )

    fig = px.imshow(
        matrix_df,
        text_auto=True,
        aspect="auto",
        title="Confusion Matrix"
    )

    return fig


def plot_roc_curve(fpr, tpr, model_name):

    roc_df = create_roc_curve_df(
        fpr,
        tpr
    )

    fig = px.line(
        roc_df,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"ROC Curve - {model_name}"
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1
    )

    return fig