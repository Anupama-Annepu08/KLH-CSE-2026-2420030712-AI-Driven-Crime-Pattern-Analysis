from pathlib import Path
import sys

import requests

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


from feature_classifier import classify_features

from feature_engineering import (
    create_temporal_features,
    create_crime_features,
    create_historical_features
)

from feature_selection import (
    correlation_selection,
    mutual_information_selection,
    random_forest_selection
)

# ---------------------------------------------------------------------------
# Compatibility shim for older model_training.py versions.
# Some versions import get_spatial_statistics from src.feature_engineering.
# ---------------------------------------------------------------------------
import src.feature_engineering as _fe

if not hasattr(_fe, "get_spatial_statistics"):
    def _get_spatial_statistics(df):
        if "spatial_signal" not in df.columns:
            raise ValueError("'spatial_signal' does not exist.")
        return {
            "total_rows": int(len(df)),
            "spatial_signal_count": int(df["spatial_signal"].sum()),
            "spatial_signal_rate": float(df["spatial_signal"].mean()),
        }
    _fe.get_spatial_statistics = _get_spatial_statistics

from model_training import (
    train_model,
    get_classification_models,
    get_regression_models
)


st.set_page_config(
    page_title="GARUDA AI",
    page_icon="🦅",
    layout="wide"
)


GARUDA_TEMPLATE = go.layout.Template(
    layout=go.Layout(
        paper_bgcolor="#fffaf2",
        plot_bgcolor="#f5f1e8",
        font=dict(
            family="Inter, Arial, sans-serif",
            color="#49685d",
            size=12
        ),
        title=dict(
            font=dict(
                color="#3f6c5d",
                size=16
            ),
            x=0.02,
            xanchor="left"
        ),
        colorway=[
            "#8FB7A7",
            "#A7C7BA",
            "#BCD8CC",
            "#D0E2DA",
            "#AFA890",
            "#C5D7D0"
        ],
        xaxis=dict(
            color="#6C8178",
            gridcolor="#E9E4DA",
            linecolor="#DED8CC",
            zerolinecolor="#DED8CC"
        ),
        yaxis=dict(
            color="#6C8178",
            gridcolor="#E9E4DA",
            linecolor="#DED8CC",
            zerolinecolor="#DED8CC"
        ),
        legend=dict(
            font=dict(color="#5E756C"),
            bgcolor="rgba(255,253,248,0.75)"
        ),
        margin=dict(l=40, r=30, t=55, b=40)
    )
)

pio.templates["garuda"] = GARUDA_TEMPLATE
px.defaults.template = "garuda"
px.defaults.color_discrete_sequence = [
    "#8FB7A7",
    "#A7C7BA",
    "#BCD8CC",
    "#D0E2DA",
    "#AFA890",
    "#C5D7D0"
]
px.defaults.color_continuous_scale = [
    [0.0, "#F7F2E8"],
    [0.25, "#E7EFEA"],
    [0.5, "#D2E2DB"],
    [0.75, "#AFCABB"],
    [1.0, "#7FA694"]
]

SOFT_GREEN = "#8FB7A7"
SOFT_GREEN_2 = "#A7C7BA"
SOFT_GREEN_3 = "#BCD8CC"
PALE_GREEN = "#D0E2DA"
SOFT_BEIGE = "#CFC4A9"
SOFT_CONTINUOUS = [
    [0.0, "#F8F3E9"],
    [0.2, "#EEF4F0"],
    [0.4, "#DCEAE3"],
    [0.6, "#C4DBD1"],
    [0.8, "#A8C6B8"],
    [1.0, "#86AB99"]
]

def polish_figure(fig):
    fig.update_layout(
        paper_bgcolor="#FFFCF6",
        plot_bgcolor="#F8F5EE",
        font=dict(color="#61786E", size=12),
        title=dict(font=dict(color="#48685D", size=15)),
        legend=dict(bgcolor="rgba(255,252,246,0.85)", bordercolor="#E3E9E4", borderwidth=1),
        hoverlabel=dict(bgcolor="#FFFCF6", font_color="#48685D"),
        margin=dict(l=35, r=25, t=55, b=40)
    )
    fig.update_xaxes(gridcolor="#ECE7DE", linecolor="#DDD7CC", zerolinecolor="#E4DED3")
    fig.update_yaxes(gridcolor="#ECE7DE", linecolor="#DDD7CC", zerolinecolor="#E4DED3")
    return fig


if "model_results" not in st.session_state:
    st.session_state.model_results = {}


if "selected_features_cache" not in st.session_state:
    st.session_state.selected_features_cache = {}


st.markdown(
    """
    <style>
    :root {
        --garuda-green: #4F8673;
        --garuda-green-2: #6FA18F;
        --garuda-green-3: #E8F1EC;
        --garuda-ink: #18322a;
        --garuda-muted: #72867f;
        --garuda-border: #dfe8e4;
        --garuda-panel: #ffffff;
        --garuda-bg: #F7F3EA;
        --garuda-red: #d9534f;
    }

    .stApp {
        background: linear-gradient(180deg, #f6f3eb 0%, #edf2ed 100%);
        color: var(--garuda-ink);
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 1.4rem;
        padding-bottom: 3rem;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a5c42 0%, #074a36 100%);
        border-right: none;
        min-width: 260px;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 1rem;
    }

    [data-testid="stSidebar"] .block-container {
        padding-left: 1rem;
        padding-right: 1rem;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3,
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] span {
        color: #f4fbf8 !important;
    }

    [data-testid="stSidebar"] .stMarkdown {
        color: #f4fbf8 !important;
    }

    [data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,.16);
    }

    .sidebar-brand {
        padding: 8px 10px 18px;
        margin-bottom: 8px;
    }

    .sidebar-brand-row {
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .sidebar-logo {
        width: 42px;
        height: 42px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        background: rgba(255,255,255,.12);
        border: 1px solid rgba(255,255,255,.20);
        font-size: 22px;
        box-shadow: 0 8px 18px rgba(0,0,0,.12);
    }

    .sidebar-brand-title {
        font-size: 25px;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: .2px;
    }

    .sidebar-brand-subtitle {
        font-size: 11px;
        color: rgba(255,255,255,.72);
        margin-top: 1px;
        letter-spacing: .5px;
    }

    .hero {
        background: #fffaf2;
        border: 1px solid var(--garuda-border);
        border-radius: 18px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 10px 28px rgba(21,55,43,.06);
    }

    .hero-top {
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 20px;
    }

    .main-title {
        font-size: 32px;
        font-weight: 850;
        letter-spacing: .1px;
        margin: 0;
        color: #1b2e26;
    }

    .subtitle {
        font-size: 14px;
        color: var(--garuda-muted);
        margin-top: 5px;
    }

    .status-pill {
        padding: 8px 13px;
        border-radius: 999px;
        border: 1px solid #b9ddd0;
        background: #edf8f3;
        color: var(--garuda-green);
        font-size: 12px;
        font-weight: 800;
        white-space: nowrap;
    }

    .section-title {
        font-size: 21px;
        font-weight: 800;
        color: #20332c;
        margin-top: 24px;
        margin-bottom: 12px;
        padding-bottom: 8px;
        border-bottom: 1px solid var(--garuda-border);
    }

    .feature-box {
        padding: 10px 13px;
        border-radius: 10px;
        margin-bottom: 7px;
        background: #f8fbf9;
        border: 1px solid #e1ebe6;
        color: var(--garuda-ink);
    }

    .dataset-banner {
        display: inline-block;
        padding: 7px 12px;
        border-radius: 999px;
        background: #eaf5f0;
        color: var(--garuda-green);
        border: 1px solid #c9e4d9;
        font-weight: 800;
        font-size: 11px;
        margin-bottom: 12px;
        letter-spacing: .4px;
    }

    [data-testid="stMetric"] {
        background: #fffaf2;
        border: 1px solid var(--garuda-border);
        border-radius: 14px;
        padding: 14px 16px;
        box-shadow: 0 6px 18px rgba(21,55,43,.05);
    }

    [data-testid="stMetricLabel"] {
        color: #7b8b84 !important;
        font-size: 12px !important;
    }

    [data-testid="stMetricValue"] {
        color: #1d3229 !important;
        font-weight: 800 !important;
    }

    .stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #0f6f51;
        background: linear-gradient(135deg, #0b694b, #0a5a42);
        color: #ffffff;
        font-weight: 800;
        padding: .62rem 1rem;
        box-shadow: 0 7px 16px rgba(11,93,67,.18);
    }

    .stButton > button:hover {
        background: linear-gradient(135deg, #0f7a59, #0b6047);
        border-color: #0d6149;
        color: #ffffff;
    }


    [data-testid="stPlotlyChart"] {
        background: #fffaf2;
        border: 1px solid #e4ded2;
        border-radius: 14px;
        padding: 8px;
        box-shadow: 0 6px 18px rgba(52, 73, 63, .045);
    }

    .stDataFrame, [data-testid="stTable"] {
        border: 1px solid var(--garuda-border);
        border-radius: 12px;
        overflow: hidden;
        background: #ffffff;
    }

    div[data-baseweb="select"] > div {
        background: #ffffff;
        border-color: #d8e4df;
        border-radius: 10px;
    }

    div[data-baseweb="select"] input {
        color: #20332c;
    }

    [data-testid="stCheckbox"] label {
        background: rgba(255,255,255,.06);
        border-radius: 8px;
        padding: 4px 6px;
    }

    .top-strip {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 12px;
        color: #6f8179;
        font-size: 12px;
    }

    .welcome {
        font-size: 14px;
        color: #465b52;
    }

    .mini-status {
        display: inline-flex;
        align-items: center;
        gap: 7px;
        color: #527065;
    }

    .mini-dot {
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #2fa66f;
        box-shadow: 0 0 0 4px rgba(47,166,111,.10);
    }


    .command-header { background:#ffffff; border:1px solid #dfe8e4; border-radius:18px; padding:18px 22px; margin-bottom:14px; box-shadow:0 8px 22px rgba(21,55,43,.05); }
    .command-header-row { display:flex; justify-content:space-between; align-items:center; gap:20px; }
    .brand-mark { width:48px; height:48px; border-radius:14px; background:#0b5d43; color:#ffffff; display:flex; align-items:center; justify-content:center; font-size:25px; box-shadow:0 8px 18px rgba(11,93,67,.18); }
    .brand-block { display:flex; align-items:center; gap:13px; }
    .brand-name { font-size:25px; font-weight:850; color:#15372c; letter-spacing:.2px; }
    .brand-tagline { font-size:12px; color:#768982; margin-top:2px; }
    .workspace-chip { padding:8px 13px; border:1px solid #cfe1d9; border-radius:999px; background:#edf7f2; color:#0b5d43; font-size:11px; font-weight:800; letter-spacing:.4px; white-space:nowrap; }
    .dataset-card { background:#ffffff; border:1px solid #dfe8e4; border-radius:16px; padding:15px 16px 14px; min-height:125px; box-shadow:0 7px 18px rgba(21,55,43,.04); position:relative; overflow:hidden; }
    .dataset-card.active { border:2px solid #58a689; box-shadow:0 9px 24px rgba(20,92,67,.10); }
    .dataset-card:before { content:""; position:absolute; top:0; left:0; right:0; height:4px; background:#b8d8cb; }
    .dataset-card.active:before { background:#0b5d43; }
    .dataset-card-title { font-size:13px; font-weight:800; color:#244037; margin-bottom:6px; }
    .dataset-card-value { font-size:27px; line-height:1.05; font-weight:850; color:#18382d; }
    .dataset-card-meta { font-size:11px; color:#7b8c85; margin-top:6px; }
    .dataset-card-status { display:inline-block; margin-top:9px; padding:4px 8px; border-radius:999px; background:#eff8f3; border:1px solid #d0e7dc; color:#167253; font-size:10px; font-weight:800; }

    .overview-grid { display:grid; grid-template-columns:2fr 1fr 1fr; gap:14px; margin:4px 0 18px; }
    .overview-card { background:#ffffff; border:1px solid #dfe8e4; border-radius:16px; padding:18px; box-shadow:0 7px 18px rgba(21,55,43,.04); }
    .overview-card h4 { margin:0 0 8px; color:#27463a; font-size:13px; text-transform:uppercase; letter-spacing:.5px; }
    .overview-big { font-size:30px; font-weight:850; color:#17382d; line-height:1.1; }
    .overview-copy { margin-top:7px; color:#71837c; font-size:12px; line-height:1.45; }
    .insight-row { display:grid; grid-template-columns:repeat(3,1fr); gap:14px; margin-bottom:18px; }
    .insight-card { background:#f8fbf9; border:1px solid #dfeae5; border-radius:14px; padding:14px 16px; }
    .insight-title { font-size:12px; font-weight:800; color:#285145; margin-bottom:5px; }
    .insight-text { font-size:12px; color:#667b72; line-height:1.45; }
    .ready-badge { display:inline-block; padding:4px 8px; border-radius:999px; background:#edf8f3; color:#167253; border:1px solid #cfe7dc; font-size:10px; font-weight:800; letter-spacing:.3px; margin-top:6px; }
    @media (max-width: 900px) { .overview-grid, .insight-row { grid-template-columns:1fr; } }
    </style>
    """,
    unsafe_allow_html=True
)


DATASETS = {
    "Chicago Crime":
        PROJECT_ROOT / "data" / "chicago_crime_2025_2026.csv",

    "UCI Communities & Crime":
        PROJECT_ROOT / "data" / "communities_crime.csv",

    "NCR Crime Data":
        PROJECT_ROOT / "data" / "NCRB_Table_1C.1.csv"
}


@st.cache_data
def load_dataset(path):

    return pd.read_csv(path)


def detect_date_columns(df):

    date_columns = []

    for column in df.columns:

        if pd.api.types.is_datetime64_any_dtype(
            df[column]
        ):

            date_columns.append(
                column
            )

        elif df[column].dtype == "object":

            sample = (
                df[column]
                .dropna()
                .astype(str)
                .head(100)
            )

            if len(sample) > 0:

                parsed = pd.to_datetime(
                    sample,
                    errors="coerce"
                )

                if parsed.notna().mean() > 0.7:

                    date_columns.append(
                        column
                    )

    return date_columns


def prepare_uci_for_model(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    data = df[
        numeric_columns
    ].copy()

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    for column in data.columns:

        median = data[column].median()

        if pd.isna(median):
            median = 0

        data[column] = data[column].fillna(
            median
        )

    return data


def prepare_ncrb_for_model(df):

    available_columns = [
        "2020",
        "2021",
        "2022",
        "Mid-Year Projected Population (in Lakhs) (2022)",
        "Rate of Violent Crimes (2022)",
        "Chargesheeting Rate (2022)"
    ]

    available_columns = [
        column
        for column in available_columns
        if column in df.columns
    ]

    data = df[
        available_columns
    ].copy()

    for column in available_columns:

        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    data = data.dropna()

    return data


def engineer_uci_features(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    target = "feature_128" if "feature_128" in numeric_columns else None

    predictors = [
        column
        for column in numeric_columns
        if column != target
    ]

    data = df[predictors].copy()
    missing_count = data.isna().sum(axis=1)
    missing_ratio = data.isna().mean(axis=1)

    for column in data.columns:

        median = data[column].median()

        if pd.isna(median):
            median = 0

        data[column] = data[column].fillna(median)

    standardized = data.copy()

    for column in standardized.columns:

        mean = standardized[column].mean()
        std = standardized[column].std(ddof=0)

        if pd.isna(std) or std == 0:
            std = 1

        standardized[column] = (
            standardized[column] - mean
        ) / std

    engineered = data.copy()

    engineered["missing_feature_count"] = missing_count
    engineered["missing_feature_ratio"] = missing_ratio
    engineered["community_profile_mean"] = standardized.mean(axis=1)
    engineered["community_profile_std"] = standardized.std(axis=1)
    engineered["community_profile_min"] = standardized.min(axis=1)
    engineered["community_profile_max"] = standardized.max(axis=1)
    engineered["community_profile_range"] = (
        engineered["community_profile_max"]
        - engineered["community_profile_min"]
    )

    if target is not None:

        target_values = pd.to_numeric(
            df[target],
            errors="coerce"
        )

        target_values = target_values.fillna(
            target_values.median()
        )

        engineered[target] = target_values

    return engineered


def engineer_ncrb_features(df):

    data = prepare_ncrb_for_model(
        df
    )

    engineered = data.copy()

    if all(
        column in engineered.columns
        for column in ["2020", "2021", "2022"]
    ):

        engineered["crime_total_3yr"] = engineered[
            ["2020", "2021", "2022"]
        ].sum(axis=1)

        engineered["crime_avg_3yr"] = engineered[
            ["2020", "2021", "2022"]
        ].mean(axis=1)

        engineered["crime_change_20_22"] = (
            engineered["2022"] - engineered["2020"]
        )

        engineered["crime_change_21_22"] = (
            engineered["2022"] - engineered["2021"]
        )

        engineered["crime_pct_change_20_22"] = (
            engineered["crime_change_20_22"]
            / engineered["2020"].replace(0, np.nan)
            * 100
        )

        engineered["crime_growth_2021"] = (
            (engineered["2021"] - engineered["2020"])
            / engineered["2020"].replace(0, np.nan)
            * 100
        )

        engineered["crime_growth_2022"] = (
            (engineered["2022"] - engineered["2021"])
            / engineered["2021"].replace(0, np.nan)
            * 100
        )

        engineered["crime_volatility"] = engineered[
            ["2020", "2021", "2022"]
        ].std(axis=1)

        engineered["crime_trend"] = (
            engineered["2022"]
            - engineered[["2020", "2021"]].mean(axis=1)
        )

    if "Chargesheeting Rate (2022)" in engineered.columns:

        engineered["chargesheet_gap"] = (
            100 - engineered["Chargesheeting Rate (2022)"]
        )

    engineered = engineered.replace(
        [np.inf, -np.inf],
        np.nan
    )

    engineered = engineered.fillna(0)

    return engineered


def build_chicago_prediction_dataset(df):

    engineered = create_temporal_features(
        df,
        "date"
    )

    engineered = create_crime_features(
        engineered
    )

    historical = create_historical_features(
        engineered
    )

    return historical


def get_chicago_features(df):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    excluded = [
        "next_day_crime_count",
        "risk_threshold",
        "high_risk",
        "beat"
    ]

    return [
        column
        for column in numeric_columns
        if column not in excluded
    ]


def get_uci_features(df, target):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    return [
        column
        for column in numeric_columns
        if column != target
    ]


def get_ncrb_features(df, target):

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    return [
        column
        for column in numeric_columns
        if column != target
    ]


def render_model_comparison(dataset_name, results):

    if not results:
        return

    rows = []

    for model_name, result in results.items():

        row = {
            "Dataset": dataset_name,
            "Model": model_name
        }

        row.update(
            result["metrics"]
        )

        rows.append(
            row
        )

    comparison = pd.DataFrame(
        rows
    )

    st.markdown(
        "### 🏆 Model Comparison"
    )

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    if "F1 Score" in comparison.columns:

        fig = px.bar(
            comparison,
            x="Model",
            y="F1 Score",
            title=f"{dataset_name} Classification Model Comparison",
            color_discrete_sequence=[SOFT_GREEN]
        )

        st.plotly_chart(
            polish_figure(fig),
            use_container_width=True
        )

    elif "R2 Score" in comparison.columns:

        fig = px.bar(
            comparison,
            x="Model",
            y="R2 Score",
            title=f"{dataset_name} Regression Model Comparison",
            color_discrete_sequence=[SOFT_GREEN]
        )

        st.plotly_chart(
            polish_figure(fig),
            use_container_width=True
        )


def render_classification_results(
    dataset_name,
    results
):

    render_model_comparison(
        dataset_name,
        results
    )

    if not results:
        return

    comparison_rows = []

    for model_name, result in results.items():

        comparison_rows.append({
            "Model": model_name,
            "Accuracy":
                result["metrics"]["Accuracy"],
            "Precision":
                result["metrics"]["Precision"],
            "Recall":
                result["metrics"]["Recall"],
            "F1 Score":
                result["metrics"]["F1 Score"],
            "ROC-AUC":
                result["metrics"]["ROC-AUC"]
        })

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison = comparison.sort_values(
        "F1 Score",
        ascending=False
    )

    best_model = comparison.iloc[0]["Model"]

    best_result = results[
        best_model
    ]

    st.success(
        f"Best {dataset_name} Model: {best_model}"
    )

    metrics = st.columns(5)

    metrics[0].metric(
        "Accuracy",
        f"{best_result['metrics']['Accuracy']:.3f}"
    )

    metrics[1].metric(
        "Precision",
        f"{best_result['metrics']['Precision']:.3f}"
    )

    metrics[2].metric(
        "Recall",
        f"{best_result['metrics']['Recall']:.3f}"
    )

    metrics[3].metric(
        "F1 Score",
        f"{best_result['metrics']['F1 Score']:.3f}"
    )

    metrics[4].metric(
        "ROC-AUC",
        f"{best_result['metrics']['ROC-AUC']:.3f}"
    )

    st.markdown(
        "### 📊 Confusion Matrix"
    )

    matrix = best_result[
        "confusion_matrix"
    ]

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "Actual Low Risk",
            "Actual High Risk"
        ],
        columns=[
            "Predicted Low Risk",
            "Predicted High Risk"
        ]
    )

    st.dataframe(
        matrix_df,
        use_container_width=True
    )

    st.markdown(
        "### 📈 ROC Curve"
    )

    roc_df = pd.DataFrame({
        "False Positive Rate":
            best_result["fpr"],

        "True Positive Rate":
            best_result["tpr"]
    })

    fig = px.line(
        roc_df,
        x="False Positive Rate",
        y="True Positive Rate",
        title=f"ROC Curve - {best_model}",
        color_discrete_sequence=[SOFT_GREEN]
    )

    fig.add_shape(
        type="line",
        x0=0,
        y0=0,
        x1=1,
        y1=1,
        line=dict(color="#C8C2B7", dash="dash", width=1.2)
    )

    st.plotly_chart(
        polish_figure(fig),
        use_container_width=True
    )

    st.markdown(
        "### 🔧 Features Used"
    )

    st.dataframe(
        pd.DataFrame({
            "Feature":
                best_result["features"]
        }),
        use_container_width=True,
        hide_index=True
    )

    st.write(
        f"Training Records: {best_result['train_size']:,}"
    )

    st.write(
        f"Testing Records: {best_result['test_size']:,}"
    )


def render_regression_results(
    dataset_name,
    results
):

    render_model_comparison(
        dataset_name,
        results
    )

    if not results:
        return

    comparison_rows = []

    for model_name, result in results.items():

        comparison_rows.append({
            "Model": model_name,
            "MAE":
                result["metrics"]["MAE"],
            "RMSE":
                result["metrics"]["RMSE"],
            "R2 Score":
                result["metrics"]["R2 Score"]
        })

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison = comparison.sort_values(
        "R2 Score",
        ascending=False
    )

    best_model = comparison.iloc[0]["Model"]

    best_result = results[
        best_model
    ]

    st.success(
        f"Best {dataset_name} Model: {best_model}"
    )

    metrics = st.columns(3)

    metrics[0].metric(
        "MAE",
        f"{best_result['metrics']['MAE']:.4f}"
    )

    metrics[1].metric(
        "RMSE",
        f"{best_result['metrics']['RMSE']:.4f}"
    )

    metrics[2].metric(
        "R² Score",
        f"{best_result['metrics']['R2 Score']:.4f}"
    )

    prediction_df = pd.DataFrame({
        "Actual":
            best_result["y_test"].values,

        "Predicted":
            best_result["predictions"]
    })

    st.markdown(
        "### 📈 Actual vs Predicted"
    )

    fig = px.scatter(
        prediction_df,
        x="Actual",
        y="Predicted",
        title=f"Actual vs Predicted - {best_model}",
        color_discrete_sequence=[SOFT_GREEN]
    )

    minimum = min(
        prediction_df["Actual"].min(),
        prediction_df["Predicted"].min()
    )

    maximum = max(
        prediction_df["Actual"].max(),
        prediction_df["Predicted"].max()
    )

    fig.add_shape(
        type="line",
        x0=minimum,
        y0=minimum,
        x1=maximum,
        y1=maximum
    )

    st.plotly_chart(
        polish_figure(fig),
        use_container_width=True
    )

    st.markdown(
        "### 🔧 Features Used"
    )

    st.dataframe(
        pd.DataFrame({
            "Feature":
                best_result["features"]
        }),
        use_container_width=True,
        hide_index=True
    )

    st.write(
        f"Training Records: {best_result['train_size']:,}"
    )

    st.write(
        f"Testing Records: {best_result['test_size']:,}"
    )


st.markdown(
    '''<div class="command-header">
        <div class="command-header-row">
            <div class="brand-block">
                <div class="brand-mark">🦅</div>
                <div>
                    <div class="brand-name">GARUDA AI</div>
                    <div class="brand-tagline">Crime Pattern Analysis · Public Safety Intelligence</div>
                </div>
            </div>
            <div class="workspace-chip">3 DATASETS · INTELLIGENCE WORKSPACE</div>
        </div>
    </div>''',
    unsafe_allow_html=True
)


st.sidebar.markdown(
    '''<div class="sidebar-brand">
        <div class="sidebar-brand-row">
            <div class="sidebar-logo">🦅</div>
            <div>
                <div class="sidebar-brand-title">GARUDA</div>
                <div class="sidebar-brand-subtitle">PUBLIC SAFETY INTELLIGENCE</div>
            </div>
        </div>
    </div>''',
    unsafe_allow_html=True
)


dataset_name = st.sidebar.selectbox(
    "📂 Select Dataset",
    list(DATASETS.keys())
)

st.sidebar.markdown(
    f'<div style="margin:8px 0 14px; padding:9px 10px; border-radius:10px; background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.12); font-size:11px; color:#eaf7f2;">ACTIVE · {dataset_name.upper()}</div>',
    unsafe_allow_html=True
)


dataset_path = DATASETS[
    dataset_name
]


if not dataset_path.exists():

    st.error(
        f"Dataset not found: {dataset_path}"
    )

    st.stop()


df = load_dataset(
    dataset_path
)


st.sidebar.markdown("---")

st.sidebar.subheader(
    "📊 Data Analysis"
)

show_overview = st.sidebar.checkbox(
    "Dataset Overview"
)

show_missing = st.sidebar.checkbox(
    "Missing Values"
)

show_univariate = st.sidebar.checkbox(
    "Univariate Analysis"
)

show_bivariate = st.sidebar.checkbox(
    "Bivariate Analysis"
)

show_correlation = st.sidebar.checkbox(
    "Correlation Analysis"
)

show_temporal = st.sidebar.checkbox(
    "Temporal Analysis"
)

show_spatial = st.sidebar.checkbox(
    "Spatial Analysis"
)


st.sidebar.markdown("---")

st.sidebar.subheader(
    "⚙️ Feature Engineering"
)

show_feature_engineering = st.sidebar.checkbox(
    "Feature Engineering"
)

show_temporal_engineering = False
show_spatial_engineering = False
show_crime_engineering = False
show_historical_engineering = False
show_uci_engineering = False
show_ncrb_engineering = False


if show_feature_engineering:

    if dataset_name == "Chicago Crime":

        show_temporal_engineering = st.sidebar.checkbox(
            "Temporal Features"
        )

        show_spatial_engineering = st.sidebar.checkbox(
            "Spatial Features"
        )

        show_crime_engineering = st.sidebar.checkbox(
            "Crime Features"
        )

        show_historical_engineering = st.sidebar.checkbox(
            "Historical Features"
        )

    elif dataset_name == "UCI Communities & Crime":

        show_uci_engineering = st.sidebar.checkbox(
            "Community Profile Features"
        )

    elif dataset_name == "NCR Crime Data":

        show_ncrb_engineering = st.sidebar.checkbox(
            "Crime Trend Features"
        )


st.sidebar.markdown("---")

st.sidebar.subheader(
    "🔎 Feature Selection"
)

show_feature_selection = st.sidebar.checkbox(
    "Feature Selection"
)


st.sidebar.markdown("---")

st.sidebar.subheader(
    "🤖 Machine Learning"
)

show_live_prediction = st.sidebar.checkbox(
    "🚀 Live GARUDA Prediction API"
)


show_model_training = st.sidebar.checkbox(
    "Model Training"
)


st.markdown(
    '<div class="section-kicker">Intelligence overview</div>',
    unsafe_allow_html=True
)

records = df.shape[0]
features = df.shape[1]
missing_total = int(df.isnull().sum().sum())
duplicates_total = int(df.duplicated().sum())

if dataset_name == "Chicago Crime":
    overview_title = "Urban incident intelligence"
    overview_copy = "Incident-level crime records with temporal, spatial and crime-type signals for predictive risk analysis."
elif dataset_name == "UCI Communities & Crime":
    overview_title = "Community risk intelligence"
    overview_copy = "Socio-economic and community indicators used to estimate crime-related outcomes across communities."
else:
    overview_title = "National crime intelligence"
    overview_copy = "India-focused aggregate crime indicators supporting regional comparison and predictive analysis."

st.markdown(
    f'''<div class="overview-grid">
        <div class="overview-card">
            <h4>{overview_title}</h4>
            <div class="overview-big">{records:,} records</div>
            <div class="overview-copy">{overview_copy}</div>
        </div>
        <div class="overview-card">
            <h4>Data quality</h4>
            <div class="overview-big">{missing_total:,}</div>
            <div class="overview-copy">Missing values · {duplicates_total:,} duplicate records</div>
        </div>
    </div>''',
    unsafe_allow_html=True
)


cards = [
    ("🗺️", "Chicago Crime", "50,000+", "Real-world incident records"),
    ("🏘️", "UCI Communities & Crime", "1,994", "Community-level crime data"),
    ("🇮🇳", "NCR Crime Data", "39", "Indian state-level records")
]

card_cols = st.columns(3)
for index, (icon, title, value, meta) in enumerate(cards):
    active = " active" if title == dataset_name else ""
    card_cols[index].markdown(
        f'''<div class="dataset-card{active}">
            <div class="dataset-card-title">{icon} {title}</div>
            <div class="dataset-card-value">{value}</div>
            <div class="dataset-card-meta">{meta}</div>
            <div class="dataset-card-status">● CONNECTED</div>
        </div>''',
        unsafe_allow_html=True
    )

st.markdown(
    '<div class="section-title">📌 Active Dataset Snapshot</div>',
    unsafe_allow_html=True
)

snapshot_cols = st.columns(4)
snapshot_values = [
    ("Records", f"{df.shape[0]:,}"),
    ("Features", str(df.shape[1])),
    ("Missing Values", f"{int(df.isnull().sum().sum()):,}"),
    ("Duplicates", f"{int(df.duplicated().sum()):,}")
]
for col, (label, value) in zip(snapshot_cols, snapshot_values):
    col.markdown(
        f'''<div class="snapshot-strip">
            <div class="snapshot-label">{label}</div>
            <div class="snapshot-number">{value}</div>
        </div>''',
        unsafe_allow_html=True
    )


if show_overview:

    st.markdown(
        '<div class="section-title">📋 Dataset Overview</div>',
        unsafe_allow_html=True
    )

    st.dataframe(
        df.head(100),
        use_container_width=True
    )

    feature_info = pd.DataFrame({
        "Feature":
            df.columns,

        "Data Type": [
            str(df[column].dtype)
            for column in df.columns
        ],

        "Missing Values": [
            int(df[column].isnull().sum())
            for column in df.columns
        ],

        "Unique Values": [
            int(df[column].nunique())
            for column in df.columns
        ]
    })

    st.markdown(
        "### Feature Information"
    )

    st.dataframe(
        feature_info,
        use_container_width=True
    )


if show_missing:

    st.markdown(
        '<div class="section-title">🧹 Missing Value Analysis</div>',
        unsafe_allow_html=True
    )

    missing_df = pd.DataFrame({
        "Feature":
            df.columns,

        "Missing Values": [
            int(df[column].isnull().sum())
            for column in df.columns
        ]
    })

    missing_df = missing_df.sort_values(
        "Missing Values",
        ascending=False
    )

    st.dataframe(
        missing_df,
        use_container_width=True
    )

    non_zero_missing = missing_df[
        missing_df["Missing Values"] > 0
    ]

    if not non_zero_missing.empty:

        fig = px.bar(
            non_zero_missing.head(50),
            x="Feature",
            y="Missing Values",
            title="Missing Values by Feature",
            color_discrete_sequence=[SOFT_GREEN_2]
        )

        st.plotly_chart(
            polish_figure(fig),
            use_container_width=True
        )

    else:

        st.success(
            "No missing values found."
        )


if show_univariate:

    st.markdown(
        '<div class="section-title">📊 Univariate Analysis</div>',
        unsafe_allow_html=True
    )

    analysis_type = st.radio(
        "Select Feature Type",
        [
            "Numerical",
            "Categorical"
        ],
        horizontal=True
    )

    if analysis_type == "Numerical":

        numerical_columns = df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numerical_columns:

            selected_column = st.selectbox(
                "Select Numerical Feature",
                numerical_columns,
                key="univariate_numeric"
            )

            fig = px.histogram(
                df,
                x=selected_column,
                title=f"Distribution of {selected_column}",
                marginal="box",
                color_discrete_sequence=[SOFT_GREEN_2]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "No numerical features available."
            )

    else:

        categorical_columns = df.select_dtypes(
            include=["object", "category", "bool"]
        ).columns.tolist()

        if categorical_columns:

            selected_column = st.selectbox(
                "Select Categorical Feature",
                categorical_columns,
                key="univariate_categorical"
            )

            value_counts = (
                df[selected_column]
                .astype(str)
                .value_counts()
                .head(20)
                .reset_index()
            )

            value_counts.columns = [
                selected_column,
                "Count"
            ]

            fig = px.bar(
                value_counts,
                x=selected_column,
                y="Count",
                title=f"Top Values of {selected_column}",
                color_discrete_sequence=[SOFT_GREEN_2]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "No categorical features available."
            )


if show_bivariate:

    st.markdown(
        '<div class="section-title">🔗 Bivariate Analysis</div>',
        unsafe_allow_html=True
    )

    numerical_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numerical_columns) >= 2:

        x_column = st.selectbox(
            "X Feature",
            numerical_columns,
            key="bivariate_x"
        )

        y_options = [
            column
            for column in numerical_columns
            if column != x_column
        ]

        y_column = st.selectbox(
            "Y Feature",
            y_options,
            key="bivariate_y"
        )

        plot_df = df[
            [x_column, y_column]
        ].copy()

        plot_df[x_column] = pd.to_numeric(
            plot_df[x_column],
            errors="coerce"
        )

        plot_df[y_column] = pd.to_numeric(
            plot_df[y_column],
            errors="coerce"
        )

        plot_df = plot_df.dropna()

        fig = px.scatter(
            plot_df,
            x=x_column,
            y=y_column,
            title=f"{x_column} vs {y_column}",
            color_discrete_sequence=[SOFT_GREEN]
        )

        st.plotly_chart(
            polish_figure(fig),
            use_container_width=True
        )

    else:

        st.info(
            "At least two numerical features are required."
        )


if show_correlation:

    st.markdown(
        '<div class="section-title">🔥 Correlation Analysis</div>',
        unsafe_allow_html=True
    )

    numerical_df = df.select_dtypes(
        include=np.number
    )

    if numerical_df.shape[1] >= 2:

        correlation_matrix = numerical_df.corr()

        ranked = correlation_matrix.abs().mean().sort_values(ascending=False)
        heatmap_columns = ranked.head(min(12, len(ranked))).index.tolist()
        heatmap_matrix = correlation_matrix.loc[heatmap_columns, heatmap_columns].round(2)

        fig = px.imshow(
            heatmap_matrix,
            aspect="auto",
            text_auto=".2f",
            color_continuous_scale=SOFT_CONTINUOUS,
            zmin=-1,
            zmax=1,
            title="Feature Correlation Overview"
        )
        fig.update_traces(
            hovertemplate="%{x} × %{y}<br>Correlation: %{z:.2f}<extra></extra>",
            colorbar=dict(title="Correlation")
        )

        st.plotly_chart(
            polish_figure(fig),
            use_container_width=True
        )

        st.caption("Showing the 12 strongest overall numerical features to keep the heatmap readable.")

    else:

        st.info(
            "Not enough numerical features."
        )


if show_temporal:

    st.markdown(
        '<div class="section-title">⏰ Temporal Analysis</div>',
        unsafe_allow_html=True
    )

    date_columns = detect_date_columns(
        df
    )

    if date_columns:

        date_column = st.selectbox(
            "Select Date Feature",
            date_columns,
            key="temporal_date"
        )

        temporal_df = df.copy()

        temporal_df[date_column] = pd.to_datetime(
            temporal_df[date_column],
            errors="coerce"
        )

        temporal_df = temporal_df.dropna(
            subset=[date_column]
        )

        temporal_df["Hour"] = temporal_df[
            date_column
        ].dt.hour

        temporal_df["Day"] = temporal_df[
            date_column
        ].dt.day_name()

        temporal_df["Month"] = temporal_df[
            date_column
        ].dt.month_name()

        hour_counts = (
            temporal_df["Hour"]
            .value_counts()
            .sort_index()
            .reset_index()
        )

        hour_counts.columns = [
            "Hour",
            "Count"
        ]

        fig = px.area(
            hour_counts,
            x="Hour",
            y="Count",
            title="Records by Hour",
            color_discrete_sequence=[SOFT_GREEN]
        )

        st.plotly_chart(
            polish_figure(fig),
            use_container_width=True
        )

        day_counts = (
            temporal_df["Day"]
            .value_counts()
            .reindex([
                "Monday",
                "Tuesday",
                "Wednesday",
                "Thursday",
                "Friday",
                "Saturday",
                "Sunday"
            ])
            .fillna(0)
            .reset_index()
        )

        day_counts.columns = [
            "Day",
            "Count"
        ]

        fig = px.bar(
            day_counts,
            x="Day",
            y="Count",
            title="Records by Day"
        )

        st.plotly_chart(
            polish_figure(fig),
            use_container_width=True
        )

        month_counts = (
            temporal_df["Month"]
            .value_counts()
            .reset_index()
        )

        month_counts.columns = [
            "Month",
            "Count"
        ]

        fig = px.area(
            month_counts,
            x="Month",
            y="Count",
            title="Records by Month",
            color_discrete_sequence=[SOFT_GREEN_2]
        )

        st.plotly_chart(
            polish_figure(fig),
            use_container_width=True
        )

    else:

        st.info(
            "No suitable date feature detected."
        )


if show_spatial:

    st.markdown(
        '<div class="section-title">📍 Spatial Analysis</div>',
        unsafe_allow_html=True
    )

    latitude_column = next(
        (
            column
            for column in df.columns
            if column.lower() in [
                "latitude",
                "lat"
            ]
        ),
        None
    )

    longitude_column = next(
        (
            column
            for column in df.columns
            if column.lower() in [
                "longitude",
                "longitude",
                "lon",
                "lng"
            ]
        ),
        None
    )

    if latitude_column and longitude_column:

        map_df = df[
            [
                latitude_column,
                longitude_column
            ]
        ].copy()

        map_df["lat"] = pd.to_numeric(
            map_df[latitude_column],
            errors="coerce"
        )

        map_df["lon"] = pd.to_numeric(
            map_df[longitude_column],
            errors="coerce"
        )

        map_df = map_df[
            ["lat", "lon"]
        ].dropna()

        map_df = map_df[
            map_df["lat"].between(-90, 90)
            &
            map_df["lon"].between(-180, 180)
        ]

        if not map_df.empty:

            st.map(
                map_df,
                use_container_width=True
            )

    else:

        st.info(
            "No latitude/longitude features detected."
        )


if show_feature_engineering:

    st.markdown(
        '<div class="section-title">⚙️ Feature Engineering</div>',
        unsafe_allow_html=True
    )

    categories = classify_features(
        df
    )

    if show_temporal_engineering:

        st.markdown(
            "### ⏰ Temporal Features"
        )

        date_columns = detect_date_columns(
            df
        )

        if date_columns:

            selected_date = st.selectbox(
                "Date Feature",
                date_columns,
                key="engineering_date"
            )

            temp_df = create_temporal_features(
                df,
                selected_date
            )

            display_columns = [
                selected_date,
                "hour",
                "day_of_week",
                "day_name",
                "month",
                "day",
                "year",
                "is_weekend",
                "is_night",
                "season"
            ]

            available_columns = [
                column
                for column in display_columns
                if column in temp_df.columns
            ]

            st.dataframe(
                temp_df[
                    available_columns
                ].head(50),
                use_container_width=True
            )

        else:

            st.info(
                "No date feature detected."
            )


    if show_spatial_engineering:

        st.markdown(
            "### 📍 Spatial Features"
        )

        spatial_features = [
            feature
            for feature in categories["spatial"]
            if feature in df.columns
        ]

        if spatial_features:

            for feature in spatial_features:

                st.markdown(
                    f'<div class="feature-box">📍 {feature}</div>',
                    unsafe_allow_html=True
                )

            st.dataframe(
                df[
                    spatial_features
                ].head(50),
                use_container_width=True
            )

        else:

            st.info(
                "No spatial features detected."
            )


    if show_crime_engineering:

        st.markdown(
            "### 🚨 Crime Features"
        )

        if "primary_type" in df.columns:

            crime_df = create_crime_features(
                df
            )

            st.dataframe(
                crime_df[
                    [
                        "primary_type",
                        "is_violent"
                    ]
                ].head(50),
                use_container_width=True
            )

            counts = (
                crime_df["is_violent"]
                .value_counts()
                .reset_index()
            )

            counts.columns = [
                "Is Violent",
                "Count"
            ]

            fig = px.bar(
                counts,
                x="Is Violent",
                y="Count",
                title="Violent vs Non-Violent Crime",
                color_discrete_sequence=[SOFT_GREEN_3]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info(
                "Crime-specific features are not available for this dataset."
            )


    if show_historical_engineering:

        st.markdown(
            "### 📈 Historical Features"
        )

        if dataset_name == "Chicago Crime":

            historical_df = build_chicago_prediction_dataset(
                df
            )

            if historical_df is not None:

                st.dataframe(
                    historical_df.head(100),
                    use_container_width=True
                )

            else:

                st.info(
                    "Historical features could not be generated."
                )

        else:

            st.info(
                "Historical beat-level features are implemented for Chicago Crime."
            )


    if show_uci_engineering:

        st.markdown(
            "### 🏘️ UCI Community Profile Features"
        )

        uci_engineered = engineer_uci_features(
            df
        )

        engineered_columns = [
            "missing_feature_count",
            "missing_feature_ratio",
            "community_profile_mean",
            "community_profile_std",
            "community_profile_min",
            "community_profile_max",
            "community_profile_range"
        ]

        available_engineered = [
            column
            for column in engineered_columns
            if column in uci_engineered.columns
        ]

        st.dataframe(
            uci_engineered[available_engineered].head(50),
            use_container_width=True
        )

        missing_plot = px.histogram(
            uci_engineered,
            x="missing_feature_ratio",
            title="UCI Missing Feature Ratio by Community",
            color_discrete_sequence=[SOFT_GREEN_2]
        )

        st.plotly_chart(
            polish_figure(missing_plot),
            use_container_width=True
        )

        if "feature_128" in uci_engineered.columns:

            profile_plot = px.scatter(
                uci_engineered,
                x="community_profile_mean",
                y="feature_128",
                title="Community Profile vs Crime Target",
                color_discrete_sequence=[SOFT_GREEN]
            )

            st.plotly_chart(
                polish_figure(profile_plot),
                use_container_width=True
            )

    if show_ncrb_engineering:

        st.markdown(
            "### 📈 NCRB Crime Trend Features"
        )

        ncrb_engineered = engineer_ncrb_features(
            df
        )

        engineered_columns = [
            "crime_total_3yr",
            "crime_avg_3yr",
            "crime_change_20_22",
            "crime_change_21_22",
            "crime_pct_change_20_22",
            "crime_growth_2021",
            "crime_growth_2022",
            "crime_volatility",
            "crime_trend",
            "chargesheet_gap"
        ]

        available_engineered = [
            column
            for column in engineered_columns
            if column in ncrb_engineered.columns
        ]

        st.dataframe(
            ncrb_engineered[available_engineered].head(50),
            use_container_width=True
        )

        trend_df = pd.DataFrame({
            "Year": [2020, 2021, 2022],
            "Average Crime Count": [
                ncrb_engineered["2020"].mean(),
                ncrb_engineered["2021"].mean(),
                ncrb_engineered["2022"].mean()
            ]
        })

        trend_plot = px.area(
            trend_df,
            x="Year",
            y="Average Crime Count",
            title="NCRB Average Crime Trend",
            color_discrete_sequence=[SOFT_GREEN_2]
        )

        st.plotly_chart(
            polish_figure(trend_plot),
            use_container_width=True
        )

        if "Rate of Violent Crimes (2022)" in ncrb_engineered.columns:

            rate_plot = px.scatter(
                ncrb_engineered,
                x="crime_change_20_22",
                y="Rate of Violent Crimes (2022)",
                title="Crime Change vs Violent Crime Rate",
                color_discrete_sequence=[SOFT_GREEN]
            )

            st.plotly_chart(
                polish_figure(rate_plot),
                use_container_width=True
            )


if show_feature_selection:

    st.markdown(
        '<div class="section-title">🔎 Feature Selection</div>',
        unsafe_allow_html=True
    )

    if dataset_name == "Chicago Crime":

        try:

            selection_df = build_chicago_prediction_dataset(
                df
            )

        except Exception:

            selection_df = df.copy()

    elif dataset_name == "UCI Communities & Crime":

        selection_df = engineer_uci_features(
            df
        )

    else:

        selection_df = engineer_ncrb_features(
            df
        )

    numeric_columns = selection_df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if len(numeric_columns) < 2:

        st.warning(
            "At least two numerical features are required."
        )

    else:

        if dataset_name == "Chicago Crime":

            target = "high_risk"

            if target not in numeric_columns:

                st.warning(
                    "Chicago target high_risk is unavailable. Run feature engineering first."
                )

                target = numeric_columns[-1]

        elif dataset_name == "UCI Communities & Crime":

            target = (
                "feature_128"
                if "feature_128" in numeric_columns
                else numeric_columns[-1]
            )

        else:

            target = (
                "Rate of Violent Crimes (2022)"
                if "Rate of Violent Crimes (2022)"
                in numeric_columns
                else numeric_columns[-1]
            )

        target = st.selectbox(
            "🎯 Select Target Feature",
            numeric_columns,
            index=numeric_columns.index(target),
            key=f"selection_target_{dataset_name}"
        )

        method = st.selectbox(
            "🧠 Select Feature Selection Method",
            [
                "Correlation",
                "Mutual Information",
                "Random Forest Importance"
            ],
            key=f"selection_method_{dataset_name}"
        )

        max_features = min(
            20,
            len(numeric_columns) - 1
        )

        top_k = st.slider(
            "🔢 Number of Features",
            min_value=3,
            max_value=max_features,
            value=min(10, max_features),
            key=f"selection_topk_{dataset_name}"
        )

        selected_features = []
        result = pd.DataFrame()

        if method == "Correlation":

            threshold = st.slider(
                "Correlation Threshold",
                min_value=0.1,
                max_value=1.0,
                value=0.3,
                step=0.05,
                key=f"selection_threshold_{dataset_name}"
            )

            selected_features, result = correlation_selection(
                selection_df,
                target,
                threshold
            )

            plot_data = result.head(
                top_k
            )

            if not plot_data.empty:

                fig = px.bar(
                    plot_data,
                    x="Correlation",
                    y="Feature",
                    orientation="h",
                    title=f"Feature Correlation with {target}",
                    color_discrete_sequence=[SOFT_GREEN]
                )

                fig.update_layout(
                    yaxis={
                        "categoryorder":
                            "total ascending"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        elif method == "Mutual Information":

            selected_features, result = mutual_information_selection(
                selection_df,
                target,
                top_k
            )

            plot_data = result.head(
                top_k
            )

            if not plot_data.empty:

                fig = px.bar(
                    plot_data,
                    x="Mutual Information",
                    y="Feature",
                    orientation="h",
                    title="Mutual Information Scores",
                    color_discrete_sequence=[SOFT_GREEN_2]
                )

                fig.update_layout(
                    yaxis={
                        "categoryorder":
                            "total ascending"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        else:

            selected_features, result = random_forest_selection(
                selection_df,
                target,
                top_k
            )

            plot_data = result.head(
                top_k
            )

            if not plot_data.empty:

                fig = px.bar(
                    plot_data,
                    x="Importance",
                    y="Feature",
                    orientation="h",
                    title="Random Forest Feature Importance",
                    color_discrete_sequence=[SOFT_GREEN_3]
                )

                fig.update_layout(
                    yaxis={
                        "categoryorder":
                            "total ascending"
                    }
                )

                st.plotly_chart(
                    fig,
                    use_container_width=True
                )

        if selected_features:

            st.session_state.selected_features_cache[
                dataset_name
            ] = selected_features

            st.markdown(
                "### ✅ Selected Features"
            )

            st.dataframe(
                pd.DataFrame({
                    "Selected Feature":
                        selected_features
                }),
                use_container_width=True,
                hide_index=True
            )

            st.markdown(
                "### 📊 Selection Results"
            )

            st.dataframe(
                result.head(top_k),
                use_container_width=True,
                hide_index=True
            )

        else:

            st.warning(
                "No features were selected using the current criteria."
            )


if show_model_training:

    st.markdown(
        '<div class="section-title">🤖 Machine Learning & Model Training</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"### 📂 Active Dataset: **{dataset_name}**"
    )


    if dataset_name == "Chicago Crime":

        st.info(
            "Chicago Crime uses classification to predict future high-risk crime conditions."
        )

        try:

            model_df = build_chicago_prediction_dataset(
                df
            )

            if model_df is None or model_df.empty:

                st.error(
                    "Chicago prediction dataset could not be generated."
                )

                st.stop()

            target = "high_risk"

            candidate_features = get_chicago_features(
                model_df
            )

            cached_features = (
                st.session_state.selected_features_cache.get(
                    dataset_name,
                    []
                )
            )

            default_features = [
                feature
                for feature in cached_features
                if feature in candidate_features
            ]

            if not default_features:

                default_features = candidate_features

            selected_features = st.multiselect(
                "🎯 Select Training Features",
                candidate_features,
                default=default_features,
                key="chicago_training_features"
            )

            model_options = list(
                get_classification_models().keys()
            )

            selected_models = st.multiselect(
                "🧠 Select Classification Models",
                model_options,
                default=model_options,
                key="chicago_training_models"
            )

            if st.button(
                "🚀 Train Chicago Models",
                key="button_chicago"
            ):

                if not selected_features:

                    st.warning(
                        "Select at least one training feature."
                    )

                elif not selected_models:

                    st.warning(
                        "Select at least one model."
                    )

                else:

                    results = {}

                    progress = st.progress(
                        0
                    )

                    for index, model_name in enumerate(
                        selected_models
                    ):

                        try:

                            results[
                                model_name
                            ] = train_model(
                                model_df,
                                selected_features,
                                target,
                                model_name,
                                task="classification"
                            )

                        except Exception as error:

                            st.error(
                                f"{model_name}: {error}"
                            )

                        progress.progress(
                            (index + 1) /
                            len(selected_models)
                        )

                    if results:

                        st.session_state.model_results[
                            "Chicago Crime"
                        ] = results

                        render_classification_results(
                            "Chicago Crime",
                            results
                        )

        except Exception as error:

            st.error(
                f"Chicago model preparation failed: {error}"
            )


    elif dataset_name == "UCI Communities & Crime":

        st.info(
            "UCI Communities & Crime uses regression to estimate the Violent Crimes Per Population target."
        )

        model_df = engineer_uci_features(
            df
        )

        target = "feature_128"

        if target not in model_df.columns:

            st.error(
                "UCI target feature_128 was not found."
            )

        else:

            candidate_features = get_uci_features(
                model_df,
                target
            )

            cached_features = (
                st.session_state.selected_features_cache.get(
                    dataset_name,
                    []
                )
            )

            default_features = [
                feature
                for feature in cached_features
                if feature in candidate_features
            ]

            if not default_features:

                default_features = candidate_features[:10]

            selected_features = st.multiselect(
                "🎯 Select Training Features",
                candidate_features,
                default=default_features,
                key="uci_training_features"
            )

            model_options = list(
                get_regression_models().keys()
            )

            selected_models = st.multiselect(
                "🧠 Select Regression Models",
                model_options,
                default=model_options,
                key="uci_training_models"
            )

            if st.button(
                "🚀 Train UCI Models",
                key="button_uci"
            ):

                if not selected_features:

                    st.warning(
                        "Select at least one training feature."
                    )

                elif not selected_models:

                    st.warning(
                        "Select at least one model."
                    )

                else:

                    results = {}

                    progress = st.progress(
                        0
                    )

                    for index, model_name in enumerate(
                        selected_models
                    ):

                        try:

                            results[
                                model_name
                            ] = train_model(
                                model_df,
                                selected_features,
                                target,
                                model_name,
                                task="regression"
                            )

                        except Exception as error:

                            st.error(
                                f"{model_name}: {error}"
                            )

                        progress.progress(
                            (index + 1) /
                            len(selected_models)
                        )

                    if results:

                        st.session_state.model_results[
                            "UCI Communities & Crime"
                        ] = results

                        render_regression_results(
                            "UCI Communities & Crime",
                            results
                        )


    elif dataset_name == "NCR Crime Data":

        st.info(
            "NCRB Crime Data uses regression to estimate the 2022 violent-crime rate."
        )

        model_df = engineer_ncrb_features(
            df
        )

        target = "Rate of Violent Crimes (2022)"

        if target not in model_df.columns:

            st.error(
                "NCRB target column was not found."
            )

        else:

            candidate_features = get_ncrb_features(
                model_df,
                target
            )

            cached_features = (
                st.session_state.selected_features_cache.get(
                    dataset_name,
                    []
                )
            )

            default_features = [
                feature
                for feature in cached_features
                if feature in candidate_features
            ]

            if not default_features:

                default_features = candidate_features

            selected_features = st.multiselect(
                "🎯 Select Training Features",
                candidate_features,
                default=default_features,
                key="ncrb_training_features"
            )

            model_options = list(
                get_regression_models().keys()
            )

            selected_models = st.multiselect(
                "🧠 Select Regression Models",
                model_options,
                default=model_options,
                key="ncrb_training_models"
            )

            if st.button(
                "🚀 Train NCRB Models",
                key="button_ncrb"
            ):

                if not selected_features:

                    st.warning(
                        "Select at least one training feature."
                    )

                elif not selected_models:

                    st.warning(
                        "Select at least one model."
                    )

                else:

                    results = {}

                    progress = st.progress(
                        0
                    )

                    for index, model_name in enumerate(
                        selected_models
                    ):

                        try:

                            results[
                                model_name
                            ] = train_model(
                                model_df,
                                selected_features,
                                target,
                                model_name,
                                task="regression"
                            )

                        except Exception as error:

                            st.error(
                                f"{model_name}: {error}"
                            )

                        progress.progress(
                            (index + 1) /
                            len(selected_models)
                        )

                    if results:

                        st.session_state.model_results[
                            "NCR Crime Data"
                        ] = results

                        render_regression_results(
                            "NCR Crime Data",
                            results
                        )



if show_live_prediction:

    st.markdown(
        '<div class="section-title">🚀 GARUDA Live Crime-Risk Prediction</div>',
        unsafe_allow_html=True
    )

    st.caption(
        "Streamlit dashboard → FastAPI → trained XGBoost E4 model → consensus risk score."
    )

    api_url = st.text_input(
        "FastAPI URL",
        value="http://127.0.0.1:8000",
        key="garuda_api_url"
    )

    health_col, status_col = st.columns([1, 3])

    with health_col:
        check_health = st.button(
            "🔍 Check API",
            key="check_garuda_api"
        )

    with status_col:
        if check_health:
            try:
                health_response = requests.get(
                    f"{api_url.rstrip('/')}/health",
                    timeout=5
                )
                health_response.raise_for_status()
                health_data = health_response.json()
                signals = health_data.get("signals", {})

                st.success(
                    f"API healthy · {health_data.get('model', 'model')} · "
                    f"Signals: {', '.join(signals.keys()) if signals else 'available'}"
                )
            except Exception as error:
                st.error(
                    f"API connection failed: {error}"
                )

    st.markdown("### 🧠 Prediction Inputs")

    input_col1, input_col2, input_col3, input_col4 = st.columns(4)

    with input_col1:
        live_daily = st.number_input(
            "Daily Crime Count",
            min_value=0.0,
            value=25.0,
            step=1.0,
            key="live_daily_crime"
        )

    with input_col2:
        live_violent = st.number_input(
            "Violent Crime Count",
            min_value=0.0,
            value=8.0,
            step=1.0,
            key="live_violent_crime"
        )

    with input_col3:
        live_7d = st.number_input(
            "Crime Count · 7 Days",
            min_value=0.0,
            value=150.0,
            step=1.0,
            key="live_7d"
        )

    with input_col4:
        live_30d = st.number_input(
            "Crime Count · 30 Days",
            min_value=0.0,
            value=620.0,
            step=1.0,
            key="live_30d"
        )

    st.markdown("### 📍 Location")

    location_col1, location_col2 = st.columns(2)

    with location_col1:
        live_latitude = st.number_input(
            "Latitude",
            min_value=-90.0,
            max_value=90.0,
            value=41.881832,
            step=0.000001,
            format="%.6f",
            key="live_latitude"
        )

    with location_col2:
        live_longitude = st.number_input(
            "Longitude",
            min_value=-180.0,
            max_value=180.0,
            value=-87.623177,
            step=0.000001,
            format="%.6f",
            key="live_longitude"
        )

    st.info(
        "Temporal consistency, anomaly detection, and DBSCAN spatial evidence "
        "are calculated automatically by the GARUDA backend."
    )

    if st.button(
        "🚀 Generate GARUDA Risk Prediction",
        type="primary",
        key="generate_garuda_prediction"
    ):

        payload = {
            "daily_crime_count": live_daily,
            "violent_crime_count": live_violent,
            "crime_count_7d": live_7d,
            "crime_count_30d": live_30d,
            "latitude": live_latitude,
            "longitude": live_longitude,
        }

        try:
            prediction_response = requests.post(
                f"{api_url.rstrip('/')}/predict",
                json=payload,
                timeout=30
            )

            prediction_response.raise_for_status()

            prediction = prediction_response.json()

            signals = prediction["signals"]
            consensus = prediction["consensus"]

            score = float(
                consensus["consensus_score"]
            )

            risk_level = consensus["risk_level"]

            ml_probability = float(
                signals["ml_probability"]
            )

            temporal_consistency = float(
                signals["temporal_consistency"]
            )

            anomaly_score = float(
                signals["anomaly_score"]
            )

            anomaly_signal = float(
                signals["anomaly_signal"]
            )

            spatial_signal = float(
                signals["spatial_signal"]
            )

            st.session_state[
                "garuda_last_prediction"
            ] = prediction

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric(
                    "ML Probability",
                    f"{ml_probability:.3f}"
                )

            with metric_col2:
                st.metric(
                    "Consensus Risk Score",
                    f"{score:.3f}"
                )

            with metric_col3:
                st.metric(
                    "Risk Level",
                    risk_level
                )

            st.progress(
                min(max(score, 0.0), 1.0),
                text=f"GARUDA Consensus Risk: {score:.3f}"
            )

            signal_df = pd.DataFrame({
                "Signal": [
                    "ML Probability",
                    "Temporal Consistency",
                    "Anomaly Signal",
                    "Spatial Signal"
                ],
                "Value": [
                    ml_probability,
                    temporal_consistency,
                    anomaly_signal,
                    spatial_signal
                ]
            })

            fig = px.bar(
                signal_df,
                x="Signal",
                y="Value",
                title="GARUDA Multi-Signal Evidence",
                range_y=[0, 1]
            )

            st.plotly_chart(
                polish_figure(fig),
                use_container_width=True
            )

            evidence_col1, evidence_col2 = st.columns(2)

            with evidence_col1:
                st.metric(
                    "Anomaly Score",
                    f"{anomaly_score:.3f}"
                )

            with evidence_col2:
                st.metric(
                    "Spatial Evidence",
                    "Detected"
                    if spatial_signal == 1
                    else "Not Detected"
                )

            location = prediction.get(
                "location",
                {}
            )

            if location:
                st.caption(
                    f"Prediction location: "
                    f"{float(location.get('latitude', live_latitude)):.6f}, "
                    f"{float(location.get('longitude', live_longitude)):.6f}"
                )

            with st.expander(
                "🔎 API Response",
                expanded=False
            ):
                st.json(prediction)

        except Exception as error:
            st.error(
                f"Prediction API request failed: {error}"
            )


st.markdown("---")

st.markdown(
    '<div class="section-title">🏆 GARUDA AI Comparative Model Analysis</div>',
    unsafe_allow_html=True
)


if st.session_state.model_results:

    comparison_rows = []

    for dataset_name_key, results in (
        st.session_state.model_results.items()
    ):

        for model_name, result in results.items():

            row = {
                "Dataset":
                    dataset_name_key,

                "Model":
                    model_name
            }

            row.update(
                result["metrics"]
            )

            comparison_rows.append(
                row
            )

    comparison_df = pd.DataFrame(
        comparison_rows
    )

    if not comparison_df.empty:

        st.dataframe(
            comparison_df,
            use_container_width=True,
            hide_index=True
        )

        classification_df = comparison_df[
            comparison_df["F1 Score"].notna()
        ] if "F1 Score" in comparison_df.columns else pd.DataFrame()

        regression_df = comparison_df[
            comparison_df["R2 Score"].notna()
        ] if "R2 Score" in comparison_df.columns else pd.DataFrame()

        if not classification_df.empty:

            st.markdown(
                "### 📊 Classification Comparison"
            )

            fig = px.bar(
                classification_df,
                x="Model",
                y="F1 Score",
                color="Dataset",
                barmode="group",
                title="Classification F1 Score Comparison",
                color_discrete_sequence=[SOFT_GREEN, SOFT_GREEN_2, SOFT_GREEN_3]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        if not regression_df.empty:

            st.markdown(
                "### 📈 Regression Comparison"
            )

            fig = px.bar(
                regression_df,
                x="Model",
                y="R2 Score",
                color="Dataset",
                barmode="group",
                title="Regression R² Score Comparison",
                color_discrete_sequence=[SOFT_GREEN, SOFT_GREEN_2, SOFT_GREEN_3]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        st.markdown(
            "### 🧠 Dataset-Level Results"
        )

        for dataset_name_key, results in (
            st.session_state.model_results.items()
        ):

            if not results:
                continue

            st.markdown(
                f"#### {dataset_name_key}"
            )

            for model_name, result in results.items():

                metrics_text = " | ".join(
                    [
                        f"{metric}: {value:.4f}"
                        for metric, value
                        in result["metrics"].items()
                    ]
                )

                st.write(
                    f"**{model_name}** → {metrics_text}"
                )

else:

    st.info(
        "Train models for Chicago, UCI, or NCRB to populate the comparative analysis."
    )