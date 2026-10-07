import streamlit as st
import pandas as pd
import plotly.graph_objects as go


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="GARUDA AI - Model Comparison",
    page_icon="🦅",
    layout="wide"
)


# ============================================================
# COLORS
# ============================================================

WHITE = "#FFFDF8"
NAVY = "#172554"
BLUE = "#2563EB"
PURPLE = "#7C3AED"
CYAN = "#0891B2"
PINK = "#DB2777"
ORANGE = "#EA580C"
RED = "#DC2626"
INDIGO = "#4F46E5"

TEXT = "#334155"
GRID = "#E2E8F0"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background: #FFFDF8;
    }

    .main .block-container {
        max-width: 1500px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        color: #172554 !important;
    }

    .main-title {
        font-size: 34px;
        font-weight: 900;
        color: #172554;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 14px;
        color: #64748B;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 900;
        color: #172554;
        margin-top: 30px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        font-size: 13px;
        color: #64748B;
        margin-bottom: 12px;
    }

    [data-testid="stPlotlyChart"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 16px;
        padding: 8px;
        box-shadow: 0 8px 25px rgba(15, 23, 42, 0.06);
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🦅 GARUDA AI — Intelligence Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="subtitle">
        AI-Driven Crime Pattern Analysis & Public Safety Intelligence System
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SECTION 1
# GARUDA HYBRID ARCHITECTURE
# ============================================================

st.markdown(
    """
    <div class="section-title">
        GARUDA Hybrid Multi-Signal Crime Risk Verification Architecture
    </div>

    <div class="section-subtitle">
        Multi-model prediction followed by temporal, anomaly and spatial
        cross-verification before generating the final consensus risk.
    </div>
    """,
    unsafe_allow_html=True
)


fig_arch = go.Figure()


# ------------------------------------------------------------
# ARCHITECTURE NODES
# ------------------------------------------------------------

nodes = [

    ("Crime Data", 0.05, 0.50, NAVY),

    ("7 Baseline ML Models", 0.25, 0.50, BLUE),

    ("ML Risk\nProbability", 0.43, 0.72, PURPLE),

    ("Temporal\nConsistency", 0.43, 0.50, CYAN),

    ("Anomaly\nSignal", 0.43, 0.28, PINK),

    ("Spatial\nSignal", 0.60, 0.50, ORANGE),

    ("Cross-\nVerification", 0.76, 0.50, INDIGO),

    ("Consensus\nRisk Score", 0.90, 0.65, RED),

    ("LOW / MEDIUM /\nHIGH", 0.90, 0.30, NAVY)
]


# ------------------------------------------------------------
# DRAW NODES
# ------------------------------------------------------------

for label, x, y, color in nodes:

    fig_arch.add_trace(
        go.Scatter(
            x=[x],
            y=[y],
            mode="markers+text",
            marker=dict(
                size=70,
                color=color,
                line=dict(
                    color="white",
                    width=3
                )
            ),
            text=[label],
            textposition="middle center",
            textfont=dict(
                color="white",
                size=11,
                family="Arial Black"
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )


# ------------------------------------------------------------
# CONNECTIONS
# ------------------------------------------------------------

connections = [

    (0.08, 0.50, 0.22, 0.50),

    (0.28, 0.50, 0.40, 0.72),
    (0.28, 0.50, 0.40, 0.50),
    (0.28, 0.50, 0.40, 0.28),

    (0.46, 0.72, 0.57, 0.50),
    (0.46, 0.50, 0.57, 0.50),
    (0.46, 0.28, 0.57, 0.50),

    (0.63, 0.50, 0.73, 0.50),

    (0.79, 0.50, 0.87, 0.65),
    (0.79, 0.50, 0.87, 0.30)
]


for x1, y1, x2, y2 in connections:

    fig_arch.add_annotation(
        x=x2,
        y=y2,
        ax=x1,
        ay=y1,
        xref="x",
        yref="y",
        axref="x",
        ayref="y",
        showarrow=True,
        arrowhead=2,
        arrowsize=1,
        arrowwidth=2,
        arrowcolor="#94A3B8"
    )


fig_arch.update_layout(

    height=520,

    paper_bgcolor=WHITE,

    plot_bgcolor=WHITE,

    xaxis=dict(
        visible=False,
        range=[0, 1]
    ),

    yaxis=dict(
        visible=False,
        range=[0, 1]
    ),

    margin=dict(
        l=20,
        r=20,
        t=30,
        b=20
    ),

    title=dict(
        text=(
            "<b>GARUDA Hybrid Multi-Signal Crime Risk "
            "Verification Model</b>"
        ),
        font=dict(
            size=21,
            color=NAVY
        ),
        x=0.02,
        xanchor="left"
    )
)


st.plotly_chart(
    fig_arch,
    use_container_width=True
)


# ============================================================
# SECTION 2
# 8-MODEL COMPARATIVE ANALYSIS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        GARUDA Hybrid Multi-Signal Crime Risk Verification Model
    </div>

    <div class="section-subtitle">
        8-Model Comparative Analysis using the completed chronological
        evaluation cohort.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ACTUAL 8-MODEL RESULTS
# ============================================================
# Common chronological evaluation cohort:
# Training rows: 17,755
# Testing rows : 4,439
#
# These are the actual values produced by the completed
# backend 8-model experiment.
# ============================================================

models = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Extra Trees",
        "HistGradientBoosting",
        "XGBoost",
        "MLP Neural Network",
        "GARUDA Hybrid"
    ],

    # --------------------------------------------------------
    # CHART 1
    # --------------------------------------------------------

    "Accuracy": [
        0.5540,
        0.6848,
        0.6880,
        0.6848,
        0.5783,
        0.6763,
        0.6369,
        0.8941
    ],

    "Precision": [
        0.0992,
        0.0979,
        0.0971,
        0.0979,
        0.0920,
        0.1005,
        0.0990,
        0.1214
    ],

    "Sensitivity": [
        0.5495,
        0.3462,
        0.3379,
        0.3462,
        0.4670,
        0.3709,
        0.4231,
        0.0467
    ],

    "F1 Score": [
        0.1681,
        0.1526,
        0.1508,
        0.1526,
        0.1537,
        0.1582,
        0.1604,
        0.0675
    ],

    "ROC-AUC": [
        0.5683,
        0.5536,
        0.5482,
        0.5538,
        0.5552,
        0.5532,
        0.5619,
        0.5235
    ],

    "PR-AUC": [
        0.1005,
        0.0964,
        0.0970,
        0.0967,
        0.0970,
        0.0963,
        0.0997,
        0.0980
    ],

    # --------------------------------------------------------
    # CHART 2
    # --------------------------------------------------------

    "Specificity": [
        0.5544,
        0.7151,
        0.7193,
        0.7151,
        0.5882,
        0.7036,
        0.6560,
        0.9698
    ],

    "MCC": [
        0.0572,
        0.0370,
        0.0347,
        0.0370,
        0.0308,
        0.0445,
        0.0454,
        0.0259
    ],

    "Cohen's Kappa": [
        0.0339,
        0.0284,
        0.0268,
        0.0284,
        0.0193,
        0.0335,
        0.0317,
        0.0230
    ],

    "Log Loss": [
        0.6992,
        0.7258,
        0.6889,
        0.7190,
        0.6946,
        0.6989,
        0.6790,
        0.5567
    ]
})


# ============================================================
# CHART 1
# CLASSIFICATION PERFORMANCE
# ============================================================

st.markdown(
    """
    <div class="section-title">
        Classification Performance
    </div>

    <div class="section-subtitle">
        Accuracy, Precision, Sensitivity, F1 Score, ROC-AUC and PR-AUC
        across all eight evaluated models.
    </div>
    """,
    unsafe_allow_html=True
)


fig_classification = go.Figure()


classification_colors = {

    "Accuracy": BLUE,

    "Precision": PURPLE,

    "Sensitivity": CYAN,

    "F1 Score": PINK,

    "ROC-AUC": ORANGE,

    "PR-AUC": INDIGO
}


for metric, color in classification_colors.items():

    fig_classification.add_trace(

        go.Bar(

            name=metric,

            x=models["Model"],

            y=models[metric],

            marker=dict(

                color=color,

                line=dict(

                    color="white",

                    width=1

                )

            ),

            text=[
                f"{value:.3f}"
                for value in models[metric]
            ],

            textposition="outside",

            textfont=dict(

                size=9,

                color=TEXT

            ),

            hovertemplate=(

                "<b>%{x}</b><br>"
                + metric
                + ": %{y:.4f}"
                + "<extra></extra>"

            )
        )
    )


fig_classification.update_layout(

    barmode="group",

    height=680,

    paper_bgcolor=WHITE,

    plot_bgcolor=WHITE,

    title=dict(

        text=(

            "<b>GARUDA Hybrid Multi-Signal Crime Risk "
            "Verification Model</b>"
            "<br>"
            "<sup>8-Model Comparative Analysis — Classification Performance</sup>"

        ),

        font=dict(

            size=21,

            color=NAVY

        ),

        x=0.02,

        xanchor="left"
    ),

    xaxis=dict(

        title=dict(

            text="<b>Machine Learning Models</b>",

            font=dict(

                size=14,

                color=NAVY
            )
        ),

        tickangle=-20,

        tickfont=dict(

            size=10,

            color=TEXT
        ),

        gridcolor=GRID,

        showline=True,

        linecolor="#CBD5E1"
    ),

    yaxis=dict(

        title=dict(

            text="<b>Performance Score</b>",

            font=dict(

                size=14,

                color=NAVY
            )
        ),

        range=[0, 1.0],

        tickfont=dict(

            size=11,

            color=TEXT
        ),

        gridcolor=GRID,

        showline=True,

        linecolor="#CBD5E1"
    ),

    legend=dict(

        orientation="h",

        y=1.08,

        x=0,

        font=dict(

            size=11,

            color=TEXT
        ),

        bgcolor="rgba(255,255,255,0)"
    ),

    margin=dict(

        l=60,

        r=30,

        t=125,

        b=135
    ),

    hovermode="x unified"
)


st.plotly_chart(

    fig_classification,

    use_container_width=True
)


# ============================================================
# CHART 2
# ADDITIONAL EVALUATION METRICS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        Additional Evaluation Metrics
    </div>

    <div class="section-subtitle">
        Specificity, Matthews Correlation Coefficient, Cohen's Kappa
        and Log Loss for the same chronological test cohort.
    </div>
    """,
    unsafe_allow_html=True
)


fig_additional = go.Figure()


additional_colors = {

    "Specificity": BLUE,

    "MCC": PURPLE,

    "Cohen's Kappa": CYAN,

    "Log Loss": ORANGE
}


for metric, color in additional_colors.items():

    fig_additional.add_trace(

        go.Bar(

            name=metric,

            x=models["Model"],

            y=models[metric],

            marker=dict(

                color=color,

                line=dict(

                    color="white",

                    width=1
                )
            ),

            text=[

                f"{value:.3f}"

                for value in models[metric]

            ],

            textposition="outside",

            textfont=dict(

                size=9,

                color=TEXT
            ),

            hovertemplate=(

                "<b>%{x}</b><br>"
                + metric
                + ": %{y:.4f}"
                + "<extra></extra>"

            )
        )
    )


fig_additional.update_layout(

    barmode="group",

    height=620,

    paper_bgcolor=WHITE,

    plot_bgcolor=WHITE,

    title=dict(

        text=(

            "<b>GARUDA Hybrid — Additional Evaluation Metrics</b>"
            "<br>"
            "<sup>Specificity, MCC, Cohen's Kappa and Log Loss</sup>"

        ),

        font=dict(

            size=21,

            color=NAVY
        ),

        x=0.02,

        xanchor="left"
    ),

    xaxis=dict(

        title=dict(

            text="<b>Machine Learning Models</b>",

            font=dict(

                size=14,

                color=NAVY
            )
        ),

        tickangle=-20,

        tickfont=dict(

            size=10,

            color=TEXT
        ),

        gridcolor=GRID,

        showline=True,

        linecolor="#CBD5E1"
    ),

    yaxis=dict(

        title=dict(

            text="<b>Metric Value</b>",

            font=dict(

                size=14,

                color=NAVY
            )
        ),

        range=[0, 1.0],

        tickfont=dict(

            size=11,

            color=TEXT
        ),

        gridcolor=GRID,

        showline=True,

        linecolor="#CBD5E1"
    ),

    legend=dict(

        orientation="h",

        y=1.08,

        x=0,

        font=dict(

            size=11,

            color=TEXT
        ),

        bgcolor="rgba(255,255,255,0)"
    ),

    margin=dict(

        l=60,

        r=30,

        t=125,

        b=135
    ),

    hovermode="x unified"
)


st.plotly_chart(

    fig_additional,

    use_container_width=True
)


# ============================================================
# EVALUATION SETUP NOTE
# ============================================================

st.markdown(
    """
    <div style="
        background:#F8FAFC;
        border:1px solid #E2E8F0;
        border-radius:12px;
        padding:14px 18px;
        margin-top:8px;
        margin-bottom:20px;
        color:#64748B;
        font-size:12px;
        line-height:1.6;
    ">

        <b style="color:#172554;">
            Evaluation setup:
        </b>

        Common chronological evaluation cohort with

        <b>17,755 training rows</b>

        and

        <b>4,439 testing rows</b>.

        GARUDA Hybrid is the proposed model and the other seven
        models are comparison baselines.

        <br><br>

        <b style="color:#172554;">
            Note:
        </b>

        Log Loss is an error metric where lower values indicate
        lower probabilistic prediction error. The displayed values
        are the actual results from the completed backend experiment.

    </div>
    """,
    unsafe_allow_html=True
)


# SECTION 3
# ABLATION STUDY
# ============================================================

st.markdown(
    """
    <div class="section-title">
        E1–E4 Ablation Analysis
    </div>

    <div class="section-subtitle">
        Progressive addition of temporal, anomaly and spatial signals using the completed backend experiment
        to the ML prediction layer.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ABLATION DATA
# ============================================================

ablation = pd.DataFrame({

    "Experiment": [

        "E1 — ML Only",

        "E2 — ML + Temporal",

        "E3 — ML + Temporal + Anomaly",

        "E4 — ML + Temporal + Anomaly + Spatial"

    ],

    "F1 Score": [

        0.1582,

        0.1634,

        0.1557,

        0.1554

    ],

    "ROC-AUC": [

        0.5532,

        0.5498,

        0.5508,

        0.5505

    ],

    "PR-AUC": [

        0.0963,

        0.0955,

        0.0969,

        0.0970

    ],

    "Recall": [

        0.3709,

        0.5522,

        0.4368,

        0.4368

    ]

})


# ============================================================
# ABLATION GRAPH
# ============================================================

fig_ablation = go.Figure()


ablation_colors = {

    "F1 Score": PINK,

    "ROC-AUC": ORANGE,

    "PR-AUC": INDIGO,

    "Recall": CYAN

}


for metric, color in ablation_colors.items():

    fig_ablation.add_trace(

        go.Bar(

            name=metric,

            x=ablation["Experiment"],

            y=ablation[metric],

            marker_color=color,

            text=[

                f"{value:.3f}"

                for value in ablation[metric]

            ],

            textposition="outside",

            textfont=dict(

                size=11,

                color=TEXT

            )

        )

    )


fig_ablation.update_layout(

    barmode="group",

    height=600,

    paper_bgcolor=WHITE,

    plot_bgcolor=WHITE,

    title=dict(

        text=(
            "<b>Progressive Hybrid Signal Ablation</b>"
        ),

        font=dict(

            size=21,

            color=NAVY

        ),

        x=0.02,

        xanchor="left"

    ),

    xaxis=dict(

        title="<b>Experiment</b>",

        tickfont=dict(

            size=11,

            color=TEXT

        ),

        gridcolor=GRID

    ),

    yaxis=dict(

        title="<b>Score</b>",

        range=[0, 0.28],

        gridcolor=GRID,

        tickfont=dict(

            size=11,

            color=TEXT

        )

    ),

    legend=dict(

        orientation="h",

        y=1.08,

        x=0

    ),

    margin=dict(

        l=60,

        r=30,

        t=100,

        b=110

    )

)


st.plotly_chart(

    fig_ablation,

    use_container_width=True

)


# ============================================================
# SECTION 4
# CONSENSUS SCORE
# ============================================================

st.markdown(
    """
    <div class="section-title">
        GARUDA Consensus Risk Score
    </div>

    <div class="section-subtitle">
        Experimental weighting of heterogeneous evidence sources.
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# CONSENSUS WEIGHTS
# ============================================================

consensus = pd.DataFrame({

    "Signal": [

        "ML Risk Probability",

        "Temporal Consistency",

        "Anomaly Signal",

        "Spatial Signal"

    ],

    "Weight": [

        0.40,

        0.20,

        0.20,

        0.20

    ]

})


# ============================================================
# CONSENSUS FORMULA GRAPH
# ============================================================

fig_weights = go.Figure()


fig_weights.add_trace(

    go.Bar(

        x=consensus["Signal"],

        y=consensus["Weight"],

        marker_color=[

            PURPLE,

            CYAN,

            PINK,

            ORANGE

        ],

        text=[

            "40%",

            "20%",

            "20%",

            "20%"

        ],

        textposition="outside",

        textfont=dict(

            size=14,

            color=NAVY

        )

    )

)


fig_weights.update_layout(

    height=500,

    paper_bgcolor=WHITE,

    plot_bgcolor=WHITE,

    title=dict(

        text=(
            "<b>GARUDA Consensus Risk Score — "
            "Experimental Signal Weights</b>"
        ),

        font=dict(

            size=20,

            color=NAVY

        ),

        x=0.02,

        xanchor="left"

    ),

    xaxis=dict(

        title="<b>Evidence Signal</b>",

        tickfont=dict(

            size=11,

            color=TEXT

        ),

        gridcolor=GRID

    ),

    yaxis=dict(

        title="<b>Weight</b>",

        range=[0, 0.48],

        tickformat=".0%",

        gridcolor=GRID,

        tickfont=dict(

            size=11,

            color=TEXT

        )

    ),

    margin=dict(

        l=60,

        r=30,

        t=100,

        b=100

    )

)


st.plotly_chart(

    fig_weights,

    use_container_width=True

)


# ============================================================
# CONSENSUS FORMULA
# ============================================================

st.markdown(
    """
    <div style="
        background:#FFFFFF;
        border:1px solid #E2E8F0;
        border-radius:16px;
        padding:20px;
        margin-top:10px;
        margin-bottom:20px;
        text-align:center;
        box-shadow:0 8px 25px rgba(15,23,42,0.05);
    ">

    <div style="
        font-size:14px;
        color:#64748B;
        margin-bottom:10px;
        font-weight:700;
    ">
        CONSENSUS RISK FORMULA
    </div>

    <div style="
        font-size:20px;
        color:#172554;
        font-weight:900;
    ">
        CRS = 0.40 × ML + 0.20 × Temporal
        + 0.20 × Anomaly + 0.20 × Spatial
    </div>

    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SECTION 5
# RISK LEVELS
# ============================================================

st.markdown(
    """
    <div class="section-title">
        Consensus Risk Interpretation
    </div>
    """,
    unsafe_allow_html=True
)


risk = pd.DataFrame({

    "Risk Level": [

        "LOW",

        "MEDIUM",

        "HIGH"

    ],

    "Lower Bound": [

        0.00,

        0.33,

        0.66

    ],

    "Upper Bound": [

        0.33,

        0.66,

        1.00

    ]

})


fig_risk = go.Figure()


fig_risk.add_trace(

    go.Bar(

        x=risk["Risk Level"],

        y=[

            0.33,

            0.33,

            0.34

        ],

        marker_color=[

            BLUE,

            ORANGE,

            RED

        ],

        text=[

            "0.00 – < 0.33",

            "0.33 – < 0.66",

            "0.66 – 1.00"

        ],

        textposition="inside",

        textfont=dict(

            color="white",

            size=14

        ),

        hovertemplate=(

            "<b>%{x}</b><br>"

            "Risk range: %{text}"

            "<extra></extra>"

        )

    )

)


fig_risk.update_layout(

    height=380,

    paper_bgcolor=WHITE,

    plot_bgcolor=WHITE,

    title=dict(

        text="<b>Consensus Risk Classification Thresholds</b>",

        font=dict(

            size=20,

            color=NAVY

        ),

        x=0.02,

        xanchor="left"

    ),

    xaxis=dict(

        title="<b>Risk Level</b>",

        gridcolor=GRID

    ),

    yaxis=dict(

        title="<b>Relative Range</b>",

        range=[0, 0.40],

        showticklabels=False,

        gridcolor=GRID

    ),

    showlegend=False,

    margin=dict(

        l=60,

        r=30,

        t=90,

        b=80

    )

)


st.plotly_chart(

    fig_risk,

    use_container_width=True

)


# ============================================================
# FOOTNOTE
# ============================================================

st.markdown(
    """
    <div style="
        margin-top:20px;
        padding:14px 18px;
        border-radius:12px;
        background:#F8FAFC;
        border:1px solid #E2E8F0;
        color:#64748B;
        font-size:11px;
        line-height:1.6;
    ">
        <b style="color:#172554;">Research note:</b>
        The GARUDA hybrid model integrates ML prediction with temporal,
        anomaly and spatial evidence. The displayed weights and risk
        thresholds are experimental configurations and should be validated
        through further experiments and ablation analysis.
    </div>
    """,
    unsafe_allow_html=True
)