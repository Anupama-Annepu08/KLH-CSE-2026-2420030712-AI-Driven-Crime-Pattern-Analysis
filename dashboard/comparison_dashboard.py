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

    ("7 ML Models", 0.25, 0.50, BLUE),

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
# 7 MODEL COMPARISON
# ============================================================

st.markdown(
    """
    <div class="section-title">
        GARUDA Hybrid Multi-Signal Crime Risk Verification Model
    </div>

    <div class="section-subtitle">
        7-Model Comparative Analysis — Accuracy, Precision, Recall,
        F1 Score, ROC-AUC and PR-AUC
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# ACTUAL E4 MODEL RESULTS
# ============================================================

models = pd.DataFrame({

    "Model": [

        "Logistic Regression",

        "Decision Tree",

        "Random Forest",

        "Extra Trees",

        "HistGradientBoosting",

        "XGBoost",

        "MLP Neural Network (Proposed Hybrid Model)"

    ],

    "Accuracy": [

        0.581795,

        0.572778,

        0.580936,

        0.581795,

        0.566337,

        0.569343,

        0.582653

    ],

    "Precision": [

        0.478261,

        0.437500,

        0.491525,

        0.494382,

        0.439739,

        0.452229,

        0.520000

    ],

    "Recall": [

        0.011305,

        0.079137,

        0.089414,

        0.045221,

        0.138746,

        0.145940,

        0.013361

    ],

    "F1 Score": [

        0.022088,

        0.134030,

        0.151304,

        0.082863,

        0.210938,

        0.220668,

        0.026052

    ],

    "ROC-AUC": [

        0.544666,

        0.510557,

        0.524096,

        0.529097,

        0.518899,

        0.520140,

        0.530738

    ],

    "PR-AUC": [

        0.461565,

        0.427421,

        0.443322,

        0.447726,

        0.434471,

        0.438243,

        0.441966

    ]

})


# ============================================================
# 7 MODEL GRAPH
# ============================================================

fig_models = go.Figure()


metric_colors = {

    "Accuracy": BLUE,

    "Precision": PURPLE,

    "Recall": CYAN,

    "F1 Score": PINK,

    "ROC-AUC": ORANGE,

    "PR-AUC": INDIGO

}


for metric, color in metric_colors.items():

    fig_models.add_trace(

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

                size=10,

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


fig_models.update_layout(

    barmode="group",

    height=680,

    paper_bgcolor=WHITE,

    plot_bgcolor=WHITE,

    title=dict(

        text=(

            "<b>GARUDA Hybrid Multi-Signal Crime Risk "
            "Verification Model</b>"
            "<br>"
            "<sup>7-Model Comparative Analysis</sup>"
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

            size=11,

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

        range=[0, 0.67],

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

        b=125

    ),

    hovermode="x unified"

)


st.plotly_chart(

    fig_models,

    use_container_width=True

)


# ============================================================
# SECTION 3
# ABLATION STUDY
# ============================================================

st.markdown(
    """
    <div class="section-title">
        E1–E4 Ablation Analysis
    </div>

    <div class="section-subtitle">
        Progressive addition of temporal, anomaly and spatial signals
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

        0.144803,

        0.200135,

        0.186903,

        0.220668

    ],

    "ROC-AUC": [

        0.515611,

        0.531877,

        0.531088,

        0.520140

    ],

    "PR-AUC": [

        0.427106,

        0.442926,

        0.436773,

        0.438243

    ],

    "Recall": [

        0.087522,

        0.128250,

        0.118718,

        0.145940

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