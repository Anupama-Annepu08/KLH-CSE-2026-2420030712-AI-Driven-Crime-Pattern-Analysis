from pathlib import Path
import sys
import json
import warnings

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest
)
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    matthews_corrcoef,
    cohen_kappa_score,
    log_loss
)

from xgboost import XGBClassifier


warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "chicago_crime_2025_2026.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))


from feature_engineering import (
    create_temporal_features,
    create_crime_features,
    create_historical_features
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

TRAIN_RATIO = 0.80

SPATIAL_EPS = 0.020

SPATIAL_MIN_SAMPLES = 5

ANOMALY_CONTAMINATION = 0.20


# ============================================================
# FEATURES
# ============================================================

BASE_FEATURES = [
    "daily_crime_count",
    "violent_crime_count",
    "crime_count_7d",
    "crime_count_30d"
]

GARUDA_FEATURES = BASE_FEATURES + [
    "temporal_consistency",
    "anomaly_score",
    "spatial_signal"
]

SPATIAL_FEATURES = [
    "avg_latitude",
    "avg_longitude"
]

ANOMALY_FEATURES = [
    "daily_crime_count",
    "violent_crime_count",
    "crime_count_7d",
    "crime_count_30d"
]


# ============================================================
# XGBOOST
# ============================================================

def create_xgboost_model(scale_pos_weight=1.0):

    return XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )


# ============================================================
# LOAD AND ENGINEER DATA
# ============================================================

def load_and_engineer_data():

    print("\n")
    print("=" * 90)
    print("GARUDA AI — 8-MODEL COMPARISON EXPERIMENT")
    print("=" * 90)

    print("\nLoading Chicago crime dataset...")

    df = pd.read_csv(DATA_PATH)

    print(f"Original dataset shape: {df.shape}")

    # --------------------------------------------------------
    # Beat-level spatial reference
    # --------------------------------------------------------
    # IMPORTANT:
    # Keep the geographic information from the RAW Chicago data.
    # The previous version could lose the latitude/longitude mapping
    # after feature engineering, which caused the spatial component
    # to receive all-NaN coordinates.
    # --------------------------------------------------------

    spatial_source = None

    required_spatial_columns = [
        "beat",
        "latitude",
        "longitude"
    ]

    if all(
        column in df.columns
        for column in required_spatial_columns
    ):

        raw_spatial = df[
            required_spatial_columns
        ].copy()

        raw_spatial["beat_key"] = (
            raw_spatial["beat"]
            .astype("string")
            .str.strip()
        )

        raw_spatial["latitude"] = pd.to_numeric(
            raw_spatial["latitude"],
            errors="coerce"
        )

        raw_spatial["longitude"] = pd.to_numeric(
            raw_spatial["longitude"],
            errors="coerce"
        )

        spatial_source = (
            raw_spatial
            .dropna(
                subset=[
                    "beat_key",
                    "latitude",
                    "longitude"
                ]
            )
            .groupby("beat_key")[[
                "latitude",
                "longitude"
            ]]
            .median()
            .reset_index()
            .rename(
                columns={
                    "latitude": "avg_latitude",
                    "longitude": "avg_longitude"
                }
            )
        )

        print(
            f"Beat-level spatial references: "
            f"{len(spatial_source):,}"
        )

        if len(spatial_source) < 2:
            raise ValueError(
                "The raw dataset does not contain at least two "
                "distinct beat-level coordinate references."
            )

    else:

        raise ValueError(
            "The Chicago crime dataset must contain beat, latitude "
            "and longitude columns for the GARUDA spatial component."
        )

    # --------------------------------------------------------
    # Feature engineering
    # --------------------------------------------------------

    df = create_temporal_features(
        df,
        "date"
    )

    df = create_crime_features(
        df
    )

    df = create_historical_features(
        df
    )

    # --------------------------------------------------------
    # Attach beat-level coordinates ROBUSTLY
    # --------------------------------------------------------

    if "beat" not in df.columns:
        raise ValueError(
            "Feature engineering removed the required 'beat' column."
        )

    df["__beat_key"] = (
        df["beat"]
        .astype("string")
        .str.strip()
    )

    # Some feature-engineering versions may already create columns with
    # these names. Remove them before attaching the authoritative raw-data
    # beat reference so pandas cannot silently create _x/_y columns.
    df.drop(
        columns=["avg_latitude", "avg_longitude", "beat_key"],
        inplace=True,
        errors="ignore"
    )

    spatial_lookup = spatial_source.set_index("beat_key")[[
        "avg_latitude",
        "avg_longitude"
    ]]

    df["avg_latitude"] = (
        df["__beat_key"]
        .map(spatial_lookup["avg_latitude"])
    )

    df["avg_longitude"] = (
        df["__beat_key"]
        .map(spatial_lookup["avg_longitude"])
    )

    df.drop(
        columns=["__beat_key"],
        inplace=True,
        errors="ignore"
    )

    coordinate_count = int(
        df[["avg_latitude", "avg_longitude"]]
        .notna()
        .all(axis=1)
        .sum()
    )

    print(
        "Spatial attachment diagnostics: "
        f"{coordinate_count:,}/{len(df):,} rows have valid coordinates"
    )

    if coordinate_count == 0:
        raise ValueError(
            "Beat-level coordinates were not attached to the engineered "
            "dataset. Check the beat values in the raw dataset."
        )

    print(
        f"Engineered dataset shape: {df.shape}"
    )

    return df


# ============================================================
# COMMON EVALUATION COHORT
# ============================================================

def prepare_common_cohort(df):

    required_columns = [
        "crime_date",
        "beat",

        "daily_crime_count",
        "violent_crime_count",

        "crime_count_7d",
        "crime_count_30d",

        "temporal_consistency",

        "next_day_crime_count",
        "high_risk"
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    # Keep the spatial columns in the SAME dataframe operation as the
    # prediction features. This avoids index-alignment problems after
    # feature engineering/sorting.
    selected_columns = required_columns + [
        column
        for column in SPATIAL_FEATURES
        if column in df.columns
    ]

    data = df[selected_columns].copy()

    numeric_columns = [
        "daily_crime_count",
        "violent_crime_count",
        "crime_count_7d",
        "crime_count_30d",
        "temporal_consistency",
        "next_day_crime_count",
        "high_risk"
    ] + [
        column
        for column in SPATIAL_FEATURES
        if column in data.columns
    ]

    for column in numeric_columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data["crime_date"] = pd.to_datetime(
        data["crime_date"],
        errors="coerce"
    )

    # The spatial component is required for this experiment.
    missing_spatial = [
        column
        for column in SPATIAL_FEATURES
        if column not in data.columns
    ]

    if missing_spatial:
        raise ValueError(
            "Required spatial columns are missing after feature "
            f"engineering: {missing_spatial}"
        )

    before = len(data)

    data = data.dropna(
        subset=[
            "crime_date",
            "daily_crime_count",
            "violent_crime_count",
            "crime_count_7d",
            "crime_count_30d",
            "temporal_consistency",
            "next_day_crime_count",
            "high_risk"
        ]
    ).copy()

    data["high_risk"] = (
        data["high_risk"]
        .astype(int)
    )

    data = (
        data
        .sort_values(
            [
                "crime_date",
                "beat"
            ]
        )
        .reset_index(drop=True)
    )

    print("\n")
    print("=" * 90)
    print("COMMON EVALUATION COHORT")
    print("=" * 90)

    print(
        f"Rows before filtering : {before:,}"
    )

    print(
        f"Rows after filtering  : {len(data):,}"
    )

    print(
        f"Rows removed           : "
        f"{before - len(data):,}"
    )

    print(
        "\nOverall target distribution:"
    )

    print(
        data["high_risk"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    # --------------------------------------------------------
    # Chronological split
    # --------------------------------------------------------

    split_index = int(
        len(data) * TRAIN_RATIO
    )

    train_data = data.iloc[:split_index].copy()
    test_data = data.iloc[split_index:].copy()

    print(
        f"\nTraining rows: {len(train_data):,}"
    )

    print(
        f"Testing rows : {len(test_data):,}"
    )

    print("\nTraining period:")
    print(
        train_data["crime_date"].min(),
        "→",
        train_data["crime_date"].max()
    )

    print("\nTesting period:")
    print(
        test_data["crime_date"].min(),
        "→",
        test_data["crime_date"].max()
    )

    print("\nTraining target:")
    print(
        train_data["high_risk"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    print("\nTesting target:")
    print(
        test_data["high_risk"]
        .value_counts()
        .sort_index()
        .to_dict()
    )

    if train_data["high_risk"].nunique() < 2:
        raise ValueError(
            "Training data contains only one class."
        )

    if test_data["high_risk"].nunique() < 2:
        raise ValueError(
            "Testing data contains only one class."
        )

    print(
        "\n✓ Chronological evaluation cohort validated."
    )

    return train_data, test_data


# ============================================================
# SPATIAL PREPARATION
# ============================================================

def prepare_spatial_coordinates(
    train_data,
    test_data
):

    train_data = train_data.copy()
    test_data = test_data.copy()

    for column in SPATIAL_FEATURES:

        train_data[column] = pd.to_numeric(
            train_data[column],
            errors="coerce"
        )

        test_data[column] = pd.to_numeric(
            test_data[column],
            errors="coerce"
        )

    # Learn beat-level spatial values from TRAINING data only.
    centroid_source = (
        train_data
        .dropna(subset=SPATIAL_FEATURES)
        .groupby("beat")[SPATIAL_FEATURES]
        .median()
    )

    for column in SPATIAL_FEATURES:

        train_data[column] = (
            train_data[column]
            .fillna(
                train_data["beat"]
                .map(centroid_source[column])
            )
        )

        test_data[column] = (
            test_data[column]
            .fillna(
                test_data["beat"]
                .map(centroid_source[column])
            )
        )

    # Training global median fallback.
    for column in SPATIAL_FEATURES:

        global_median = (
            train_data[column]
            .median()
        )

        if pd.isna(global_median):

            global_median = 0.0

        train_data[column] = (
            train_data[column]
            .fillna(global_median)
        )

        test_data[column] = (
            test_data[column]
            .fillna(global_median)
        )

    if (
        train_data[SPATIAL_FEATURES]
        .isna()
        .sum()
        .sum()
        > 0
    ):

        raise ValueError(
            "Training spatial coordinates contain missing values."
        )

    if (
        test_data[SPATIAL_FEATURES]
        .isna()
        .sum()
        .sum()
        > 0
    ):

        raise ValueError(
            "Testing spatial coordinates contain missing values."
        )

    return train_data, test_data


# ============================================================
# ISOLATION FOREST
# ============================================================

def add_anomaly_signal(
    train_data,
    test_data
):

    train_data = train_data.copy()
    test_data = test_data.copy()

    model = IsolationForest(
        n_estimators=200,
        contamination=ANOMALY_CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    X_train = train_data[
        ANOMALY_FEATURES
    ]

    X_test = test_data[
        ANOMALY_FEATURES
    ]

    model.fit(
        X_train
    )

    train_data["anomaly_score"] = (
        -model.score_samples(
            X_train
        )
    )

    test_data["anomaly_score"] = (
        -model.score_samples(
            X_test
        )
    )

    train_data["anomaly_signal"] = (
        model.predict(
            X_train
        ) == -1
    ).astype(int)

    test_data["anomaly_signal"] = (
        model.predict(
            X_test
        ) == -1
    ).astype(int)

    return (
        train_data,
        test_data,
        model
    )


# ============================================================
# DBSCAN + LOCAL SPATIAL DENSITY
# ============================================================

def add_spatial_signal(
    train_data,
    test_data
):
    """
    Construct spatial evidence from unique training beat locations.

    The spatial component is intentionally independent of the target.
    It uses only geographic coordinates available in the training set:

      1. One centroid is retained for each unique training beat.
      2. DBSCAN describes the spatial clustering structure.
      3. Mean distance to the k nearest OTHER training beats measures
         local geographic density.
      4. The density is converted to a percentile/rank score using only
         the training beats. Smaller neighbour distance => higher score.
      5. Known test beats inherit their training-derived spatial score.
         An unseen test beat is assigned the score of its nearest training
         beat using its coordinate only; no target information is used.

    A rank transformation is used instead of a p05-p95 min-max transform.
    This avoids the previous failure mode where a highly concentrated
    density distribution collapsed to the same clipped value.
    """

    train_data = train_data.copy()
    test_data = test_data.copy()

    # --------------------------------------------------------
    # 1. Unique training beat centroids
    # --------------------------------------------------------
    train_beats = (
        train_data[["beat"] + SPATIAL_FEATURES]
        .copy()
    )

    train_beats[SPATIAL_FEATURES] = (
        train_beats[SPATIAL_FEATURES]
        .apply(pd.to_numeric, errors="coerce")
    )

    train_beats = (
        train_beats
        .replace([np.inf, -np.inf], np.nan)
        .dropna(subset=SPATIAL_FEATURES)
        .groupby("beat", as_index=False)[SPATIAL_FEATURES]
        .median()
    )

    if len(train_beats) < max(2, SPATIAL_MIN_SAMPLES):
        raise ValueError(
            "Insufficient unique training beats for spatial evidence: "
            f"{len(train_beats)}."
        )

    train_coordinates = train_beats[
        SPATIAL_FEATURES
    ].to_numpy(dtype=float)

    if not np.isfinite(train_coordinates).all():
        raise ValueError(
            "Training beat coordinates contain non-finite values."
        )

    # A geographic sanity check. If every beat has exactly the same
    # coordinate, there is genuinely no spatial information to use.
    if np.unique(train_coordinates, axis=0).shape[0] < 2:
        raise ValueError(
            "All training beats have identical coordinates. "
            "A spatial evidence component cannot be estimated from this data."
        )

    # --------------------------------------------------------
    # 2. DBSCAN spatial structure
    # --------------------------------------------------------
    dbscan = DBSCAN(
        eps=SPATIAL_EPS,
        min_samples=SPATIAL_MIN_SAMPLES
    )
    dbscan.fit(train_coordinates)

    train_labels = np.asarray(dbscan.labels_)
    non_noise = train_labels[train_labels != -1]
    clusters = (
        len(np.unique(non_noise))
        if len(non_noise) > 0
        else 0
    )
    noise_rate = float(np.mean(train_labels == -1))

    # --------------------------------------------------------
    # 3. Local geographic density on UNIQUE beats
    # --------------------------------------------------------
    k = min(10, len(train_coordinates) - 1)

    neighbors = NearestNeighbors(n_neighbors=k + 1)
    neighbors.fit(train_coordinates)

    distances, _ = neighbors.kneighbors(train_coordinates)
    distances = np.asarray(distances, dtype=float)

    # Column 0 is the beat itself and therefore has distance zero.
    mean_neighbor_distance = np.mean(
        distances[:, 1:],
        axis=1
    )

    if not np.isfinite(mean_neighbor_distance).all():
        finite = mean_neighbor_distance[np.isfinite(mean_neighbor_distance)]
        if finite.size == 0:
            raise ValueError(
                "Unable to calculate finite geographic neighbour distances."
            )
        replacement = float(np.median(finite))
        mean_neighbor_distance = np.nan_to_num(
            mean_neighbor_distance,
            nan=replacement,
            posinf=replacement,
            neginf=replacement
        )

    # If all unique beats are geometrically identical, there is no spatial
    # variation. Do not manufacture a signal in that case.
    if np.ptp(mean_neighbor_distance) <= 1e-12:
        raise ValueError(
            "Geographic neighbour distances are constant across all training "
            "beats. No meaningful spatial variation is available."
        )

    # --------------------------------------------------------
    # 4. Training-only percentile/rank spatial evidence
    # --------------------------------------------------------
    # Smaller neighbour distance means denser local geography and therefore
    # higher spatial evidence. Rank is robust to extreme coordinate spacing.
    density_percentile = pd.Series(
        mean_neighbor_distance
    ).rank(
        method="average",
        pct=True
    ).to_numpy(dtype=float)

    density_signal = 1.0 - density_percentile

    # DBSCAN spatial structure:
    # non-noise beats belong to a detected geographic cluster,
    # while DBSCAN noise points receive zero cluster support.
    cluster_signal = (
        train_labels != -1
    ).astype(float)

    # Combine continuous geographic density with DBSCAN cluster support.
    # This is an experimental spatial-evidence formulation.
    beat_signal = (
        0.50 * density_signal
        + 0.50 * cluster_signal
    )

    # Keep the range inside [0, 1].
    beat_signal = np.clip(
        np.nan_to_num(
            beat_signal,
            nan=0.0,
            posinf=1.0,
            neginf=0.0
        ),
        0.0,
        1.0
    )

    if np.unique(beat_signal).size <= 1:
        raise ValueError(
            "Spatial evidence remained constant after rank transformation."
        )

    train_beats["dbscan_label"] = train_labels
    train_beats["mean_neighbor_distance"] = mean_neighbor_distance
    train_beats["spatial_signal"] = beat_signal

    beat_signal_map = dict(
        zip(
            train_beats["beat"].tolist(),
            train_beats["spatial_signal"].tolist()
        )
    )

    default_signal = float(np.median(beat_signal))

    # --------------------------------------------------------
    # 5. Training observations
    # --------------------------------------------------------
    train_data["spatial_signal"] = (
        train_data["beat"]
        .map(beat_signal_map)
        .fillna(default_signal)
        .astype(float)
    )

    # --------------------------------------------------------
    # 6. Test observations
    # --------------------------------------------------------
    test_known_signal = test_data["beat"].map(beat_signal_map)
    unseen_mask = test_known_signal.isna()
    unseen_count = int(unseen_mask.sum())

    if unseen_mask.any():
        test_coordinates = (
            test_data.loc[unseen_mask, SPATIAL_FEATURES]
            .apply(pd.to_numeric, errors="coerce")
            .to_numpy(dtype=float)
        )

        coordinate_medians = np.median(train_coordinates, axis=0)
        for j in range(test_coordinates.shape[1]):
            bad = ~np.isfinite(test_coordinates[:, j])
            test_coordinates[bad, j] = coordinate_medians[j]

        nearest = NearestNeighbors(n_neighbors=1)
        nearest.fit(train_coordinates)
        _, nearest_index = nearest.kneighbors(test_coordinates)
        nearest_index = nearest_index[:, 0]

        test_known_signal.loc[unseen_mask] = beat_signal[nearest_index]

    test_data["spatial_signal"] = (
        test_known_signal
        .fillna(default_signal)
        .astype(float)
    )

    train_signal = np.clip(
        np.nan_to_num(
            train_data["spatial_signal"].to_numpy(dtype=float),
            nan=default_signal,
            posinf=1.0,
            neginf=0.0
        ),
        0.0,
        1.0
    )

    test_signal = np.clip(
        np.nan_to_num(
            test_data["spatial_signal"].to_numpy(dtype=float),
            nan=default_signal,
            posinf=1.0,
            neginf=0.0
        ),
        0.0,
        1.0
    )

    train_data["spatial_signal"] = train_signal
    test_data["spatial_signal"] = test_signal

    unique_test_values = int(pd.Series(test_signal).nunique())

    stats = {
        "unique_training_beats": int(len(train_beats)),
        "clusters": int(clusters),
        "noise_rate": float(noise_rate),
        "cluster_supported_beats": int(np.sum(cluster_signal == 1)),
        "cluster_support_rate": float(np.mean(cluster_signal)),
        "unseen_test_beats": unseen_count,
        "train_signal_min": float(np.min(train_signal)),
        "train_signal_mean": float(np.mean(train_signal)),
        "train_signal_max": float(np.max(train_signal)),
        "test_signal_min": float(np.min(test_signal)),
        "test_signal_mean": float(np.mean(test_signal)),
        "test_signal_max": float(np.max(test_signal)),
        "test_signal_unique_values": unique_test_values,
        "neighbor_distance_min": float(np.min(mean_neighbor_distance)),
        "neighbor_distance_median": float(np.median(mean_neighbor_distance)),
        "neighbor_distance_max": float(np.max(mean_neighbor_distance))
    }

    print("\nSpatial evidence diagnostics:")
    print(f"  Unique training beats : {stats['unique_training_beats']:,}")
    print(f"  DBSCAN clusters       : {stats['clusters']}")
    print(f"  DBSCAN noise rate     : {stats['noise_rate']:.4f}")
    print(
        f"  Cluster-supported beats: {stats['cluster_supported_beats']:,} "
        f"({stats['cluster_support_rate']:.4f})"
    )
    print(f"  Unseen test beats     : {stats['unseen_test_beats']:,}")
    print(
        f"  Train signal          : min={stats['train_signal_min']:.4f}, "
        f"mean={stats['train_signal_mean']:.4f}, "
        f"max={stats['train_signal_max']:.4f}"
    )
    print(
        f"  Test signal           : min={stats['test_signal_min']:.4f}, "
        f"mean={stats['test_signal_mean']:.4f}, "
        f"max={stats['test_signal_max']:.4f}"
    )
    print(f"  Unique test signals   : {unique_test_values}")
    print(
        f"  Neighbour distance   : min={stats['neighbor_distance_min']:.6f}, "
        f"median={stats['neighbor_distance_median']:.6f}, "
        f"max={stats['neighbor_distance_max']:.6f}"
    )

    if unique_test_values <= 1:
        raise ValueError(
            "Spatial evidence is constant across the test set. "
            "The geographic coordinates do not provide usable variation "
            "under the current spatial representation."
        )

    return (
        train_data,
        test_data,
        dbscan,
        stats
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    y_probability
):

    y_probability = np.clip(
        np.asarray(y_probability, dtype=float),
        1e-7,
        1 - 1e-7
    )

    if not np.isfinite(y_probability).all():
        raise ValueError(
            "Model probability contains non-finite values after sanitization."
        )

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    )

    tn, fp, fn, tp = (
        matrix.ravel()
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0.0
    )

    sensitivity = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    metrics = {

        "Accuracy":
            accuracy_score(
                y_true,
                y_pred
            ),

        "Precision":
            precision_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "Sensitivity":
            sensitivity,

        "Specificity":
            specificity,

        "F1 Score":
            f1_score(
                y_true,
                y_pred,
                zero_division=0
            ),

        "ROC-AUC":
            roc_auc_score(
                y_true,
                y_probability
            ),

        "PR-AUC":
            average_precision_score(
                y_true,
                y_probability
            ),

        "MCC":
            matthews_corrcoef(
                y_true,
                y_pred
            ),

        "Cohen's Kappa":
            cohen_kappa_score(
                y_true,
                y_pred
            ),

        "Log Loss":
            log_loss(
                y_true,
                y_probability,
                labels=[0, 1]
            ),

        "True Negatives":
            int(tn),

        "False Positives":
            int(fp),

        "False Negatives":
            int(fn),

        "True Positives":
            int(tp)
    }

    return metrics, matrix


# ============================================================
# CLASS WEIGHTS
# ============================================================

def calculate_balanced_sample_weights(y):

    y = np.asarray(y)

    negative_count = np.sum(y == 0)
    positive_count = np.sum(y == 1)

    total = len(y)

    weight_negative = (
        total / (2 * negative_count)
        if negative_count > 0
        else 1.0
    )

    weight_positive = (
        total / (2 * positive_count)
        if positive_count > 0
        else 1.0
    )

    weights = np.where(
        y == 1,
        weight_positive,
        weight_negative
    )

    return weights


# ============================================================
# TRAIN BASELINE MODEL
# ============================================================

def train_baseline_models(
    X_train,
    y_train
):

    print("\n")
    print("=" * 90)
    print("TRAINING 7 COMPARISON MODELS")
    print("=" * 90)

    sample_weights = (
        calculate_balanced_sample_weights(
            y_train
        )
    )

    models = {}

    # --------------------------------------------------------
    # 1. Logistic Regression
    # --------------------------------------------------------

    models[
        "Logistic Regression"
    ] = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    # --------------------------------------------------------
    # 2. Decision Tree
    # --------------------------------------------------------

    models[
        "Decision Tree"
    ] = DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=None
    )

    # --------------------------------------------------------
    # 3. Random Forest
    # --------------------------------------------------------

    models[
        "Random Forest"
    ] = RandomForestClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    # --------------------------------------------------------
    # 4. Extra Trees
    # --------------------------------------------------------

    models[
        "Extra Trees"
    ] = ExtraTreesClassifier(
        n_estimators=300,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

    # --------------------------------------------------------
    # 5. HistGradientBoosting
    # --------------------------------------------------------

    models[
        "HistGradientBoosting"
    ] = HistGradientBoostingClassifier(
        max_iter=200,
        learning_rate=0.05,
        max_leaf_nodes=31,
        random_state=RANDOM_STATE
    )

    # --------------------------------------------------------
    # 6. XGBoost
    # --------------------------------------------------------

    negative_count = int(
        np.sum(y_train == 0)
    )

    positive_count = int(
        np.sum(y_train == 1)
    )

    scale_pos_weight = (
        negative_count / positive_count
        if positive_count > 0
        else 1.0
    )

    models[
        "XGBoost"
    ] = create_xgboost_model(
        scale_pos_weight=scale_pos_weight
    )

    # --------------------------------------------------------
    # 7. MLP Neural Network
    # --------------------------------------------------------

    models[
        "MLP Neural Network"
    ] = Pipeline(
        [
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.0001,
                    learning_rate_init=0.001,
                    max_iter=500,
                    early_stopping=True,
                    validation_fraction=0.15,
                    n_iter_no_change=20,
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    results = {}

    for name, model in models.items():

        print(
            f"\nTraining: {name}"
        )

        if name in [
            "Random Forest",
            "Extra Trees"
        ]:

            model.fit(
                X_train,
                y_train
            )

        elif name == "XGBoost":

            model.fit(
                X_train,
                y_train
            )

        elif name == "Logistic Regression":

            model.fit(
                X_train,
                y_train,
                model__sample_weight=sample_weights
            )

        elif name == "MLP Neural Network":

            # MLPClassifier does not support sample_weight in the
            # installed scikit-learn API. Use deterministic minority
            # oversampling instead of silently training an unbalanced
            # MLP that predicts only the majority class.
            rng = np.random.RandomState(RANDOM_STATE)
            X_array = np.asarray(X_train)
            y_array = np.asarray(y_train)

            negative_idx = np.flatnonzero(y_array == 0)
            positive_idx = np.flatnonzero(y_array == 1)

            if len(positive_idx) == 0 or len(negative_idx) == 0:
                raise ValueError(
                    "MLP training requires both target classes."
                )

            if len(positive_idx) < len(negative_idx):
                extra_positive = rng.choice(
                    positive_idx,
                    size=len(negative_idx) - len(positive_idx),
                    replace=True
                )
                balanced_idx = np.concatenate(
                    [negative_idx, positive_idx, extra_positive]
                )
            else:
                balanced_idx = np.concatenate(
                    [negative_idx, positive_idx]
                )

            rng.shuffle(balanced_idx)

            model.fit(
                X_array[balanced_idx],
                y_array[balanced_idx]
            )

        else:

            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weights
            )

        results[name] = model

    return results


# ============================================================
# PREDICT BASELINE MODELS
# ============================================================

def evaluate_baseline_models(
    models,
    X_test,
    y_test
):

    results = []

    for name, model in models.items():

        print(
            f"\nEvaluating: {name}"
        )

        y_pred = model.predict(
            X_test
        )

        y_probability = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

        metrics, matrix = (
            calculate_metrics(
                y_test,
                y_pred,
                y_probability
            )
        )

        results.append(
            {
                "Model": name,
                "Type": "Comparison",
                "Features": BASE_FEATURES.copy(),
                "Metrics": metrics,
                "Confusion_Matrix": matrix.tolist()
            }
        )

    return results


# ============================================================
# GARUDA HYBRID
# ============================================================

def run_garuda_hybrid(
    train_data,
    test_data
):

    print("\n")
    print("=" * 90)
    print("GARUDA HYBRID — PROPOSED MODEL")
    print("=" * 90)

    train_garuda = train_data.copy()
    test_garuda = test_data.copy()

    # --------------------------------------------------------
    # 1. Isolation Forest
    # --------------------------------------------------------

    (
        train_garuda,
        test_garuda,
        anomaly_model
    ) = add_anomaly_signal(
        train_garuda,
        test_garuda
    )

    # --------------------------------------------------------
    # 2. Spatial preparation
    # --------------------------------------------------------

    (
        train_garuda,
        test_garuda
    ) = prepare_spatial_coordinates(
        train_garuda,
        test_garuda
    )

    # --------------------------------------------------------
    # 3. DBSCAN + spatial evidence
    # --------------------------------------------------------

    (
        train_garuda,
        test_garuda,
        spatial_model,
        spatial_stats
    ) = add_spatial_signal(
        train_garuda,
        test_garuda
    )

    # --------------------------------------------------------
    # 4. XGBoost ML component
    # --------------------------------------------------------

    X_train = train_garuda[
        BASE_FEATURES
    ]

    X_test = test_garuda[
        BASE_FEATURES
    ]

    y_train = train_garuda[
        "high_risk"
    ]

    y_test = test_garuda[
        "high_risk"
    ]

    negative_count = int(
        np.sum(y_train == 0)
    )

    positive_count = int(
        np.sum(y_train == 1)
    )

    scale_pos_weight = (
        negative_count / positive_count
        if positive_count > 0
        else 1.0
    )

    ml_model = create_xgboost_model(
        scale_pos_weight=scale_pos_weight
    )

    print(
        "\nTraining GARUDA XGBoost component..."
    )

    ml_model.fit(
        X_train,
        y_train
    )

    ml_probability = (
        ml_model.predict_proba(
            X_test
        )[:, 1]
    )

    # --------------------------------------------------------
    # 5. Heterogeneous signals
    # --------------------------------------------------------

    temporal_signal = (
        test_garuda[
            "temporal_consistency"
        ]
        .astype(float)
        .clip(0.0, 1.0)
        .to_numpy()
    )

    anomaly_signal = (
        test_garuda[
            "anomaly_signal"
        ]
        .astype(float)
        .clip(0.0, 1.0)
        .to_numpy()
    )

    spatial_signal = (
        test_garuda[
            "spatial_signal"
        ]
        .astype(float)
        .clip(0.0, 1.0)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Validate and sanitize hybrid signals
    # --------------------------------------------------------
    # The common cohort already removes missing values from the
    # core prediction features. This additional guard protects
    # the consensus layer from any non-finite value generated
    # by a model or numerical spatial-density calculation.
    #
    # For a probability/signal, non-finite values are mapped to
    # the neutral lower-bound value 0.0 before consensus.
    # This keeps ROC-AUC, PR-AUC and Log Loss computable while
    # preserving the defined 0-to-1 signal range.
    ml_probability = np.nan_to_num(
        np.asarray(ml_probability, dtype=float),
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    temporal_signal = np.nan_to_num(
        np.asarray(temporal_signal, dtype=float),
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    anomaly_signal = np.nan_to_num(
        np.asarray(anomaly_signal, dtype=float),
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    spatial_signal = np.nan_to_num(
        np.asarray(spatial_signal, dtype=float),
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    if not (
        np.isfinite(ml_probability).all()
        and np.isfinite(temporal_signal).all()
        and np.isfinite(anomaly_signal).all()
        and np.isfinite(spatial_signal).all()
    ):
        raise ValueError(
            "GARUDA hybrid signals contain non-finite values "
            "after sanitization."
        )

    print("\nGARUDA signal diagnostics:")
    for label, values in [
        ("ML probability", ml_probability),
        ("Temporal signal", temporal_signal),
        ("Anomaly signal", anomaly_signal),
        ("Spatial signal", spatial_signal)
    ]:
        print(
            f"  {label}: min={np.min(values):.4f}, "
            f"mean={np.mean(values):.4f}, "
            f"max={np.max(values):.4f}"
        )

    # --------------------------------------------------------
    # 6. GARUDA consensus layer
    #
    # Experimental weights defined by the project:
    #
    # ML probability = 0.40
    # Temporal       = 0.20
    # Anomaly        = 0.20
    # Spatial        = 0.20
    # --------------------------------------------------------

    consensus_score = (
        0.40 * ml_probability
        + 0.20 * temporal_signal
        + 0.20 * anomaly_signal
        + 0.20 * spatial_signal
    )

    consensus_score = np.clip(
        consensus_score,
        0.0,
        1.0
    )

    print(
        "  Consensus: "
        f"min={np.min(consensus_score):.4f}, "
        f"mean={np.mean(consensus_score):.4f}, "
        f"max={np.max(consensus_score):.4f}, "
        f"p95={np.percentile(consensus_score, 95):.4f}, "
        f">=0.66={int(np.sum(consensus_score >= 0.66))}"
    )

    # The target represents HIGH risk.
    # GARUDA's defined HIGH threshold is >= 0.66.
    garuda_prediction = (
        consensus_score >= 0.66
    ).astype(int)

    metrics, matrix = (
        calculate_metrics(
            y_test,
            garuda_prediction,
            consensus_score
        )
    )

    result = {

        "Model":
            "GARUDA Hybrid",

        "Type":
            "Proposed",

        "Features":
            GARUDA_FEATURES.copy(),

        "Metrics":
            metrics,

        "Confusion_Matrix":
            matrix.tolist(),

        "Configuration": {

            "ml_weight": 0.40,

            "temporal_weight": 0.20,

            "anomaly_weight": 0.20,

            "spatial_weight": 0.20,

            "low_threshold": 0.33,

            "high_threshold": 0.66,

            "scale_pos_weight":
                float(scale_pos_weight),

            "anomaly_contamination":
                ANOMALY_CONTAMINATION,

            "spatial_eps":
                SPATIAL_EPS,

            "spatial_min_samples":
                SPATIAL_MIN_SAMPLES
        },

        "Signal_Summary": {

            "mean_ml_probability":
                float(
                    np.mean(
                        ml_probability
                    )
                ),

            "mean_temporal_signal":
                float(
                    np.mean(
                        temporal_signal
                    )
                ),

            "mean_anomaly_signal":
                float(
                    np.mean(
                        anomaly_signal
                    )
                ),

            "mean_spatial_signal":
                float(
                    np.mean(
                        spatial_signal
                    )
                ),

            "mean_consensus_score":
                float(
                    np.mean(
                        consensus_score
                    )
                ),

            "min_consensus_score":
                float(np.min(consensus_score)),

            "max_consensus_score":
                float(np.max(consensus_score)),

            "p95_consensus_score":
                float(np.percentile(consensus_score, 95)),

            "high_risk_predictions":
                int(np.sum(garuda_prediction == 1)),

            "test_rows":
                int(len(consensus_score))
        }
    }

    return (
        result,
        ml_model,
        anomaly_model,
        spatial_model,
        spatial_stats
    )


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    all_results,
    train_data,
    test_data
):

    rows = []

    for result in all_results:

        row = {
            "Model":
                result["Model"],

            "Type":
                result["Type"]
        }

        row.update(
            result["Metrics"]
        )

        rows.append(
            row
        )

    comparison = pd.DataFrame(
        rows
    )

    metric_columns = [
        "Accuracy",
        "Precision",
        "Sensitivity",
        "Specificity",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC",
        "MCC",
        "Cohen's Kappa",
        "Log Loss"
    ]

    comparison = comparison[
        [
            "Model",
            "Type"
        ]
        + metric_columns
        + [
            "True Negatives",
            "False Positives",
            "False Negatives",
            "True Positives"
        ]
    ]

    csv_path = (
        RESULTS_DIR
        / "garuda_8_model_comparison.csv"
    )

    comparison.to_csv(
        csv_path,
        index=False
    )

    # --------------------------------------------------------
    # JSON
    # --------------------------------------------------------

    json_results = []

    for result in all_results:

        json_results.append(
            {
                "Model":
                    result["Model"],

                "Type":
                    result["Type"],

                "Features":
                    result["Features"],

                "Metrics":
                    result["Metrics"],

                "Confusion_Matrix":
                    result["Confusion_Matrix"],

                "Configuration":
                    result.get(
                        "Configuration",
                        {}
                    ),

                "Signal_Summary":
                    result.get(
                        "Signal_Summary",
                        {}
                    )
            }
        )

    json_path = (
        RESULTS_DIR
        / "garuda_8_model_comparison.json"
    )

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            json_results,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # Confusion matrices
    # --------------------------------------------------------

    confusion_rows = []

    for result in all_results:

        matrix = np.asarray(
            result["Confusion_Matrix"]
        )

        confusion_rows.append(
            {
                "Model":
                    result["Model"],

                "True_Negative":
                    int(matrix[0, 0]),

                "False_Positive":
                    int(matrix[0, 1]),

                "False_Negative":
                    int(matrix[1, 0]),

                "True_Positive":
                    int(matrix[1, 1])
            }
        )

    confusion_df = pd.DataFrame(
        confusion_rows
    )

    confusion_path = (
        RESULTS_DIR
        / "garuda_8_model_confusion_matrices.csv"
    )

    confusion_df.to_csv(
        confusion_path,
        index=False
    )

    # --------------------------------------------------------
    # Dataset information
    # --------------------------------------------------------

    dataset_info = {

        "train_rows":
            int(len(train_data)),

        "test_rows":
            int(len(test_data)),

        "train_start":
            str(
                train_data["crime_date"].min()
            ),

        "train_end":
            str(
                train_data["crime_date"].max()
            ),

        "test_start":
            str(
                test_data["crime_date"].min()
            ),

        "test_end":
            str(
                test_data["crime_date"].max()
            ),

        "train_target_distribution":
            train_data[
                "high_risk"
            ]
            .value_counts()
            .sort_index()
            .to_dict(),

        "test_target_distribution":
            test_data[
                "high_risk"
            ]
            .value_counts()
            .sort_index()
            .to_dict()
    }

    dataset_path = (
        RESULTS_DIR
        / "garuda_8_model_comparison_dataset_info.json"
    )

    with open(
        dataset_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset_info,
            file,
            indent=4,
            default=int
        )

    return (
        comparison,
        csv_path,
        json_path,
        confusion_path
    )


# ============================================================
# CREATE COMPARISON CHART
# ============================================================

def create_metric_chart(
    comparison
):

    metrics = [
        "Accuracy",
        "Precision",
        "Sensitivity",
        "Specificity",
        "F1 Score",
        "ROC-AUC",
        "PR-AUC",
        "MCC",
        "Cohen's Kappa"
    ]

    # --------------------------------------------------------
    # One chart per metric
    # --------------------------------------------------------

    for metric in metrics:

        plt.figure(
            figsize=(12, 6)
        )

        plt.bar(
            comparison["Model"],
            comparison[metric]
        )

        plt.title(
            f"8-Model Comparison — {metric}"
        )

        plt.ylabel(
            metric
        )

        plt.xticks(
            rotation=30,
            ha="right"
        )

        plt.ylim(
            min(
                0,
                comparison[metric].min() - 0.05
            ),
            max(
                1,
                comparison[metric].max() + 0.05
            )
        )

        plt.tight_layout()

        output_path = (
            RESULTS_DIR
            / (
                "comparison_"
                + metric.lower()
                .replace(" ", "_")
                .replace("-", "_")
                .replace("'", "")
                + ".png"
            )
        )

        plt.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close()


# ============================================================
# PRINT FINAL TABLE
# ============================================================

def print_final_table(
    comparison
):

    columns = [

        "Model",

        "Accuracy",
        "Precision",

        "Sensitivity",
        "Specificity",

        "F1 Score",

        "ROC-AUC",
        "PR-AUC",

        "MCC",
        "Cohen's Kappa",

        "Log Loss"
    ]

    print("\n")
    print("=" * 150)
    print("GARUDA AI — FINAL 8-MODEL COMPARISON")
    print("=" * 150)

    print(
        comparison[
            columns
        ].to_string(
            index=False,
            float_format=lambda x:
            f"{x:.4f}"
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load data
    # --------------------------------------------------------

    df = load_and_engineer_data()

    # --------------------------------------------------------
    # 2. Common cohort
    # --------------------------------------------------------

    (
        train_data,
        test_data
    ) = prepare_common_cohort(
        df
    )

    # --------------------------------------------------------
    # 3. Baseline data
    # --------------------------------------------------------

    X_train = train_data[
        BASE_FEATURES
    ]

    y_train = train_data[
        "high_risk"
    ]

    X_test = test_data[
        BASE_FEATURES
    ]

    y_test = test_data[
        "high_risk"
    ]

    # --------------------------------------------------------
    # 4. Seven comparison models
    # --------------------------------------------------------

    models = train_baseline_models(
        X_train,
        y_train
    )

    baseline_results = (
        evaluate_baseline_models(
            models,
            X_test,
            y_test
        )
    )

    # --------------------------------------------------------
    # 5. GARUDA Hybrid
    # --------------------------------------------------------

    (
        garuda_result,
        garuda_ml_model,
        anomaly_model,
        spatial_model,
        spatial_stats
    ) = run_garuda_hybrid(
        train_data,
        test_data
    )

    garuda_result[
        "Spatial_Stats"
    ] = spatial_stats

    all_results = (
        baseline_results
        + [garuda_result]
    )

    expected_models = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Extra Trees",
        "HistGradientBoosting",
        "XGBoost",
        "MLP Neural Network",
        "GARUDA Hybrid"
    ]

    actual_models = [
        result["Model"]
        for result in all_results
    ]

    if actual_models != expected_models:
        raise ValueError(
            "8-model comparison validation failed. "
            f"Expected {expected_models}, got {actual_models}."
        )

    # --------------------------------------------------------
    # 6. Save
    # --------------------------------------------------------

    (
        comparison,
        csv_path,
        json_path,
        confusion_path
    ) = save_results(
        all_results,
        train_data,
        test_data
    )

    # --------------------------------------------------------
    # 7. Charts
    # --------------------------------------------------------

    create_metric_chart(
        comparison
    )

    # --------------------------------------------------------
    # 8. Final output
    # --------------------------------------------------------

    print_final_table(
        comparison
    )

    print("\n")
    print("=" * 150)
    print("SAVED ARTIFACTS")
    print("=" * 150)

    print(
        f"\nComparison CSV:\n{csv_path}"
    )

    print(
        f"\nComparison JSON:\n{json_path}"
    )

    print(
        f"\nConfusion matrices:\n{confusion_path}"
    )

    print(
        f"\nCharts directory:\n{RESULTS_DIR}"
    )

    print("\n")
    print("=" * 150)
    print(
        "8-MODEL COMPARISON COMPLETED"
    )
    print("=" * 150)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()

