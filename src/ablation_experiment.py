from pathlib import Path

import sys

import json

import pickle

import warnings



import numpy as np

import pandas as pd



from sklearn.ensemble import IsolationForest

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

MODELS_DIR = PROJECT_ROOT / "models"



RESULTS_DIR.mkdir(

    parents=True,

    exist_ok=True

)



MODELS_DIR.mkdir(

    parents=True,

    exist_ok=True

)



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



SPATIAL_EPS = 0.002



SPATIAL_MIN_SAMPLES = 10



ANOMALY_CONTAMINATION = 0.20





BASE_FEATURES = [

    "daily_crime_count",

    "violent_crime_count",

    "crime_count_7d",

    "crime_count_30d"

]





TEMPORAL_FEATURES = BASE_FEATURES + [

    "temporal_consistency"

]





ANOMALY_FEATURES = [

    "daily_crime_count",

    "violent_crime_count",

    "crime_count_7d",

    "crime_count_30d"

]





SPATIAL_FEATURES = [

    "avg_latitude",

    "avg_longitude"

]





# ============================================================

# XGBOOST MODEL

# ============================================================

def create_xgboost_model(

    scale_pos_weight=1.0,

    random_state=None

):



    if random_state is None:

        random_state = RANDOM_STATE



    return XGBClassifier(

        n_estimators=250,

        max_depth=6,

        learning_rate=0.05,

        subsample=0.8,

        colsample_bytree=0.8,

        objective="binary:logistic",

        eval_metric="logloss",

        scale_pos_weight=scale_pos_weight,

        random_state=random_state,

        n_jobs=-1

    )



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

    print("=" * 80)

    print("GARUDA AI — HYBRID ABLATION EXPERIMENT")

    print("=" * 80)



    print("\nLoading Chicago crime dataset...")



    df = pd.read_csv(DATA_PATH)



    print(

        f"Original dataset shape: {df.shape}"

    )



    # --------------------------------------------------------

    # Build stable beat-level spatial reference from the

    # original crime records.

    #

    # These are location characteristics of the beat, not

    # future target information. They are later attached to

    # the engineered beat-day data for E4.

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



        spatial_source = (

            df[

                required_spatial_columns

            ]

            .copy()

        )



        spatial_source["latitude"] = pd.to_numeric(

            spatial_source["latitude"],

            errors="coerce"

        )



        spatial_source["longitude"] = pd.to_numeric(

            spatial_source["longitude"],

            errors="coerce"

        )



        spatial_source = (

            spatial_source

            .dropna(

                subset=[

                    "beat",

                    "latitude",

                    "longitude"

                ]

            )

            .groupby("beat")[

                ["latitude", "longitude"]

            ]

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



    else:



        print(

            "\nWarning: latitude/longitude columns were not "

            "found in the raw dataset."

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

    # Ensure beat-level spatial references are available.

    #

    # IMPORTANT:

    # create_historical_features() in the current GARUDA

    # feature-engineering pipeline may already create

    # avg_latitude / avg_longitude.  Merging spatial_source

    # unconditionally would then create _x/_y duplicate

    # columns and remove the expected names.

    #

    # Therefore:

    #   1. Keep existing engineered spatial columns when present.

    #   2. Only merge the raw beat reference when they are absent.

    #   3. Fill missing values from the training-independent beat

    #      reference when available.

    # --------------------------------------------------------



    spatial_columns_present = all(

        column in df.columns

        for column in SPATIAL_FEATURES

    )



    if spatial_source is not None and "beat" in df.columns:



        if not spatial_columns_present:



            df = df.merge(

                spatial_source,

                on="beat",

                how="left"

            )



        else:



            # Existing coordinates are authoritative.

            # Only fill missing coordinates from the stable

            # beat-level reference.

            spatial_lookup = spatial_source.set_index("beat")



            for column in SPATIAL_FEATURES:



                df[column] = pd.to_numeric(

                    df[column],

                    errors="coerce"

                )



                mapped_values = df["beat"].map(

                    spatial_lookup[column]

                )



                df[column] = df[column].fillna(

                    mapped_values

                )



    # Final diagnostic before the common cohort.

    spatial_status = [

        column

        for column in SPATIAL_FEATURES

        if column in df.columns

    ]



    print(

        "Spatial columns available: "

        f"{spatial_status}"

    )



    if not all(

        column in df.columns

        for column in SPATIAL_FEATURES

    ):

        raise ValueError(

            "Spatial coordinates could not be prepared. "

            f"Expected columns: {SPATIAL_FEATURES}; "

            f"available columns: {list(df.columns)}"

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

        "high_risk",



        # Keep beat-level spatial references in the common cohort.

        # E1-E3 ignore these columns; E4-E8 use them for spatial evidence.

        "avg_latitude",

        "avg_longitude"

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



    data = df[

        required_columns

    ].copy()



    numeric_columns = [



        "daily_crime_count",

        "violent_crime_count",



        "crime_count_7d",

        "crime_count_30d",



        "temporal_consistency",



        "next_day_crime_count",

        "high_risk",



        "avg_latitude",

        "avg_longitude"

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



    data = data.sort_values(

        [

            "crime_date",

            "beat"

        ]

    ).reset_index(

        drop=True

    )



    print("\n")

    print("=" * 80)

    print("COMMON EVALUATION COHORT")

    print("=" * 80)



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



    train_data = data.iloc[

        :split_index

    ].copy()



    test_data = data.iloc[

        split_index:

    ].copy()



    print(

        f"\nTraining rows: {len(train_data):,}"

    )



    print(

        f"Testing rows : {len(test_data):,}"

    )



    print(

        "\nTraining period:"

    )



    print(

        train_data["crime_date"].min(),

        "→",

        train_data["crime_date"].max()

    )



    print(

        "\nTesting period:"

    )



    print(

        test_data["crime_date"].min(),

        "→",

        test_data["crime_date"].max()

    )



    print(

        "\nTraining target:"

    )



    print(

        train_data["high_risk"]

        .value_counts()

        .sort_index()

        .to_dict()

    )



    print(

        "\nTesting target:"

    )



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



    return (

        train_data,

        test_data

    )





# ============================================================

# ISOLATION FOREST

# ============================================================





def add_anomaly_signal(

    train_data,

    test_data,

    random_state=None

):



    train_data = train_data.copy()

    test_data = test_data.copy()



    if random_state is None:

        random_state = RANDOM_STATE



    model = IsolationForest(

        n_estimators=200,

        contamination=ANOMALY_CONTAMINATION,

        random_state=random_state,

        n_jobs=-1

    )



    X_train = train_data[

        ANOMALY_FEATURES

    ]



    X_test = test_data[

        ANOMALY_FEATURES

    ]

    train_data,

    test_data



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



    print(

        f"\nTraining anomaly rate: "

        f"{train_data['anomaly_signal'].mean():.4f}"

    )



    print(

        f"Testing anomaly rate : "

        f"{test_data['anomaly_signal'].mean():.4f}"

    )



    return (

        train_data,

        test_data,

        model

    )





# ============================================================

# PREPARE SPATIAL COORDINATES

# ============================================================



def prepare_spatial_coordinates(train_data, test_data):

    """

    Prepare finite beat-level spatial coordinates for E4-E8.



    The engineered dataset already contains avg_latitude and avg_longitude

    created from the raw Chicago crime coordinates. These columns are kept

    in the common cohort but are not used by E1-E3.



    Any missing coordinate is first filled from the corresponding training

    beat and then from the training global median. Test observations are

    never used to construct the spatial reference.

    """



    train_data = train_data.copy()

    test_data = test_data.copy()



    for column in SPATIAL_FEATURES:

        if column not in train_data.columns:

            raise ValueError(

                f"Training data is missing required spatial column: {column}"

            )



        if column not in test_data.columns:

            raise ValueError(

                f"Testing data is missing required spatial column: {column}"

            )



        train_data[column] = pd.to_numeric(

            train_data[column],

            errors="coerce"

        )



        test_data[column] = pd.to_numeric(

            test_data[column],

            errors="coerce"

        )



        # Treat +/-inf exactly like missing coordinates.

        train_data.loc[~np.isfinite(train_data[column]), column] = np.nan

        test_data.loc[~np.isfinite(test_data[column]), column] = np.nan



    # --------------------------------------------------------

    # Training-only beat reference

    # --------------------------------------------------------



    centroid_source = (

        train_data

        .dropna(subset=SPATIAL_FEATURES)

        .groupby("beat")[SPATIAL_FEATURES]

        .median()

    )



    for column in SPATIAL_FEATURES:

        train_data[column] = train_data[column].fillna(

            train_data["beat"].map(centroid_source[column])

        )



        test_data[column] = test_data[column].fillna(

            test_data["beat"].map(centroid_source[column])

        )



    # --------------------------------------------------------

    # Training-only global fallback

    # --------------------------------------------------------



    for column in SPATIAL_FEATURES:

        global_median = train_data[column].median()



        if not np.isfinite(global_median):

            raise ValueError(

                f"No finite training reference exists for {column}."

            )



        train_data[column] = train_data[column].fillna(global_median)

        test_data[column] = test_data[column].fillna(global_median)



    # --------------------------------------------------------

    # Final finite validation

    # --------------------------------------------------------



    train_coordinates = train_data[SPATIAL_FEATURES].to_numpy(dtype=float)

    test_coordinates = test_data[SPATIAL_FEATURES].to_numpy(dtype=float)



    if not np.isfinite(train_coordinates).all():

        raise ValueError(

            "Training spatial coordinates contain non-finite values after preparation."

        )



    if not np.isfinite(test_coordinates).all():

        raise ValueError(

            "Testing spatial coordinates contain non-finite values after preparation."

        )



    print("\nSpatial coordinates prepared successfully:")

    print(f"  Training spatial missing values: {int(np.isnan(train_coordinates).sum())}")

    print(f"  Testing spatial missing values : {int(np.isnan(test_coordinates).sum())}")

    print(f"  Training spatial beat references: {len(centroid_source):,}")

    print(

        f"  Latitude range : {train_data['avg_latitude'].min():.6f} "

        f"-> {train_data['avg_latitude'].max():.6f}"

    )

    print(

        f"  Longitude range: {train_data['avg_longitude'].min():.6f} "

        f"-> {train_data['avg_longitude'].max():.6f}"

    )



    return train_data, test_data





# ============================================================

# DBSCAN SPATIAL SIGNAL

# ============================================================



def add_spatial_signal(

    train_data,

    test_data

):

    """

    Generate spatial evidence using DBSCAN structure plus local spatial

    density.



    DBSCAN identifies the spatial structure in the training locations.

    A continuous density score is then computed from nearby training points.

    This avoids the previous failure mode where every point received the

    same binary spatial_signal=1.

    """



    train_data = train_data.copy()

    test_data = test_data.copy()



    train_coordinates = (

        train_data[

            SPATIAL_FEATURES

        ].to_numpy(dtype=float)

    )



    test_coordinates = (

        test_data[

            SPATIAL_FEATURES

        ].to_numpy(dtype=float)

    )



    if not np.isfinite(train_coordinates).all():

        raise ValueError(

            "Training spatial coordinates contain non-finite values."

        )



    if not np.isfinite(test_coordinates).all():

        raise ValueError(

            "Testing spatial coordinates contain non-finite values."

        )



    # --------------------------------------------------------

    # DBSCAN structure

    # --------------------------------------------------------



    model = DBSCAN(

        eps=SPATIAL_EPS,

        min_samples=SPATIAL_MIN_SAMPLES

    )



    model.fit(

        train_coordinates

    )



    train_labels = model.labels_



    non_noise = train_labels[

        train_labels != -1

    ]



    clusters = (

        len(np.unique(non_noise))

        if len(non_noise) > 0

        else 0

    )



    noise_rate = (

        float(np.mean(train_labels == -1))

    )



    # --------------------------------------------------------

    # Continuous local spatial density

    # --------------------------------------------------------



    k = min(

        10,

        max(2, len(train_coordinates) - 1)

    )



    neighbors = NearestNeighbors(

        n_neighbors=k + 1

    )



    neighbors.fit(

        train_coordinates

    )



    train_distances, _ = neighbors.kneighbors(

        train_coordinates

    )



    # Ignore self-distance in column 0.

    train_mean_distance = np.mean(

        train_distances[:, 1:],

        axis=1

    )



    train_density_raw = 1.0 / (

        train_mean_distance + 1e-9

    )



    test_neighbors = NearestNeighbors(

        n_neighbors=min(k, len(train_coordinates))

    )



    test_neighbors.fit(

        train_coordinates

    )



    test_distances, _ = test_neighbors.kneighbors(

        test_coordinates

    )



    test_mean_distance = np.mean(

        test_distances,

        axis=1

    )



    test_density_raw = 1.0 / (

        test_mean_distance + 1e-9

    )



    # Normalize using TRAINING distribution only.

    density_low = np.percentile(

        train_density_raw,

        5

    )



    density_high = np.percentile(

        train_density_raw,

        95

    )



    if density_high <= density_low:

        density_high = density_low + 1e-9



    train_signal = np.clip(

        (

            train_density_raw - density_low

        )

        /

        (

            density_high - density_low

        ),

        0.0,

        1.0

    )



    test_signal = np.clip(

        (

            test_density_raw - density_low

        )

        /

        (

            density_high - density_low

        ),

        0.0,

        1.0

    )



    train_data["spatial_signal"] = train_signal

    test_data["spatial_signal"] = test_signal



    stats = {

        "clusters": int(clusters),

        "noise_rate": float(noise_rate),

        "train_signal_rate": float(

            train_signal.mean()

        ),

        "test_signal_rate": float(

            test_signal.mean()

        ),

        "train_signal_std": float(

            train_signal.std()

        ),

        "test_signal_std": float(

            test_signal.std()

        )

    }



    print(

        f"\nDBSCAN clusters: {clusters}"

    )



    print(

        f"Training spatial signal mean: "

        f"{stats['train_signal_rate']:.4f}"

    )



    print(

        f"Testing spatial signal mean : "

        f"{stats['test_signal_rate']:.4f}"

    )



    print(

        f"Training spatial signal std : "

        f"{stats['train_signal_std']:.4f}"

    )



    print(

        f"Testing spatial signal std  : "

        f"{stats['test_signal_std']:.4f}"

    )



    print(

        f"Training spatial noise rate : "

        f"{noise_rate:.4f}"

    )



    return (

        train_data,

        test_data,

        model,

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



        "Recall":

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



    return (

        metrics,

        matrix

    )





# ============================================================

# RUN EXPERIMENT

# ============================================================



def run_experiment(
    name,
    train_base,
    test_base,
    use_temporal=False,
    use_anomaly=False,
    use_spatial=False,
    random_state=None
):

    if random_state is None:
        random_state = RANDOM_STATE




    print("\n")

    print("#" * 80)

    print(name)

    print("#" * 80)



    train_data = train_base.copy()

    test_data = test_base.copy()



    anomaly_model = None

    spatial_model = None

    spatial_stats = {}



    features = BASE_FEATURES.copy()



    # --------------------------------------------------------

    # TEMPORAL

    # --------------------------------------------------------



    if use_temporal:



        features.append(

            "temporal_consistency"

        )



        print(

            "\nTemporal consistency: ENABLED"

        )



    # --------------------------------------------------------

    # ANOMALY

    # --------------------------------------------------------



    if use_anomaly:



        print(

            "\nIsolation Forest: ENABLED"

        )



        (

            train_data,

            test_data,

            anomaly_model

        ) = add_anomaly_signal(

            train_data,

            test_data,

            random_state=random_state

        )



        features.append(

            "anomaly_score"

        )



    # --------------------------------------------------------

    # SPATIAL

    # --------------------------------------------------------



    if use_spatial:



        print(

            "\nDBSCAN spatial analysis + local density: ENABLED"

        )



        (

            train_data,

            test_data

        ) = prepare_spatial_coordinates(

            train_data,

            test_data

        )



        (

            train_data,

            test_data,

            spatial_model,

            spatial_stats

        ) = add_spatial_signal(

            train_data,

            test_data

        )



        features.append(

            "spatial_signal"

        )



    print(

        "\nFeatures:"

    )



    for feature in features:



        print(

            f"  - {feature}"

        )



    # --------------------------------------------------------

    # DATA

    # --------------------------------------------------------



    X_train = train_data[

        features

    ]



    y_train = train_data[

        "high_risk"

    ]



    X_test = test_data[

        features

    ]



    y_test = test_data[

        "high_risk"

    ]



    # --------------------------------------------------------

    # XGBOOST

    # --------------------------------------------------------



    # Compute class weight from TRAINING DATA ONLY.

    # This addresses the approximately 8% positive class prevalence

    # without using any information from the chronological test set.

    negative_count = int(

        (y_train == 0).sum()

    )



    positive_count = int(

        (y_train == 1).sum()

    )



    scale_pos_weight = (

        negative_count / positive_count

        if positive_count > 0

        else 1.0

    )



    print(

        f"\nTraining class distribution: "

        f"negative={negative_count:,}, "

        f"positive={positive_count:,}"

    )



    print(

        f"Training-only scale_pos_weight: "

        f"{scale_pos_weight:.4f}"

    )



    model = create_xgboost_model(

        scale_pos_weight=scale_pos_weight,

        random_state=random_state

    )



    print(

        "\nTraining XGBoost..."

    )



    model.fit(

        X_train,

        y_train

    )



    y_pred = model.predict(

        X_test

    )



    y_probability = (

        model.predict_proba(

            X_test

        )[:, 1]

    )



    # --------------------------------------------------------

    # METRICS

    # --------------------------------------------------------



    metrics, matrix = calculate_metrics(

        y_test,

        y_pred,

        y_probability

    )



    print(

        "\nPerformance:"

    )



    print(

        f"Accuracy      : {metrics['Accuracy']:.4f}"

    )



    print(

        f"Precision     : {metrics['Precision']:.4f}"

    )



    print(

        f"Sensitivity   : {metrics['Sensitivity']:.4f}"

    )



    print(

        f"Specificity   : {metrics['Specificity']:.4f}"

    )



    print(

        f"F1 Score      : {metrics['F1 Score']:.4f}"

    )



    print(

        f"ROC-AUC       : {metrics['ROC-AUC']:.4f}"

    )



    print(

        f"PR-AUC        : {metrics['PR-AUC']:.4f}"

    )



    print(

        f"MCC           : {metrics['MCC']:.4f}"

    )



    print(

        "Cohen's Kappa : {:.4f}".format(metrics["Cohen's Kappa"])

    )



    print(

        f"Log Loss      : {metrics['Log Loss']:.4f}"

    )



    print(

        "\nConfusion Matrix:"

    )



    print(

        matrix

    )



    return {



        "experiment":

            name,



        "features":

            features,



        "metrics":

            metrics,



        "confusion_matrix":

            matrix.tolist(),



        "model":

            model,



        "anomaly_model":

            anomaly_model,



        "spatial_model":

            spatial_model,



        "spatial_stats":

            spatial_stats,



        "test_data":

            test_data,



        "y_probability":

            y_probability,



        "y_pred":

            y_pred,



        "scale_pos_weight":

            float(scale_pos_weight),

        "random_state":

            int(random_state)

    }





# ============================================================

# SAVE RESULTS

# ============================================================



def save_results(results):



    rows = []



    for result in results:



        row = {

            "Experiment":

                result["experiment"]

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



    csv_path = (

        RESULTS_DIR

        / "garuda_ablation_results.csv"

    )



    comparison.to_csv(

        csv_path,

        index=False

    )



    json_data = []



    for result in results:



        json_data.append({



            "experiment":

                result["experiment"],



            "features":

                result["features"],



            "metrics":

                result["metrics"],



            "confusion_matrix":

                result["confusion_matrix"],



            "spatial_stats":

                result["spatial_stats"],



            "scale_pos_weight":

                result["scale_pos_weight"]

        })



    json_path = (

        RESULTS_DIR

        / "garuda_ablation_results.json"

    )



    with open(

        json_path,

        "w",

        encoding="utf-8"

    ) as file:



        json.dump(

            json_data,

            file,

            indent=4

        )



    return (

        comparison,

        csv_path,

        json_path

    )





# ============================================================

# SAVE E4 MODEL

# ============================================================



def save_e4_model(result):



    model_path = (

        MODELS_DIR

        / "garuda_xgboost_e4.pkl"

    )



    artifact = {



        "model":

            result["model"],



        "anomaly_model":

            result["anomaly_model"],



        "spatial_model":

            result["spatial_model"],



        "features":

            result["features"],



        "spatial_stats":

            result["spatial_stats"],



        "configuration": {



            "random_state":

                RANDOM_STATE,



            "spatial_eps":

                SPATIAL_EPS,



            "spatial_min_samples":

                SPATIAL_MIN_SAMPLES,



            "anomaly_contamination":

                ANOMALY_CONTAMINATION,



            "spatial_coordinate_strategy":

                "training_beat_reference_with_training_only_missing_value_fallback",



            "spatial_evidence_strategy":

                "dbscan_structure_plus_training_normalized_local_density",



            "class_weight_strategy":

                "training_only_scale_pos_weight"

        }

    }



    with open(

        model_path,

        "wb"

    ) as file:



        pickle.dump(

            artifact,

            file

        )



    return model_path





# ============================================================
# REPEATED-SEED STATISTICAL VALIDATION
# ============================================================

RANDOM_SEEDS = [42, 52, 62, 72, 82]

ABLATION_DEFINITIONS = [
    ("E1 — ML Only", False, False, False),
    ("E2 — ML + Temporal", True, False, False),
    ("E3 — ML + Anomaly", False, True, False),
    ("E4 — ML + Spatial", False, False, True),
    ("E5 — ML + Temporal + Anomaly", True, True, False),
    ("E6 — ML + Temporal + Spatial", True, False, True),
    ("E7 — ML + Anomaly + Spatial", False, True, True),
    ("E8 — ML + Temporal + Anomaly + Spatial", True, True, True),
]

STATISTICAL_METRICS = [
    "Accuracy", "Precision", "Sensitivity", "Specificity",
    "F1 Score", "ROC-AUC", "PR-AUC", "MCC",
    "Cohen's Kappa", "Log Loss"
]

def save_repeated_seed_results(all_results):
    raw_rows = []
    for result in all_results:
        row = {"Seed": result["random_state"], "Experiment": result["experiment"]}
        row.update(result["metrics"])
        raw_rows.append(row)

    raw_df = pd.DataFrame(raw_rows)
    raw_csv = RESULTS_DIR / "garuda_ablation_repeated_seed_results.csv"
    raw_df.to_csv(raw_csv, index=False)

    summary_rows = []
    for experiment in [item[0] for item in ABLATION_DEFINITIONS]:
        subset = raw_df[raw_df["Experiment"] == experiment]
        row = {"Experiment": experiment, "Seeds": len(subset)}
        for metric in STATISTICAL_METRICS:
            values = pd.to_numeric(subset[metric], errors="coerce")
            mean_value = float(values.mean())
            std_value = float(values.std(ddof=1))
            row[f"{metric} Mean"] = mean_value
            row[f"{metric} Std"] = std_value
            row[f"{metric} Mean ± Std"] = f"{mean_value:.4f} ± {std_value:.4f}"
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_csv = RESULTS_DIR / "garuda_ablation_repeated_seed_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    json_path = RESULTS_DIR / "garuda_ablation_repeated_seed_summary.json"
    json_data = {
        "seeds": RANDOM_SEEDS,
        "number_of_seeds": len(RANDOM_SEEDS),
        "common_cohort": True,
        "chronological_split": True,
        "spatial_component_deterministic": True,
        "results": summary_df.to_dict(orient="records"),
    }
    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(json_data, file, indent=4)

    return raw_df, summary_df, raw_csv, summary_csv, json_path


def main():
    df = load_and_engineer_data()
    train_data, test_data = prepare_common_cohort(df)

    print("\n" + "=" * 120)
    print("GARUDA AI — REPEATED-SEED 8-COMBINATION ABLATION VALIDATION")
    print("=" * 120)
    print("\nSame chronological cohort is reused for every seed.")
    print(f"Seeds: {RANDOM_SEEDS}")
    print("DBSCAN/spatial evidence remains deterministic.")
    print("Original single-seed result files will NOT be overwritten.\n")

    all_results = []
    for seed in RANDOM_SEEDS:
        print("\n" + "=" * 120)
        print(f"RUNNING SEED {seed}")
        print("=" * 120)

        for name, use_temporal, use_anomaly, use_spatial in ABLATION_DEFINITIONS:
            result = run_experiment(
                name, train_data, test_data,
                use_temporal=use_temporal,
                use_anomaly=use_anomaly,
                use_spatial=use_spatial,
                random_state=seed
            )
            all_results.append(result)

    raw_df, summary_df, raw_csv, summary_csv, json_path = save_repeated_seed_results(all_results)

    print("\n" + "=" * 150)
    print("GARUDA AI — REPEATED-SEED MEAN ± STD SUMMARY")
    print("=" * 150)

    display_columns = [
        "Experiment", "Accuracy Mean ± Std", "Precision Mean ± Std",
        "Sensitivity Mean ± Std", "Specificity Mean ± Std",
        "F1 Score Mean ± Std", "ROC-AUC Mean ± Std", "PR-AUC Mean ± Std",
        "MCC Mean ± Std", "Cohen's Kappa Mean ± Std", "Log Loss Mean ± Std"
    ]
    print(summary_df[display_columns].to_string(index=False))

    print("\n" + "=" * 120)
    print("SAVED STATISTICAL VALIDATION ARTIFACTS")
    print("=" * 120)
    print(f"\nRaw seed-level CSV:\n{raw_csv}")
    print(f"\nMean ± Std CSV:\n{summary_csv}")
    print(f"\nSummary JSON:\n{json_path}")
    print("\nNo original single-seed ablation result file was overwritten.")
    print("No model artifact was overwritten.")
    print("\n" + "=" * 120)
    print("GARUDA REPEATED-SEED ABLATION VALIDATION COMPLETED")
    print("=" * 120)


if __name__ == "__main__":
    main()


