import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
)

from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier

from src.feature_engineering import (
    add_spatial_signal,
    get_spatial_statistics,
)


# ============================================================
# CONFIGURATION
# ============================================================

EXCLUDED_CLASSIFICATION_FEATURES = [
    "next_day_crime_count",
    "risk_threshold",
    "high_risk",
    "beat",
    "crime_date",
]

EXCLUDED_REGRESSION_FEATURES = [
    "next_day_crime_count",
    "beat",
    "crime_date",
]

ANOMALY_FEATURES = [
    "daily_crime_count",
    "violent_crime_count",
    "crime_count_7d",
    "crime_count_30d",
]

SPATIAL_FEATURES = [
    "avg_latitude",
    "avg_longitude",
]

SPATIAL_EPS = 0.002
SPATIAL_MIN_SAMPLES = 10


# ============================================================
# DATA PREPARATION
# ============================================================

def prepare_data(df, features, target, task="classification"):

    data = df.copy()

    if task == "classification":
        excluded = EXCLUDED_CLASSIFICATION_FEATURES
    else:
        excluded = EXCLUDED_REGRESSION_FEATURES

    requested_features = [
        feature
        for feature in features
        if feature != "anomaly_score"
        and feature != "spatial_signal"
    ]

    valid_features = [
        feature
        for feature in requested_features
        if feature in data.columns
        and feature not in excluded
        and feature != target
    ]

    required_columns = valid_features + [target]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    data = data[required_columns].copy()

    for column in valid_features:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce"
        )

    data[target] = pd.to_numeric(
        data[target],
        errors="coerce"
    )

    data = data.dropna()

    return data, valid_features


# ============================================================
# TIME ORDERING
# ============================================================

def sort_by_time(df):

    data = df.copy()

    if "crime_date" in data.columns:

        data["crime_date"] = pd.to_datetime(
            data["crime_date"],
            errors="coerce"
        )

        data = data.sort_values("crime_date")

    elif "date" in data.columns:

        data["date"] = pd.to_datetime(
            data["date"],
            errors="coerce"
        )

        data = data.sort_values("date")

    return data.reset_index(drop=True)


# ============================================================
# CHRONOLOGICAL SPLIT
# ============================================================

def time_split(data, train_ratio=0.8):

    split_index = int(
        len(data) * train_ratio
    )

    train_data = data.iloc[
        :split_index
    ].copy()

    test_data = data.iloc[
        split_index:
    ].copy()

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

    available_features = [
        feature
        for feature in ANOMALY_FEATURES
        if feature in train_data.columns
        and feature in test_data.columns
    ]

    if not available_features:
        raise ValueError(
            "No valid features available for Isolation Forest."
        )

    X_train = train_data[
        available_features
    ].copy()

    X_test = test_data[
        available_features
    ].copy()

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
            "NaN values found in Isolation Forest training data."
        )

    if X_test.isna().any().any():
        raise ValueError(
            "NaN values found in Isolation Forest test data."
        )

    isolation_forest = IsolationForest(
        n_estimators=200,
        contamination="auto",
        random_state=42,
        n_jobs=-1,
    )

    # IMPORTANT:
    # Fit ONLY on training data.

    isolation_forest.fit(X_train)

    train_raw_score = (
        isolation_forest.score_samples(X_train)
    )

    test_raw_score = (
        isolation_forest.score_samples(X_test)
    )

    # Higher value = more anomalous

    train_data["anomaly_score"] = (
        -train_raw_score
    )

    test_data["anomaly_score"] = (
        -test_raw_score
    )

    train_predictions = (
        isolation_forest.predict(X_train)
    )

    test_predictions = (
        isolation_forest.predict(X_test)
    )

    train_data["anomaly_signal"] = (
        train_predictions == -1
    ).astype(int)

    test_data["anomaly_signal"] = (
        test_predictions == -1
    ).astype(int)

    print("\n" + "=" * 60)
    print("ISOLATION FOREST ANOMALY DETECTION")
    print("=" * 60)

    print("Features used:")

    for feature in available_features:
        print(f"  - {feature}")

    print(
        "\nTraining rows:",
        len(train_data)
    )

    print(
        "Testing rows :",
        len(test_data)
    )

    print(
        "\nTraining anomaly rate:",
        round(
            train_data[
                "anomaly_signal"
            ].mean(),
            4
        )
    )

    print(
        "Testing anomaly rate :",
        round(
            test_data[
                "anomaly_signal"
            ].mean(),
            4
        )
    )

    print("=" * 60)

    return (
        train_data,
        test_data,
        isolation_forest,
    )


# ============================================================
# CLASSIFICATION MODELS
# ============================================================

def get_classification_models():

    models = {

        "Logistic Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LogisticRegression(
                    max_iter=1000,
                    random_state=42
                )
            ),
        ]),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),

        "Extra Trees": ExtraTreesClassifier(
            n_estimators=250,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),

        "HistGradientBoosting": (
            HistGradientBoostingClassifier(
                max_iter=200,
                learning_rate=0.05,
                max_leaf_nodes=31,
                random_state=42
            )
        ),

        "XGBoost": XGBClassifier(
            n_estimators=250,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=42,
            n_jobs=-1
        ),

        "MLP Neural Network": Pipeline([
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
                    random_state=42
                )
            ),
        ]),
    }

    return models


# ============================================================
# REGRESSION MODELS
# ============================================================

def get_regression_models():

    models = {

        "Linear Regression": Pipeline([
            (
                "scaler",
                StandardScaler()
            ),
            (
                "model",
                LinearRegression()
            ),
        ]),

        "Decision Tree": DecisionTreeRegressor(
            max_depth=8,
            random_state=42
        ),

        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=12,
            random_state=42,
            n_jobs=-1
        ),
    }

    return models


# ============================================================
# CLASSIFICATION TRAINING
# ============================================================

def train_classification_model(
    df,
    features,
    target="high_risk",
    model_name="Random Forest",
    train_ratio=0.8,
):

    data = sort_by_time(df)

    use_anomaly = (
        "anomaly_score" in features
    )

    use_spatial = (
        "spatial_signal" in features
    )

    base_features = [
        feature
        for feature in features
        if feature not in [
            "anomaly_score",
            "spatial_signal",
        ]
    ]

    # --------------------------------------------------------
    # Spatial coordinates are required internally to create
    # the DBSCAN signal, but they are NOT automatically used
    # as final ML features.
    # --------------------------------------------------------

    if use_spatial:

        for spatial_feature in SPATIAL_FEATURES:

            if spatial_feature not in base_features:
                base_features.append(
                    spatial_feature
                )

    data, valid_features = prepare_data(
        data,
        base_features,
        target,
        task="classification",
    )

    train_data, test_data = time_split(
        data,
        train_ratio=train_ratio
    )

    anomaly_model = None
    spatial_model = None
    spatial_statistics = None

    # --------------------------------------------------------
    # ANOMALY SIGNAL
    # --------------------------------------------------------

    if use_anomaly:

        (
            train_data,
            test_data,
            anomaly_model
        ) = add_anomaly_signal(
            train_data,
            test_data
        )

    # --------------------------------------------------------
    # SPATIAL SIGNAL
    # --------------------------------------------------------

    if use_spatial:

        (
            train_data,
            test_data,
            spatial_model
        ) = add_spatial_signal(
            train_data,
            test_data,
            eps=SPATIAL_EPS,
            min_samples=SPATIAL_MIN_SAMPLES,
        )

        spatial_statistics = (
            get_spatial_statistics(
                train_data,
                test_data,
                spatial_model,
            )
        )

        print("\n" + "=" * 60)
        print("DBSCAN SPATIAL SIGNAL")
        print("=" * 60)

        print(
            "DBSCAN eps:",
            SPATIAL_EPS
        )

        print(
            "DBSCAN min_samples:",
            SPATIAL_MIN_SAMPLES
        )

        print(
            "Training spatial signal rate:",
            round(
                spatial_statistics[
                    "train_spatial_signal_rate"
                ],
                4
            )
        )

        print(
            "Testing spatial signal rate:",
            round(
                spatial_statistics[
                    "test_spatial_signal_rate"
                ],
                4
            )
        )

        print(
            "Training clusters:",
            spatial_statistics[
                "train_clusters"
            ]
        )

        print(
            "Training noise rate:",
            round(
                spatial_statistics[
                    "train_noise_rate"
                ],
                4
            )
        )

        print("=" * 60)

    # --------------------------------------------------------
    # FINAL FEATURES
    # --------------------------------------------------------

    final_features = [
        feature
        for feature in valid_features
        if feature not in SPATIAL_FEATURES
    ]

    if use_anomaly:

        final_features.append(
            "anomaly_score"
        )

    if use_spatial:

        final_features.append(
            "spatial_signal"
        )

    X_train = train_data[
        final_features
    ]

    y_train = train_data[
        target
    ]

    X_test = test_data[
        final_features
    ]

    y_test = test_data[
        target
    ]

    if y_train.nunique() < 2:

        raise ValueError(
            "Training data contains only one target class."
        )

    if y_test.nunique() < 2:

        raise ValueError(
            "Testing data contains only one target class."
        )

    models = get_classification_models()

    if model_name not in models:

        raise ValueError(
            f"Unknown classification model: {model_name}"
        )

    model = models[
        model_name
    ]

    print(
        f"\nTraining {model_name}..."
    )

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    if hasattr(
        model,
        "predict_proba"
    ):

        y_probability = (
            model.predict_proba(
                X_test
            )[:, 1]
        )

    else:

        y_probability = None

    metrics = {

        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "F1 Score": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),
    }

    if y_probability is not None:

        metrics["ROC-AUC"] = (
            roc_auc_score(
                y_test,
                y_probability
            )
        )

        metrics["PR-AUC"] = (
            average_precision_score(
                y_test,
                y_probability
            )
        )

        fpr, tpr, thresholds = (
            roc_curve(
                y_test,
                y_probability
            )
        )

    else:

        metrics["ROC-AUC"] = None
        metrics["PR-AUC"] = None

        fpr = None
        tpr = None
        thresholds = None

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    print(
        model_name,
        metrics
    )

    return {

        "model": model,

        "model_name": model_name,

        "features": final_features,

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test,

        "y_pred": y_pred,

        "y_probability": y_probability,

        "metrics": metrics,

        "confusion_matrix": cm,

        "fpr": fpr,

        "tpr": tpr,

        "thresholds": thresholds,

        "anomaly_model": anomaly_model,

        "spatial_model": spatial_model,

        "spatial_statistics": spatial_statistics,

        "train_data": train_data,

        "test_data": test_data,
    }


# ============================================================
# REGRESSION TRAINING
# ============================================================

def train_regression_model(
    df,
    features,
    target,
    model_name="Random Forest",
    train_ratio=0.8,
):

    data = sort_by_time(df)

    data, valid_features = prepare_data(
        data,
        features,
        target,
        task="regression",
    )

    train_data, test_data = time_split(
        data,
        train_ratio=train_ratio
    )

    X_train = train_data[
        valid_features
    ]

    y_train = train_data[
        target
    ]

    X_test = test_data[
        valid_features
    ]

    y_test = test_data[
        target
    ]

    models = get_regression_models()

    if model_name not in models:

        raise ValueError(
            f"Unknown regression model: {model_name}"
        )

    model = models[
        model_name
    ]

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    metrics = {

        "MAE": mean_absolute_error(
            y_test,
            y_pred
        ),

        "MSE": mse,

        "RMSE": mse ** 0.5,

        "R2": r2_score(
            y_test,
            y_pred
        ),
    }

    print(
        model_name,
        metrics
    )

    return {

        "model": model,

        "model_name": model_name,

        "features": valid_features,

        "X_train": X_train,

        "X_test": X_test,

        "y_train": y_train,

        "y_test": y_test,

        "y_pred": y_pred,

        "metrics": metrics,
    }


# ============================================================
# GENERIC TRAINING
# ============================================================

def train_model(
    df,
    features,
    target,
    task="classification",
    model_name=None,
    train_ratio=0.8,
):

    if task == "classification":

        if model_name is None:
            model_name = "Random Forest"

        return train_classification_model(
            df=df,
            features=features,
            target=target,
            model_name=model_name,
            train_ratio=train_ratio,
        )

    if task == "regression":

        if model_name is None:
            model_name = "Random Forest"

        return train_regression_model(
            df=df,
            features=features,
            target=target,
            model_name=model_name,
            train_ratio=train_ratio,
        )

    raise ValueError(
        "task must be 'classification' or 'regression'"
    )


# ============================================================
# TRAIN ALL CLASSIFICATION MODELS
# ============================================================

def train_all_models(
    df,
    features,
    target="high_risk",
    train_ratio=0.8,
):

    results = {}

    model_names = [
        "Logistic Regression",
        "Decision Tree",
        "Random Forest",
        "Extra Trees",
        "HistGradientBoosting",
        "XGBoost",
        "MLP Neural Network",
    ]

    for model_name in model_names:

        print("\n")
        print(
            "#" * 70
        )

        print(
            f"TRAINING: {model_name}"
        )

        print(
            "#" * 70
        )

        results[
            model_name
        ] = train_classification_model(
            df=df,
            features=features,
            target=target,
            model_name=model_name,
            train_ratio=train_ratio,
        )

    return results


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models(
    results,
    metric="F1 Score"
):

    rows = []

    for model_name, result in results.items():

        row = {
            "Model": model_name
        }

        for metric_name, value in (
            result["metrics"].items()
        ):

            row[
                metric_name
            ] = value

        rows.append(row)

    comparison = pd.DataFrame(
        rows
    )

    if metric in comparison.columns:

        comparison = comparison.sort_values(
            metric,
            ascending=False
        )

    return comparison.reset_index(
        drop=True
    )


# ============================================================
# E3 EXPERIMENT
# ============================================================

def run_e3_experiment(df):

    features = [

        "daily_crime_count",

        "violent_crime_count",

        "crime_count_7d",

        "crime_count_30d",

        "temporal_consistency",

        "anomaly_score",
    ]

    print("\n")

    print(
        "=" * 70
    )

    print(
        "GARUDA E3 EXPERIMENT"
    )

    print(
        "ML + TEMPORAL + ANOMALY"
    )

    print(
        "=" * 70
    )

    results = train_all_models(
        df=df,
        features=features,
        target="high_risk",
        train_ratio=0.8,
    )

    comparison = compare_models(
        results,
        metric="F1 Score"
    )

    print("\n")

    print(
        "=" * 70
    )

    print(
        "E3 MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "=" * 70
    )

    return (
        results,
        comparison
    )


# ============================================================
# E4 EXPERIMENT
# ============================================================

def run_e4_experiment(df):

    features = [

        "daily_crime_count",

        "violent_crime_count",

        "crime_count_7d",

        "crime_count_30d",

        "temporal_consistency",

        "anomaly_score",

        "spatial_signal",
    ]

    print("\n")

    print(
        "=" * 70
    )

    print(
        "GARUDA E4 EXPERIMENT"
    )

    print(
        "ML + TEMPORAL + ANOMALY + SPATIAL"
    )

    print(
        "=" * 70
    )

    print(
        "\nSpatial method: DBSCAN"
    )

    print(
        "eps:",
        SPATIAL_EPS
    )

    print(
        "min_samples:",
        SPATIAL_MIN_SAMPLES
    )

    results = train_all_models(
        df=df,
        features=features,
        target="high_risk",
        train_ratio=0.8,
    )

    comparison = compare_models(
        results,
        metric="F1 Score"
    )

    print("\n")

    print(
        "=" * 70
    )

    print(
        "E4 MODEL COMPARISON"
    )

    print(
        "=" * 70
    )

    print(
        comparison.to_string(
            index=False
        )
    )

    print(
        "=" * 70
    )

    # --------------------------------------------------------
    # Spatial statistics
    # --------------------------------------------------------

    first_result = next(
        iter(results.values())
    )

    spatial_statistics = (
        first_result[
            "spatial_statistics"
        ]
    )

    print("\n")

    print(
        "=" * 70
    )

    print(
        "E4 SPATIAL STATISTICS"
    )

    print(
        "=" * 70
    )

    if spatial_statistics is not None:

        print(
            "Training spatial signal rate:",
            round(
                spatial_statistics[
                    "train_spatial_signal_rate"
                ],
                4
            )
        )

        print(
            "Testing spatial signal rate:",
            round(
                spatial_statistics[
                    "test_spatial_signal_rate"
                ],
                4
            )
        )

        print(
            "Training clusters:",
            spatial_statistics[
                "train_clusters"
            ]
        )

        print(
            "Training noise rate:",
            round(
                spatial_statistics[
                    "train_noise_rate"
                ],
                4
            )
        )

    print(
        "=" * 70
    )

    return (
        results,
        comparison
    )