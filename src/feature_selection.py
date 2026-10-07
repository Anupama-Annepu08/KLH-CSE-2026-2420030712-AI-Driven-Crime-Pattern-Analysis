import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


EXCLUDED_FEATURES = [
    "next_day_crime_count",
    "risk_threshold",
    "high_risk"
]


def prepare_features(df, target):
    numeric_df = df.select_dtypes(include=np.number).copy()

    excluded = [
        column
        for column in EXCLUDED_FEATURES
        if column in numeric_df.columns
    ]

    excluded = [
        column
        for column in excluded
        if column != target
    ]

    numeric_df = numeric_df.drop(
        columns=excluded,
        errors="ignore"
    )

    return numeric_df


def correlation_selection(df, target, threshold=0.5):
    numeric_df = prepare_features(df, target)

    if target not in numeric_df.columns:
        return [], pd.DataFrame()

    correlations = (
        numeric_df
        .corr()[target]
        .drop(target)
        .abs()
        .sort_values(ascending=False)
    )

    result = pd.DataFrame({
        "Feature": correlations.index,
        "Correlation": correlations.values
    })

    selected = result[
        result["Correlation"] >= threshold
    ]["Feature"].tolist()

    return selected, result


def mutual_information_selection(df, target, top_k=10):
    numeric_df = prepare_features(df, target)

    if target not in numeric_df.columns:
        return [], pd.DataFrame()

    numeric_df = numeric_df.dropna()

    if numeric_df.empty:
        return [], pd.DataFrame()

    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]

    if X.empty:
        return [], pd.DataFrame()

    scores = mutual_info_classif(
        X,
        y,
        random_state=42
    )

    result = pd.DataFrame({
        "Feature": X.columns,
        "Mutual Information": scores
    }).sort_values(
        "Mutual Information",
        ascending=False
    )

    selected = (
        result
        .head(top_k)["Feature"]
        .tolist()
    )

    return selected, result


def random_forest_selection(df, target, top_k=10):
    numeric_df = prepare_features(df, target)

    if target not in numeric_df.columns:
        return [], pd.DataFrame()

    numeric_df = numeric_df.dropna()

    if numeric_df.empty:
        return [], pd.DataFrame()

    X = numeric_df.drop(columns=[target])
    y = numeric_df[target]

    if X.empty:
        return [], pd.DataFrame()

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X, y)

    result = pd.DataFrame({
        "Feature": X.columns,
        "Importance": model.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    )

    selected = (
        result
        .head(top_k)["Feature"]
        .tolist()
    )

    return selected, result