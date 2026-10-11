import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN


# ============================================================
# DATE COLUMN DETECTION
# ============================================================

def detect_date_columns(df):

    date_columns = []

    for column in df.columns:

        if pd.api.types.is_datetime64_any_dtype(df[column]):
            date_columns.append(column)
            continue

        sample = df[column].dropna().head(100)

        if len(sample) == 0:
            continue

        parsed = pd.to_datetime(
            sample,
            errors="coerce",
            format="mixed"
        )

        if parsed.notna().mean() >= 0.8:
            date_columns.append(column)

    return date_columns


# ============================================================
# TEMPORAL FEATURES
# ============================================================

def create_temporal_features(df, date_column=None):

    result = df.copy()

    if date_column is None:

        detected = detect_date_columns(result)

        if not detected:
            raise ValueError(
                "No date column could be detected."
            )

        date_column = detected[0]

    if date_column not in result.columns:

        raise ValueError(
            f"Date column '{date_column}' does not exist."
        )

    result["crime_date"] = pd.to_datetime(
        result[date_column],
        errors="coerce",
        format="mixed"
    )

    result["year"] = result["crime_date"].dt.year
    result["month"] = result["crime_date"].dt.month
    result["day"] = result["crime_date"].dt.day
    result["day_of_week"] = result["crime_date"].dt.dayofweek

    result["is_weekend"] = (
        result["day_of_week"] >= 5
    ).astype(int)

    result["hour"] = result["crime_date"].dt.hour

    return result


# ============================================================
# CRIME FEATURES
# ============================================================

def create_crime_features(df):

    result = df.copy()

    if "crime_date" not in result.columns:

        raise ValueError(
            "'crime_date' must exist before "
            "creating crime features."
        )

    if "beat" not in result.columns:

        raise ValueError(
            "'beat' column is required."
        )

    result = result.dropna(
        subset=[
            "crime_date",
            "beat"
        ]
    ).copy()

    # --------------------------------------------------------
    # Numeric coordinates
    # --------------------------------------------------------

    if "latitude" in result.columns:

        result["latitude"] = pd.to_numeric(
            result["latitude"],
            errors="coerce"
        )

    if "longitude" in result.columns:

        result["longitude"] = pd.to_numeric(
            result["longitude"],
            errors="coerce"
        )

    # --------------------------------------------------------
    # Violent crime
    # --------------------------------------------------------

    violent_keywords = [
        "ASSAULT",
        "BATTERY",
        "ROBBERY",
        "HOMICIDE",
        "KIDNAPPING",
        "CRIM SEXUAL ASSAULT"
    ]

    if "primary_type" in result.columns:

        result["_is_violent"] = (
            result["primary_type"]
            .astype(str)
            .str.upper()
            .str.contains(
                "|".join(violent_keywords),
                na=False
            )
            .astype(int)
        )

    else:

        result["_is_violent"] = 0

    # --------------------------------------------------------
    # Night indicator
    # --------------------------------------------------------

    result["_is_night"] = (
        (result["hour"] < 6)
        |
        (result["hour"] >= 22)
    ).astype(int)

    # Aggregate crimes by calendar day, not exact timestamp.
    # The hour feature has already been extracted and is preserved.
    result["crime_date"] = result["crime_date"].dt.normalize()

    # --------------------------------------------------------
    # Daily beat-level aggregation
    # --------------------------------------------------------

    daily = (
        result
        .groupby(
            [
                "beat",
                "crime_date"
            ]
        )
        .agg(
            daily_crime_count=(
                "crime_date",
                "size"
            ),

            violent_crime_count=(
                "_is_violent",
                "sum"
            ),

            avg_latitude=(
                "latitude",
                "mean"
            ),

            avg_longitude=(
                "longitude",
                "mean"
            ),

            hour_mean=(
                "hour",
                "mean"
            ),

            weekend_ratio=(
                "is_weekend",
                "mean"
            ),

            night_ratio=(
                "_is_night",
                "mean"
            )
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Complete calendar for every beat
    # --------------------------------------------------------

    start_date = daily["crime_date"].min()
    end_date = daily["crime_date"].max()

    dates = pd.date_range(
        start=start_date,
        end=end_date,
        freq="D"
    )

    beats = (
        daily["beat"]
        .dropna()
        .unique()
    )

    calendar = pd.MultiIndex.from_product(
        [
            beats,
            dates
        ],
        names=[
            "beat",
            "crime_date"
        ]
    ).to_frame(
        index=False
    )

    daily = calendar.merge(
        daily,
        on=[
            "beat",
            "crime_date"
        ],
        how="left"
    )

    # --------------------------------------------------------
    # Zero-crime days
    # --------------------------------------------------------

    daily["daily_crime_count"] = (
        daily["daily_crime_count"]
        .fillna(0)
    )

    daily["violent_crime_count"] = (
        daily["violent_crime_count"]
        .fillna(0)
    )

    # --------------------------------------------------------
    # Sort
    # --------------------------------------------------------

    daily = daily.sort_values(
        [
            "beat",
            "crime_date"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Calendar-aware rolling features
    # --------------------------------------------------------

    rolling_results = []

    for beat, group in daily.groupby(
        "beat",
        sort=False
    ):

        group = group.sort_values(
            "crime_date"
        ).copy()

        indexed = group.set_index(
            "crime_date"
        )

        group["crime_count_7d"] = (
            indexed[
                "daily_crime_count"
            ]
            .rolling(
                "7D",
                min_periods=1
            )
            .sum()
            .values
        )

        group["crime_count_30d"] = (
            indexed[
                "daily_crime_count"
            ]
            .rolling(
                "30D",
                min_periods=1
            )
            .sum()
            .values
        )

        rolling_results.append(
            group
        )

    daily = pd.concat(
        rolling_results,
        ignore_index=True
    )

    return daily


# ============================================================
# TEMPORAL CONSISTENCY
# ============================================================

def _temporal_consistency_for_group(group):

    group = group.sort_values(
        "crime_date"
    ).copy()

    counts = (
        group["daily_crime_count"]
        .astype(float)
    )

    # Historical median using only previous observations
    historical_median = (
        counts
        .shift(1)
        .expanding()
        .median()
    )

    previous_above_median = (
        counts.shift(1)
        >
        historical_median
    ).astype(float)

    temp = pd.DataFrame(
        {
            "crime_date":
                group["crime_date"].values,

            "above_median":
                previous_above_median.values
        }
    )

    temp = temp.set_index(
        "crime_date"
    )

    consistency = (
        temp["above_median"]
        .rolling(
            "7D",
            min_periods=1
        )
        .mean()
    )

    group["temporal_consistency"] = (
        consistency
        .fillna(0)
        .values
    )

    return group


def add_temporal_consistency(df):

    groups = []

    for _, group in df.groupby(
        "beat",
        sort=False
    ):

        groups.append(
            _temporal_consistency_for_group(
                group
            )
        )

    result = pd.concat(
        groups,
        ignore_index=True
    )

    result["temporal_consistency"] = (
        result["temporal_consistency"]
        .clip(0, 1)
        .fillna(0)
    )

    return result


# ============================================================
# NEXT-DAY TARGET
# ============================================================

def create_historical_features(df):

    result = df.copy()

    result = result.sort_values(
        [
            "beat",
            "crime_date"
        ]
    ).reset_index(
        drop=True
    )

    # --------------------------------------------------------
    # Historical threshold available at current day
    # --------------------------------------------------------

    result["risk_threshold"] = (

        result
        .groupby("beat")[
            "daily_crime_count"
        ]
        .transform(
            lambda x:
            x.shift(1)
            .expanding()
            .median()
        )
    )

    # --------------------------------------------------------
    # NEXT-DAY CRIME COUNT
    #
    # shift(-1) means tomorrow's crime count.
    # --------------------------------------------------------

    result["next_day_crime_count"] = (

        result
        .groupby("beat")[
            "daily_crime_count"
        ]
        .shift(-1)

    )

    # --------------------------------------------------------
    # NEXT-DAY HIGH-RISK TARGET
    #
    # Tomorrow is high-risk if tomorrow's crime count
    # exceeds the historical threshold available TODAY.
    #
    # This makes the prediction task:
    #
    #       Day t information
    #              ↓
    #       Predict Day t+1
    #
    # --------------------------------------------------------

    result["high_risk"] = np.where(

        result["next_day_crime_count"].isna()
        |
        result["risk_threshold"].isna(),

        np.nan,

        (
            result["next_day_crime_count"]
            >
            result["risk_threshold"]
        ).astype(int)

    )

    # --------------------------------------------------------
    # Temporal consistency
    # --------------------------------------------------------

    result = add_temporal_consistency(
        result
    )

    return result


# ============================================================
# SPATIAL SIGNAL
# ============================================================

def add_spatial_signal(
    train_df,
    test_df,
    eps=0.002,
    min_samples=10
):
    """
    Generate spatial-support signals using DBSCAN.

    DBSCAN is fitted only on valid training coordinates.
    Test rows receive spatial support when they fall within
    eps of a training core sample.

    Returns:
        train_df, test_df, fitted_dbscan_model

    If no valid training coordinates exist, the model returned
    is None and all spatial signals remain zero.
    """

    train_df = train_df.copy()
    test_df = test_df.copy()

    coordinate_columns = [
        "avg_latitude",
        "avg_longitude"
    ]

    # Ensure the expected coordinate columns exist.
    for column in coordinate_columns:
        if column not in train_df.columns:
            train_df[column] = np.nan

        if column not in test_df.columns:
            test_df[column] = np.nan

    train_coordinates = (
        train_df[coordinate_columns]
        .apply(pd.to_numeric, errors="coerce")
    )

    test_coordinates = (
        test_df[coordinate_columns]
        .apply(pd.to_numeric, errors="coerce")
    )

    # Reject invalid coordinates, including infinity.
    train_coordinates = train_coordinates.replace(
        [np.inf, -np.inf], np.nan
    )
    test_coordinates = test_coordinates.replace(
        [np.inf, -np.inf], np.nan
    )

    train_valid = train_coordinates.notna().all(axis=1)
    test_valid = test_coordinates.notna().all(axis=1)

    # Unknown/missing locations default to no spatial support.
    train_df["spatial_signal"] = 0
    test_df["spatial_signal"] = 0

    valid_train = train_coordinates.loc[train_valid].to_numpy()

    if len(valid_train) == 0:
        print(
            "WARNING: No valid training coordinates. "
            "Spatial signals remain zero."
        )
        return train_df, test_df, None

    dbscan = DBSCAN(
        eps=eps,
        min_samples=min_samples
    )

    dbscan.fit(valid_train)

    # --------------------------------------------------------
    # Training signals
    # --------------------------------------------------------

    train_indices = train_coordinates.index[train_valid]

    train_df.loc[train_indices, "spatial_signal"] = (
        dbscan.labels_ != -1
    ).astype(int)

    # --------------------------------------------------------
    # Test signals
    # --------------------------------------------------------
    # DBSCAN has no native predict() method. Use training core
    # samples only to determine whether a test point has support.
    # Test coordinates are never used to fit DBSCAN.

    core_samples = dbscan.components_

    valid_test = test_coordinates.loc[test_valid].to_numpy()
    test_indices = test_coordinates.index[test_valid]

    if len(core_samples) > 0:
        for position, point in enumerate(valid_test):
            distances = np.sqrt(
                np.sum((core_samples - point) ** 2, axis=1)
            )

            if np.any(distances <= eps):
                test_df.loc[
                    test_indices[position],
                    "spatial_signal"
                ] = 1

    # --------------------------------------------------------
    # Diagnostics
    # --------------------------------------------------------

    labels = dbscan.labels_
    non_noise_labels = labels[labels != -1]

    n_clusters = (
        len(np.unique(non_noise_labels))
        if len(non_noise_labels) > 0
        else 0
    )

    noise_rate = (
        float(np.mean(labels == -1))
        if len(labels) > 0
        else 0.0
    )

    print("\n" + "=" * 60)
    print("DBSCAN SPATIAL SIGNAL")
    print("=" * 60)
    print("DBSCAN eps:", eps)
    print("DBSCAN min_samples:", min_samples)
    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))
    print("Valid training coordinates:", int(train_valid.sum()))
    print("Valid testing coordinates:", int(test_valid.sum()))
    print("Training clusters:", n_clusters)
    print("Training noise rate:", round(noise_rate, 4))
    print(
        "Training spatial signal rate:",
        round(float(train_df["spatial_signal"].mean()), 4)
    )
    print(
        "Testing spatial signal rate:",
        round(float(test_df["spatial_signal"].mean()), 4)
    )
    print("=" * 60)

    # IMPORTANT: return the fitted DBSCAN model, not a statistics dict.
    return train_df, test_df, dbscan


# ============================================================
# SPATIAL STATISTICS
# ============================================================

def get_spatial_statistics(
    train_data,
    test_data,
    spatial_model
):
    """
    Summarize spatial signals and fitted training DBSCAN results.

    The returned keys match those consumed by model_training.py.
    """

    train_signal = train_data.get(
        "spatial_signal",
        pd.Series(0, index=train_data.index, dtype=float)
    )

    test_signal = test_data.get(
        "spatial_signal",
        pd.Series(0, index=test_data.index, dtype=float)
    )

    labels = getattr(spatial_model, "labels_", None)

    if labels is not None and len(labels) > 0:
        non_noise_labels = labels[labels != -1]

        train_clusters = (
            len(np.unique(non_noise_labels))
            if len(non_noise_labels) > 0
            else 0
        )

        train_noise_rate = float(np.mean(labels == -1))
    else:
        train_clusters = 0
        train_noise_rate = 0.0

    return {
        "train_spatial_signal_rate": (
            float(train_signal.mean())
            if len(train_signal)
            else 0.0
        ),
        "test_spatial_signal_rate": (
            float(test_signal.mean())
            if len(test_signal)
            else 0.0
        ),
        "train_clusters": int(train_clusters),
        "train_noise_rate": float(train_noise_rate),
    }
