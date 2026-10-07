import pandas as pd


def dataset_summary(df: pd.DataFrame) -> dict:
    """
    Return basic dataset information.
    """
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum())
    }


def feature_information(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return feature-level information.
    """
    return pd.DataFrame({
        "Feature": df.columns,
        "Data Type": [
            str(df[col].dtype)
            for col in df.columns
        ],
        "Missing Values": [
            int(df[col].isnull().sum())
            for col in df.columns
        ],
        "Unique Values": [
            int(df[col].nunique())
            for col in df.columns
        ]
    })


def numerical_features(df: pd.DataFrame) -> list:
    """
    Return numerical feature names.
    """
    return df.select_dtypes(
        include="number"
    ).columns.tolist()


def categorical_features(df: pd.DataFrame) -> list:
    """
    Return categorical feature names.
    """
    return df.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()