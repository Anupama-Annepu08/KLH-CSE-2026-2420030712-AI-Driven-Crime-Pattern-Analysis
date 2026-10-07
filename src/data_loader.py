from pathlib import Path
import pandas as pd


def load_csv(file_path: Path) -> pd.DataFrame:
    """
    Load a CSV dataset.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    return pd.read_csv(file_path)