"""Client helpers for the GARUDA FastAPI inference service."""

import requests


def check_api_health(api_url: str, timeout: int = 5):
    """Return the FastAPI health response or an error message."""
    try:
        response = requests.get(
            f"{api_url.rstrip('/')}/health",
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json(), None
    except Exception as error:
        return None, str(error)


def predict_risk(api_url: str, payload: dict, timeout: int = 30):
    """Send a GARUDA risk-prediction request to FastAPI."""
    try:
        response = requests.post(
            f"{api_url.rstrip('/')}/predict",
            json=payload,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.HTTPError as error:
        detail = None
        try:
            detail = error.response.json()
        except Exception:
            detail = error.response.text if error.response is not None else str(error)
        return None, str(detail)
    except Exception as error:
        return None, str(error)
