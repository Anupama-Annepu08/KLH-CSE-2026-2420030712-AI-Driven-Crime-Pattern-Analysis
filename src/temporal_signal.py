import numpy as np


def calculate_temporal_consistency(
    daily_crime_count,
    crime_count_7d,
    crime_count_30d
):
    """
    Calculate an experimental temporal-consistency signal.

    The signal measures how consistent recent crime activity is
    with the broader 30-day activity level.

    Inputs:
        daily_crime_count : current daily crime count
        crime_count_7d    : rolling 7-day crime count
        crime_count_30d   : rolling 30-day crime count

    Output:
        Value in [0, 1]

        1.0 -> highly consistent recent activity
        0.0 -> highly inconsistent recent activity

    NOTE:
        This is an operational signal for the GARUDA API.
        It is not claimed to reproduce an experimentally validated
        temporal-consistency definition until that definition is
        formally fixed and evaluated.
    """

    daily = max(float(daily_crime_count), 0.0)
    count_7d = max(float(crime_count_7d), 0.0)
    count_30d = max(float(crime_count_30d), 0.0)

    avg_7d = count_7d / 7.0
    avg_30d = count_30d / 30.0

    values = np.array(
        [daily, avg_7d, avg_30d],
        dtype=float
    )

    if not np.isfinite(values).all():
        return 0.0

    maximum = float(values.max())
    minimum = float(values.min())

    if maximum == 0:
        return 1.0

    consistency = 1.0 - (
        (maximum - minimum) / maximum
    )

    return float(
        np.clip(
            consistency,
            0.0,
            1.0
        )
    )