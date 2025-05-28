import pandas as pd
import numpy as np

def coral_trend(close: pd.Series, smoothing: int = 21, constant_d: float = 0.4) -> pd.DataFrame:
    """https://www.tradingview.com/v/qyUwc2Al/"""
    src = close.copy()
    di = (smoothing - 1.0) / 2.0 + 1.0
    c1 = 2 / (di + 1.0)
    c2 = 1 - c1

    c3 = 3.0 * (constant_d ** 2 + constant_d ** 3)
    c4 = -3.0 * (2.0 * constant_d ** 2 + constant_d + constant_d ** 3)
    c5 = 3.0 * constant_d + 1.0 + constant_d ** 3 + 3.0 * constant_d ** 2

    # Recursive filters
    i1 = [np.nan] * len(src)
    i2 = [np.nan] * len(src)
    i3 = [np.nan] * len(src)
    i4 = [np.nan] * len(src)
    i5 = [np.nan] * len(src)
    i6 = [np.nan] * len(src)
    bfr = [np.nan] * len(src)
    direction = [0] * len(src)

    for i in range(len(src)):
        x = src.iloc[i]
        i1[i] = c1 * x + c2 * (i1[i - 1] if i > 0 else x)
        i2[i] = c1 * i1[i] + c2 * (i2[i - 1] if i > 0 else i1[i])
        i3[i] = c1 * i2[i] + c2 * (i3[i - 1] if i > 0 else i2[i])
        i4[i] = c1 * i3[i] + c2 * (i4[i - 1] if i > 0 else i3[i])
        i5[i] = c1 * i4[i] + c2 * (i5[i - 1] if i > 0 else i4[i])
        i6[i] = c1 * i5[i] + c2 * (i6[i - 1] if i > 0 else i5[i])

        bfr[i] = -constant_d ** 3 * i6[i] + c3 * i5[i] + c4 * i4[i] + c5 * i3[i]

        if i > 0:
            if bfr[i] > bfr[i - 1]:
                direction[i] = 1
            elif bfr[i] < bfr[i - 1]:
                direction[i] = -1
            else:
                direction[i] = direction[i - 1]  # preserve previous trend
        else:
            direction[i] = 0

    return pd.DataFrame({
        'bfr': bfr,
        'trend_direction': direction
    }, index=close.index)
