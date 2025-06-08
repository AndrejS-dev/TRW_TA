import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('sar')
def sar(high: pd.Series, low: pd.Series, close: pd.Series, start=0.02, inc=0.02, max_val=0.2) -> pd.Series:
    sar = np.full(len(close), np.nan)
    max_min = np.nan
    acc = np.nan
    is_below = None

    for i in range(1, len(close)):
        if i == 1:
            if close[i] > close[i-1]:
                is_below = True
                max_min = high[i]
                sar[i] = low[i-1]
            else:
                is_below = False
                max_min = low[i]
                sar[i] = high[i-1]
            acc = start
            continue

        prev_sar = sar[i-1]
        sar[i] = prev_sar + acc * (max_min - prev_sar)

        is_first_trend_bar = False

        if is_below:
            if sar[i] > low[i]:
                is_below = False
                is_first_trend_bar = True
                sar[i] = max(high[i], max_min)
                max_min = low[i]
                acc = start
        else:
            if sar[i] < high[i]:
                is_below = True
                is_first_trend_bar = True
                sar[i] = min(low[i], max_min)
                max_min = high[i]
                acc = start

        if not is_first_trend_bar:
            if is_below and high[i] > max_min:
                max_min = high[i]
                acc = min(acc + inc, max_val)
            elif not is_below and low[i] < max_min:
                max_min = low[i]
                acc = min(acc + inc, max_val)

        # Apply the max/min logic from Pine
        if is_below:
            sar[i] = min(sar[i], low[i-1])
            if i > 1:
                sar[i] = min(sar[i], low[i-2])
        else:
            sar[i] = max(sar[i], high[i-1])
            if i > 1:
                sar[i] = max(sar[i], high[i-2])

    return pd.Series(sar, index=close.index)
