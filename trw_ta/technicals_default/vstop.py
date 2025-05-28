import pandas as pd
import numpy as np
import trw_ta as ta

def vstop(src: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20, factor: float = 2.0) -> pd.DataFrame:
    atr_vals = ta.atr2(high, low, close, length)
    stop = [np.nan] * len(src)
    uptrend = [True] * len(src)

    max_price = src.iloc[0]
    min_price = src.iloc[0]
    trend = True

    for i in range(1, len(src)):
        if pd.isna(src.iloc[i]) or pd.isna(atr_vals.iloc[i]):
            stop[i] = np.nan
            uptrend[i] = trend
            continue

        atr_m = atr_vals.iloc[i] * factor

        # Update max/min depending on current trend
        if trend:
            max_price = max(max_price, src.iloc[i])
        else:
            min_price = min(min_price, src.iloc[i])

        prev_stop = stop[i - 1] if not pd.isna(stop[i - 1]) else src.iloc[i - 1]

        if trend:
            stop_val = max(prev_stop, max_price - atr_m)
        else:
            stop_val = min(prev_stop, min_price + atr_m)

        new_trend = src.iloc[i] >= stop_val
        stop[i] = stop_val
        uptrend[i] = new_trend

        # Flip logic: reset max/min and stop if trend changed
        if new_trend != trend:
            max_price = min_price = src.iloc[i]
            stop[i] = max_price - atr_m if new_trend else min_price + atr_m

        trend = new_trend

    return pd.DataFrame({
        'vstop': stop,
        'uptrend': uptrend
    }, index=src.index)
