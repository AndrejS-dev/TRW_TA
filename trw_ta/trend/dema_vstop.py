import pandas as pd
import numpy as np
import trw_ta as ta

def dema_vstop(high: pd.Series, low: pd.Series, close: pd.Series, dema_len: int = 30, vstop_len: int = 10, multiplier: float = 2.0) -> pd.Series:
    """https://www.tradingview.com/script/G3eTG18L-Dema-Vstop-viResearch/"""
    src = ta.dema(close, dema_len)
    atr_vals = ta.atr1(high, low, close, vstop_len)
    atr_mul = atr_vals * multiplier

    stop_list = []
    trend_list = []
    vii_list = []

    uptrend = True
    max_val = src.iloc[0]
    min_val = src.iloc[0]
    stop = src.iloc[0]
    vii = 0

    for i in range(len(src)):
        s = src.iloc[i]
        if np.isnan(s) or np.isnan(atr_mul.iloc[i]):
            stop_list.append(np.nan)
            trend_list.append(uptrend)
            vii_list.append(vii)
            continue

        max_val = max(max_val, s)
        min_val = min(min_val, s)

        if uptrend:
            stop = max(stop, max_val - atr_mul.iloc[i])
        else:
            stop = min(stop, min_val + atr_mul.iloc[i])

        prev_uptrend = uptrend
        uptrend = s >= stop

        # If trend direction changes
        if uptrend != prev_uptrend:
            max_val = s
            min_val = s
            stop = max_val - atr_mul.iloc[i] if uptrend else min_val + atr_mul.iloc[i]

        # Set persistent vii
        if uptrend and not (not uptrend):
            vii = 1
        elif not uptrend:
            vii = -1

        stop_list.append(stop)
        trend_list.append(uptrend)
        vii_list.append(vii)

    return pd.Series(vii_list, index=close.index)
