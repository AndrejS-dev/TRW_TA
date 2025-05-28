import pandas as pd
import numpy as np
import trw_ta as ta

def volume_trend_swing_points(close: pd.Series, volume: pd.Series, x: int = 30, y: int = 30) -> pd.Series:
    """https://www.tradingview.com/script/oGM8qJqv-Volume-Trend-Swing-Points-viResearch/"""
    pvt = ta.price_volume_trend(close, volume)
    highest = pvt.rolling(window=x).max()
    lowest = pvt.rolling(window=y).min()

    vii = []
    last_vii = 0

    for i in range(len(pvt)):
        is_high = pvt.iloc[i] == highest.iloc[i]
        is_low = pvt.iloc[i] == lowest.iloc[i]

        if is_low:
            last_vii = -1
        elif is_high and not is_low:
            last_vii = 1

        vii.append(last_vii)

    return pd.Series(vii, index=close.index)
