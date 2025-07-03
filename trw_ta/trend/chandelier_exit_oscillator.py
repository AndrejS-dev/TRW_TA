import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('ce_osc')
def chandelier_exit_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 22,
                                multiplier: float = 3.0, norm_smooth: int = 3,
                                mode: str = 'regular') -> pd.Series:
    """https://www.tradingview.com/v/4D43DcbR/"""
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=length).mean()

    highest_high = (close.rolling(window=length).max() + high.rolling(window=length).max()) / 2
    lowest_low = (close.rolling(window=length).min() + low.rolling(window=length).min()) / 2
    chand_long = highest_high - atr * multiplier
    chand_short = lowest_low + atr * multiplier

    dir_ = pd.Series(index=close.index, dtype=int)
    dir_.iloc[0] = 1
    for i in range(1, len(close)):
        if close.iloc[i] > chand_short.iloc[i]:
            dir_.iloc[i] = 1
        elif close.iloc[i] < chand_long.iloc[i]:
            dir_.iloc[i] = -1
        else:
            dir_.iloc[i] = dir_.iloc[i - 1]

    avg_chand = (chand_long + chand_short) / 2
    bull = close > avg_chand
    bear = close < avg_chand

    if mode == 'normalized':
        osc = pd.Series(0, index=close.index, dtype=float)
        max_ = pd.Series(np.nan, index=close.index)
        min_ = pd.Series(np.nan, index=close.index)

        for i in range(1, len(close)):
            if bull.iloc[i] and not bull.iloc[i - 1]:
                max_.iloc[i] = close.iloc[i]
            elif bear.iloc[i] and not bear.iloc[i - 1]:
                min_.iloc[i] = close.iloc[i]
            else:
                max_.iloc[i] = max(close.iloc[i], max_.iloc[i - 1]) if not pd.isna(max_.iloc[i - 1]) else close.iloc[i]
                min_.iloc[i] = min(close.iloc[i], min_.iloc[i - 1]) if not pd.isna(min_.iloc[i - 1]) else close.iloc[i]

            denominator = max_.iloc[i] - min_.iloc[i]
            if denominator != 0:
                osc.iloc[i] = (close.iloc[i] - min_.iloc[i]) / denominator * 100

        osc = osc.rolling(window=norm_smooth).mean()

    else:
        osc = ((close - chand_long) + (close - chand_short)) / 2
        osc = osc.rolling(window=norm_smooth).mean()

    return osc.fillna(0)
