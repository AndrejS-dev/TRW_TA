import pandas as pd
import numpy as np
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('line', 'signal')
def follow_line(high: pd.Series, low: pd.Series, close: pd.Series, atr_period: int = 5, bb_period: int = 21,
                bb_deviation: float = 1.0, use_atr_filter: bool = True) -> pd.DataFrame:
    """https://www.tradingview.com/v/UXKo4RaJ/"""
    sma_close = ta.ma(close, bb_period, 'SMA')
    std = ta.stdev(close, bb_period)
    bb_upper = sma_close + bb_deviation * std
    bb_lower = sma_close - bb_deviation * std

    atr_val = ta.atr2(high, low, close, atr_period)

    follow_line = pd.Series(np.nan, index=close.index)
    i_trend = pd.Series(0, index=close.index)
    signal = pd.Series(0, index=close.index)

    persistent_signal = 0

    for i in range(1, len(close)):
        bb_signal = 0
        if close.iloc[i] > bb_upper.iloc[i]:
            bb_signal = 1
        elif close.iloc[i] < bb_lower.iloc[i]:
            bb_signal = -1

        prev_fl = follow_line.iloc[i - 1]

        if bb_signal == 1:
            new_fl = low.iloc[i] - atr_val.iloc[i] if use_atr_filter else low.iloc[i]
            follow_line.iloc[i] = max(new_fl, prev_fl) if not np.isnan(prev_fl) else new_fl
        elif bb_signal == -1:
            new_fl = high.iloc[i] + atr_val.iloc[i] if use_atr_filter else high.iloc[i]
            follow_line.iloc[i] = min(new_fl, prev_fl) if not np.isnan(prev_fl) else new_fl
        else:
            follow_line.iloc[i] = prev_fl

        if not np.isnan(follow_line.iloc[i]) and not np.isnan(prev_fl):
            if follow_line.iloc[i] > prev_fl:
                i_trend.iloc[i] = 1
            elif follow_line.iloc[i] < prev_fl:
                i_trend.iloc[i] = -1
            else:
                i_trend.iloc[i] = i_trend.iloc[i - 1]
        else:
            i_trend.iloc[i] = i_trend.iloc[i - 1]

        if i_trend.iloc[i - 1] == -1 and i_trend.iloc[i] == 1:
            persistent_signal = 1
        elif i_trend.iloc[i - 1] == 1 and i_trend.iloc[i] == -1:
            persistent_signal = -1

        signal.iloc[i] = persistent_signal

    return pd.DataFrame({
        "line": follow_line,
        "signal": signal
    })
