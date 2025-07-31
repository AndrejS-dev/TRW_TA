import pandas as pd
import numpy as np
import trw_ta
from trw_ta import register_outputs

@register_outputs('signal')
def sma_trend_spectrum(source: pd.Series, from_: int = 2, by_: int = 1,
                            signal_smooth: str = 'SMA', smooth_len: int = 14) -> pd.Series:
    """https://www.tradingview.com/script/vgnEQWlK-SMA-Trend-Spectrum-InvestorUnknown/"""
    n_trends = 31
    trend_values = []

    def f_trend(x1, x2, x3):
        min_ = trw_ta.sma(source, x1)
        mid = trw_ta.sma(source, x2)
        max_ = trw_ta.sma(source, x3)
        condition_all = (source > min_) & (source > mid) & (source > max_)
        condition_any = (source > min_) | (source > mid) | (source > max_)
        return np.where(condition_all, 1.0, np.where(condition_any, 0.5, 0.0))

    for x in range(n_trends):
        x1 = from_ + (0 + by_ * x * 3)
        x2 = from_ + (1 + by_ * x * 3)
        x3 = from_ + (2 + by_ * x * 3)
        trend_values.append(f_trend(x1, x2, x3))

    trend_stack = np.vstack(trend_values)
    avg_trend = np.mean(trend_stack, axis=0)
    avg_trend_series = pd.Series(avg_trend, index=source.index)

    if signal_smooth.upper() == 'RAW':
        signal = avg_trend_series
    elif signal_smooth.upper() == 'SMA':
        signal = trw_ta.sma(avg_trend_series, smooth_len)
    elif signal_smooth.upper() == 'EMA':
        signal = trw_ta.ema(avg_trend_series, smooth_len)
    else:
        raise ValueError("signal_smooth must be one of ['RAW', 'SMA', 'EMA']")

    return signal
