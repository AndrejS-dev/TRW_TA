import numpy as np
import pandas as pd
from trw_ta import register_outputs

@register_outputs('tfo')
def trendflex_oscillator(series: pd.Series, period_essf: float = 7.5, period_tflx: int = 20, period_frms: float = 33.0) -> pd.Series:
    """https://www.tradingview.com/v/8ZSALctc/"""
    sqrt2pi = np.sqrt(2.0) * np.pi
    alpha = sqrt2pi / period_essf
    beta = np.exp(-alpha)
    coef2 = -beta ** 2
    coef1 = 2.0 * beta * np.cos(alpha)
    coef0 = 1.0 - coef1 - coef2

    sma2 = (series + series.shift(1).fillna(series)) * 0.5
    essf = pd.Series(np.zeros(len(series)), index=series.index)
    for i in range(2, len(series)):
        essf.iloc[i] = (coef0 * sma2.iloc[i] +
                        coef1 * essf.iloc[i - 1] +
                        coef2 * essf.iloc[i - 2])

    sum_diffs = pd.Series(np.zeros(len(series)), index=series.index)
    for i in range(period_tflx, len(series)):
        s = 0.0
        for j in range(1, period_tflx + 1):
            s += essf.iloc[i] - essf.iloc[i - j]
        sum_diffs.iloc[i] = s / period_tflx

    squared = sum_diffs ** 2
    frms = squared.ewm(span=period_frms, adjust=False).mean().apply(np.sqrt)

    with np.errstate(divide='ignore', invalid='ignore'):
        tfo = sum_diffs / frms.replace(0, np.nan)

    return tfo.fillna(0)
