import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('nko')
def normalized_kama_osc(close: pd.Series, fast_length: int = 2, slow_length: int = 30, er_period: int = 10, normalization_lookback: int = 100) -> pd.Series:
    """https://www.tradingview.com/script/OwtiIzT3-Normalized-KAMA-Oscillator-Ikke-Omar/"""
    change = (close - close.shift(er_period)).abs()
    volatility = close.diff().abs().rolling(window=er_period).sum()
    er = change / volatility

    fast_sc = 2 / (fast_length + 1)
    slow_sc = 2 / (slow_length + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc

    ema_fast = close.ewm(span=fast_length, adjust=False).mean()
    kama = ema_fast + sc * (close - ema_fast)

    lowest = kama.rolling(window=normalization_lookback).min()
    highest = kama.rolling(window=normalization_lookback).max()
    normalized = (kama - lowest) / (highest - lowest) - 0.5

    nko = []
    last = 0
    for val in normalized:
        if val > 0:
            last = 1
        elif val < 0:
            last = -1
        nko.append(last)

    return pd.Series(nko, index=close.index)
