import numpy as np
import pandas as pd
import trw_ta
from trw_ta import register_outputs

def stoch_k(series: pd.Series, length: int):
    lowest = series.rolling(length).min()
    highest = series.rolling(length).max()
    return 100 * (series - lowest) / (highest - lowest)

@register_outputs('zscore', 'smoothed_scaled', 'ltm', 'momentum_shift')
def stochastic_zscore(source: pd.Series, length: int = 21):
    """https://www.tradingview.com/v/1k7HqCpx/"""
    basis = source.rolling(length).mean()
    stdev = source.rolling(length).std()
    zscore = (source - basis) / stdev

    stochZ = stoch_k(zscore, length)
    scaledSZ = stochZ / 25 - 2
    smoothed_scaled = trw_ta.hma(scaledSZ, length)
    ltm = trw_ta.alma(zscore, length, 0, 0.1)

    smoothed_prev = smoothed_scaled.shift(1)
    momentum_shift = smoothed_scaled > smoothed_prev

    return pd.DataFrame({
        'zscore': zscore,
        'smoothed_scaled': smoothed_scaled,
        'ltm': ltm,
        'momentum_shift': momentum_shift.astype(int),
    })
