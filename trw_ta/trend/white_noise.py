import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('wm')
def white_noise(source: pd.Series, noise_length: int = 20, use_normalization: bool = True, normalization_lookback: int = 100) -> pd.Series:
    """https://www.tradingview.com/script/S6qTswV5-White-Noise/"""
    ma = source.rolling(window=noise_length).mean()
    dist = (source - ma) / ma

    if use_normalization:
        lowest = dist.rolling(window=normalization_lookback).min()
        highest = dist.rolling(window=normalization_lookback).max()

        normalized = (dist - lowest) / (highest - lowest) - 0.5
        wn = np.where(normalized < 0, -1, 1)
    else:
        wn = np.where(dist < 0, -1, 1)

    return pd.Series(wn, index=source.index)
