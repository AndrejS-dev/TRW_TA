import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('ci')
def choppiness_index(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    high_low = high - low
    high_close = (high - close.shift(1)).abs()
    low_close = (low - close.shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)

    atr_sum = tr.rolling(window=length).sum()

    high_max = high.rolling(window=length).max()
    low_min = low.rolling(window=length).min()

    chop = 100 * np.log10(atr_sum / (high_max - low_min)) / np.log10(length)
    return chop
