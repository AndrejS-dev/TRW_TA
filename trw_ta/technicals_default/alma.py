import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('alma')
def alma(source: pd.Series, window: int = 9, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """Arnaud Legoux Moving Average (ALMA)"""
    m = offset * (window - 1)
    s = window / sigma
    weights = np.array([
        np.exp(-((i - m) ** 2) / (2 * s ** 2))
        for i in range(window)
    ])
    weights /= weights.sum()  # Normalize

    return source.rolling(window=window).apply(lambda x: np.dot(x, weights), raw=True)