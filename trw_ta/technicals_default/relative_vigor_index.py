import pandas as pd
import numpy as np
from trw_ta import register_outputs

def swma(series: pd.Series) -> pd.Series:
    """
    Sine-weighted moving average (SWMA), as in TradingView's ta.swma.
    Equivalent to a simple weighted moving average with specific sine-like weights.
    In this implementation, we use a 4-period approximation as in many Pine examples:
    weights = [1, 2, 2, 1]
    """
    weights = [1, 2, 2, 1]
    return series.rolling(window=4).apply(lambda x: np.dot(x, weights) / sum(weights), raw=True)

@register_outputs('rvi', 'signal')
def relative_vigor_index(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, length: int = 10) -> pd.DataFrame:
    """
    Relative Vigor Index (RVGI)
    
    Parameters:
        open_, high, low, close (pd.Series): OHLC price series
        length (int): Lookback period (default = 10)
    
    Returns:
        pd.DataFrame: ['rvgi', 'signal'] columns
    """
    numerator = swma(close - open_)
    denominator = swma(high - low)

    rvgi_raw = numerator.rolling(length).sum() / denominator.rolling(length).sum()
    signal = swma(rvgi_raw)

    return pd.DataFrame({
        'rvgi': rvgi_raw,
        'signal': signal
    })
