import pandas as pd
from trw_ta import register_outputs

@register_outputs('k', 'd')
def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period_k: int = 14, smooth_k: int = 1, period_d: int = 3) -> pd.DataFrame:
    """
    Stochastic Oscillator.
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    period_k : int, default 14
        Period k.
    smooth_k : int, default 1
        Smooth k.
    period_d : int, default 3
        Period d.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    lowest_low = low.rolling(window=period_k).min()
    highest_high = high.rolling(window=period_k).max()

    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = raw_k.rolling(window=smooth_k).mean()
    d = k.rolling(window=period_d).mean()

    return pd.DataFrame({
        "k": k,
        "d": d
    })
