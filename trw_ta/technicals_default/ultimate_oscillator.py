import pandas as pd
from trw_ta import register_outputs

@register_outputs('uo')
def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, length1: int = 7, length2: int = 14, length3: int = 28) -> pd.Series:
    """
    Ultimate Oscillator.
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    length1 : int, default 7
        Length1.
    length2 : int, default 14
        Length2.
    length3 : int, default 28
        Length3.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    prev_close = close.shift(1)
    high_ = high.combine(prev_close, max)
    low_ = low.combine(prev_close, min)

    bp = close - low_
    tr_ = high_ - low_

    def avg(bp_, tr_, length):
        return bp_.rolling(length).sum() / tr_.rolling(length).sum()

    avg1 = avg(bp, tr_, length1)
    avg2 = avg(bp, tr_, length2)
    avg3 = avg(bp, tr_, length3)

    uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
    return uo
