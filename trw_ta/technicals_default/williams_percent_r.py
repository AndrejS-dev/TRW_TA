import pandas as pd
from trw_ta import register_outputs

@register_outputs('wpr')
def williams_percent_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """
    Williams %R.
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    length : int, default 14
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    highest_high = high.rolling(window=length).max()
    lowest_low = low.rolling(window=length).min()
    percent_r = 100 * (close - highest_high) / (highest_high - lowest_low)
    return percent_r
