import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('cc')
def coppock_curve(close: pd.Series, wma_length: int = 10, long_roc_length: int = 14, short_roc_length: int = 11) -> pd.Series:
    # Rate of Change (RoC)
    """
    Coppock Curve.
    
    Parameters
    ----------
    close : pd.Series
        Close price series.
    wma_length : int, default 10
        Lookback / smoothing period.
    long_roc_length : int, default 14
        Lookback / smoothing period.
    short_roc_length : int, default 11
        Lookback / smoothing period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    roc_long = close.pct_change(periods=long_roc_length) * 100
    roc_short = close.pct_change(periods=short_roc_length) * 100
    roc_sum = roc_long + roc_short

    coppock = ta.wma(roc_sum, wma_length)
    return coppock
