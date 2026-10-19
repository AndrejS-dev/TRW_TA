import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('bandwidth')
def bollinger_bandwidth(source: pd.Series, length: int = 20, mult: float = 2.0) -> pd.Series:
    """
    Bollinger Bandwidth.
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    length : int, default 20
        Lookback period.
    mult : float, default 2.0
        Band or ATR multiplier.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    basis = ta.sma(source, length)
    dev = mult * ta.stdev(source, length)
    upper = basis + dev
    lower = basis - dev
    return ((upper - lower) / basis) * 100