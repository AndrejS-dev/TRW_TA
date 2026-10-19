import trw_ta as ta
import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('ao')
def awesome_oscillator(high: pd.Series, low: pd.Series, shorter_len: int = 5, longer_len: int = 34) -> pd.Series:
    """
    Awesome Oscillator.
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    shorter_len : int, default 5
        Lookback / smoothing period.
    longer_len : int, default 34
        Lookback / smoothing period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    hl2 = (high + low) / 2
    ao = ta.sma(hl2, shorter_len) - ta.sma(hl2, longer_len)
    return pd.DataFrame(ao)
