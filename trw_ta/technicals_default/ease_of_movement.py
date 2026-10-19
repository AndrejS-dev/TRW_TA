import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('eom')
def ease_of_movement(high: pd.Series, low: pd.Series, volume: pd.Series, length: int = 14, div: int = 10000) -> pd.Series:
    """
    Ease of Movement.
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    volume : pd.Series
        Volume series.
    length : int, default 14
        Lookback period.
    div : int, default 10000
        Div.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    hl2 = (high + low) / 2
    return ta.sma(div * hl2.diff() * (high- low) / volume, length)