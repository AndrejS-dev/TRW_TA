import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('bbp')
def bull_bear_power(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 13) -> pd.Series:
    """
    Bulls/Bears Power.
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    length : int, default 13
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    bullPower = high - ta.ema(close, length)
    bearPower = low - ta.ema(close, length)
    return bullPower + bearPower