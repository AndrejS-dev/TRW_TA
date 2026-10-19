import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('hv')
def historical_volatility(close: pd.Series, length: int = 10, timeframe: str = 'daily') -> pd.Series:
    """
    Historical Volatility.
    
    Parameters
    ----------
    close : pd.Series
        Close price series.
    length : int, default 10
        Lookback period.
    timeframe : str, default 'daily'
        Candle timeframe (e.g. 1h, 1d).
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    annual = 365
    timeframe = timeframe.lower()
    
    if timeframe in ['1d', 'daily', 'intraday']:
        per = 1
    else:
        per = 7  # e.g., weekly or anything longer

    log_returns = np.log(close / close.shift(1))
    stdev = log_returns.rolling(window=length).std()
    hv = 100 * stdev * np.sqrt(annual / per)
    return hv
