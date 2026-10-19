import pandas as pd
import numpy as np
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('osc')
def ttm_squeeze_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
    """
    Ttm squeeze oscillator.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/v/eFs5kf8F/
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    length : int, default 20
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    e1 = (high.rolling(length).max() + low.rolling(length).min()) / 2 + close.rolling(length).mean()
    osc_input = close - e1 / 2
    osc = ta.linear_regression(osc_input, length)

    return osc
