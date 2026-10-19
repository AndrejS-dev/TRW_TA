import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('signal')
def adaptive_price_zone(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 21, threshold: float = 2.0, smooth: int = 5) -> pd.DataFrame:
    """
    Adaptive price zone.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/v/3VKZ57Sl/
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    length : int, default 21
        Lookback period.
    threshold : float, default 2.0
        Threshold.
    smooth : int, default 5
        Smoothing length.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    typical = (high + low + close) / 3
    volatility = typical.diff().abs().ewm(span=length, adjust=False).mean()
    ema_typical = typical.ewm(span=smooth, adjust=False).mean()
    upper_band = ema_typical + threshold * volatility
    lower_band = ema_typical - threshold * volatility

    signal = pd.Series(index=typical.index, dtype=int)
    signal.iloc[0] = 0  # initialize with neutral trend

    for i in range(1, len(typical)):
        if typical.iloc[i] > upper_band.iloc[i - 1]:
            signal.iloc[i] = 1
        elif typical.iloc[i] < lower_band.iloc[i - 1]:
            signal.iloc[i] = -1
        else:
            signal.iloc[i] = signal.iloc[i - 1]

    return pd.DataFrame({"signal": signal})
