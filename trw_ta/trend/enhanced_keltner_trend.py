import pandas as pd
import numpy as np
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('basis', ' trend')
def enhanced_keltner_trend(high: pd.Series, low: pd.Series, close: pd.Series,
                           ma_len: int = 11, atr_len: int = 9, atr_mult: float = 0.7,
                           use_atr: bool = True, ma_type: str = "EMA") -> pd.DataFrame:
    """
    Enhanced keltner trend.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/v/Brk1dlzM/
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    ma_len : int, default 11
        Lookback / smoothing period.
    atr_len : int, default 9
        Lookback / smoothing period.
    atr_mult : float, default 0.7
        Multiplier.
    use_atr : bool, default True
        Use atr.
    ma_type : str, default 'EMA'
        Moving-average type (e.g. SMA, EMA, WMA).
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    price_basis = ta.ma(close, ma_len, ma_type)

    if use_atr:
        channel_width = ta.atr1(high, low, close, atr_len)
    else:
        high_low_range = high.rolling(ma_len).max() - low.rolling(ma_len).min()
        channel_width = high_low_range

    mov_factor = ta.ma(channel_width, ma_len, ma_type) * atr_mult
    upper_band = price_basis + mov_factor
    lower_band = price_basis - mov_factor

    trend = []
    current_trend = -1  # initial state (bearish)

    for i in range(len(close)):
        if i == 0 or pd.isna(upper_band[i-1]) or pd.isna(lower_band[i-1]):
            trend.append(current_trend)
            continue

        if close[i] > upper_band[i-1]:
            current_trend = 1
        elif close[i] < lower_band[i-1]:
            current_trend = -1

        trend.append(current_trend)

    return pd.DataFrame({
        "price_basis": price_basis,
        "trend": trend
    })
