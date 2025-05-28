import pandas as pd
import numpy as np
import trw_ta as ta

def enhanced_keltner_trend(high: pd.Series, low: pd.Series, close: pd.Series,
                           ma_len: int = 11, atr_len: int = 9, atr_mult: float = 0.7,
                           use_atr: bool = True, ma_type: str = "EMA") -> pd.DataFrame:
    """https://www.tradingview.com/v/Brk1dlzM/"""
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
