import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('ema', 'upper_band', 'lower_band', 'trend')
def ema_volatility_channel(close: pd.Series, ema_length: int = 21, smooth: int = 5,
                           volt_length: int = 21, threshold: float = 1.8) -> pd.DataFrame:
    """https://www.tradingview.com/v/xYgI9yQG/"""
    ema = close.ewm(span=ema_length, adjust=False).mean()
    volatility = close.diff().abs().ewm(span=volt_length, adjust=False).mean()

    smoothed_ema = ema.ewm(span=smooth, adjust=False).mean()
    
    upper_band = smoothed_ema + threshold * volatility
    lower_band = smoothed_ema - threshold * volatility

    trend = pd.Series(index=close.index, dtype=int)
    trend.iloc[0] = 0
    for i in range(1, len(close)):
        if close.iloc[i] > upper_band.shift(1).iloc[i]:
            trend.iloc[i] = 1
        elif close.iloc[i] < lower_band.shift(1).iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

    return pd.DataFrame({
        'ema': ema,
        'upper_band': upper_band,
        'lower_band': lower_band,
        'trend': trend
    })
