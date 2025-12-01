import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('vpi')
def velocity_pressure_index(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    velocity_len: int = 14, pressure_len: int = 21, smooth_factor: int = 3,
    norm_len: int = 50
) -> pd.DataFrame:
    """https://www.tradingview.com/script/1jOINpEP-Velocity-Pressure-Index-AlphaNatt/"""
    typical = (high + low + close) / 3
    price_change = typical.diff()
    velocity = price_change.rolling(velocity_len).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) == velocity_len else np.nan, raw=True
    )
    vol_avg = volume.rolling(pressure_len).mean()
    bull_pressure = volume * np.where(close > open_, 1, 0.3)
    bear_pressure = volume * np.where(close < open_, 1, 0.3)
    bull_ratio = bull_pressure.rolling(pressure_len).mean() / vol_avg
    bear_ratio = bear_pressure.rolling(pressure_len).mean() / vol_avg
    pressure_diff = bull_ratio - bear_ratio
    raw_index = velocity * (1 + pressure_diff)
    smoothed = raw_index.ewm(span=smooth_factor, adjust=False).mean()
    highest = smoothed.rolling(norm_len).max()
    lowest = smoothed.rolling(norm_len).min()
    range_val = highest - lowest
    normalized = np.where(range_val != 0, ((smoothed - lowest) / range_val - 0.5) * 200, 0)

    return pd.DataFrame({
        "vpi": normalized,
    }, index=close.index)