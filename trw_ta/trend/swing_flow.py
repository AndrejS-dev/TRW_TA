import pandas as pd
import numpy as np
from trw_ta import register_outputs


@register_outputs('trend')
def swing_flow(high: pd.Series, low: pd.Series, close: pd.Series, len_period: int = 10, 
               band_mult: float = 2.0, volatility_length: int = 50) -> pd.DataFrame:
    """https://www.tradingview.com/script/BtGC9TGm-Swing-Flow-Indicator-ChartPrime/"""
    vol_proxy = (high.rolling(volatility_length, min_periods=volatility_length//2)
                 .max() - low.rolling(volatility_length, min_periods=volatility_length//2).min())
    volatility = vol_proxy * band_mult

    window = 2 * len_period + 1
    is_pivot_high = high == high.rolling(window, min_periods=window//2 + 1).max()
    is_pivot_low  = low  == low.rolling(window, min_periods=window//2 + 1).min()

    swing_high = pd.Series(np.nan, index=high.index, dtype=float)
    swing_low  = pd.Series(np.nan, index=low.index, dtype=float)

    last_h = high.iloc[0] if not high.empty else np.nan
    last_l = low.iloc[0]  if not low.empty  else np.nan

    for i in range(len(high)):
        if is_pivot_high.iloc[i] and pd.notna(high.iloc[i]):
            last_h = high.iloc[i]
        swing_high.iloc[i] = last_h

        if is_pivot_low.iloc[i] and pd.notna(low.iloc[i]):
            last_l = low.iloc[i]
        swing_low.iloc[i] = last_l

    mid_raw = (swing_high + swing_low) / 2
    mid = mid_raw.rolling(len_period, min_periods=1).mean()
    upper_raw = mid + volatility.fillna(0)
    lower_raw = mid - volatility.fillna(0)
    upper = upper_raw.rolling(len_period * 2, min_periods=1).mean()
    lower = lower_raw.rolling(len_period * 2, min_periods=1).mean()

    epsilon = 1e-8
    cross_up = (close > upper + epsilon) & (close.shift(1) <= upper.shift(1) + epsilon)
    cross_dn = (close < lower - epsilon) & (close.shift(1) >= lower.shift(1) - epsilon)

    trend = pd.Series(-1, index=close.index, dtype='int8')  # start bearish by default

    for i in range(1, len(close)):
        if cross_up.iloc[i]:
            trend.iloc[i] = 1
        elif cross_dn.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i-1]

    changed = trend != trend.shift(1).fillna(-1)
    swing_up_signal   = changed & cross_up & (trend == 1)
    swing_down_signal = changed & cross_dn & (trend == -1)

    return pd.DataFrame({'trend': trend}).rename_axis(None, axis=1)