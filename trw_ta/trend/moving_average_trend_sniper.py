import pandas as pd
import numpy as np
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('signal', 'mats_line')
def moving_average_trend_sniper(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 30) -> pd.Series:
    """https://www.tradingview.com/script/iwEyBE2d-Moving-Average-Trend-Sniper-ChartPrime/"""
    # Input validation
    if not all(isinstance(s, pd.Series) for s in [high, low, close]):
        raise TypeError("Inputs must be pandas Series")
    if not (high.index.equals(low.index) and low.index.equals(close.index)):
        raise ValueError("Input Series must have the same index")
    if high.isna().any() or low.isna().any() or close.isna().any():
        raise ValueError("Input Series must not contain NaN values")
    if len(high) < max(length, 10):
        raise ValueError(f"Input Series must have at least {max(length, 10)} rows")
    if length < 2:
        raise ValueError("Length must be at least 2")

    def tra():
        atr = ta.ema(ta.tr(high, low, close), length).fillna(0)
        slope = (close - close.shift(10)) / (atr * 10).replace(0, 1)  # Avoid division by zero
        angle_rad = np.arctan2(slope, 1)
        degrees = angle_rad * 180 / np.pi
        source = pd.Series(np.where(degrees > 0, high, low)).rolling(window=2, min_periods=1).mean()
        return source.fillna(method='ffill')

    def mats(source, length):
        higher_high = (ta.highest(high, length).diff() > 0).fillna(False)
        lower_low = (ta.lowest(low, length).diff() < 0).fillna(False)
        time_constant = ((higher_high | lower_low).astype(float)).rolling(window=length, min_periods=1).mean() ** 2
        time_constant = time_constant.fillna(0)  # Ensure no NaN in time_constant
        smooth = pd.Series(np.zeros(len(source)), index=source.index)
        smooth.iloc[0] = source.iloc[0]  # Initialize first value
        for i in range(1, len(source)):
            smooth.iloc[i] = smooth.iloc[i-1] + time_constant.iloc[i] * (source.iloc[i] - smooth.iloc[i-1])
        wilders_period = length * 4 - 1
        atr = abs(smooth.diff()).fillna(0)
        ma_atr = ta.ema(atr, wilders_period).fillna(0)
        delta_fast_atr = ta.ema(ma_atr, wilders_period).fillna(0) * length * 0.4
        result = pd.Series(np.zeros(len(source)), index=source.index)
        result.iloc[0] = source.iloc[0]  # Initialize first value
        for i in range(1, len(source)):
            if smooth.iloc[i] > result.iloc[i-1]:
                result.iloc[i] = result.iloc[i-1] if smooth.iloc[i] - delta_fast_atr.iloc[i] < result.iloc[i-1] else smooth.iloc[i] - delta_fast_atr.iloc[i]
            else:
                result.iloc[i] = result.iloc[i-1] if smooth.iloc[i] + delta_fast_atr.iloc[i] > result.iloc[i-1] else smooth.iloc[i] + delta_fast_atr.iloc[i]
        return result

    source = tra()
    mats_line = mats(source, length)
    close_sma = close.rolling(window=2, min_periods=1).mean().fillna(close)
    signal = pd.Series(np.where(close_sma > mats_line, 1, -1), index=close.index)
    # Ensure signal is persistent by filling forward
    signal = signal.fillna(method='ffill').fillna(-1)  # Default to -1 for initial NaN

    return pd.DataFrame({
        'signal': signal,
        'mats_line': mats_line
    }, index=close.index)