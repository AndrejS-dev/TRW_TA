import numpy as np
import pandas as pd
from trw_ta import register_outputs

@register_outputs('wave_up', 'wave_down')
def weis_wave_candle(
    open: pd.Series,
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    method: str = 'LazyBear Style',
    trend_period: int = 4,
    use_shadows: bool = True
) -> pd.DataFrame:
    """https://www.tradingview.com/script/SwRzdvoL/"""
    candle_size = np.abs(close - open) + (high - low) * 0.5 if use_shadows else np.abs(close - open)

    wave = pd.Series(np.nan, index=close.index)
    wave_total = pd.Series(np.nan, index=close.index)

    if method == 'LazyBear Style':
        mov = np.where(close > close.shift(1), 1, np.where(close < close.shift(1), -1, 0))
        
        trend = pd.Series(np.nan, index=close.index)

        for i in range(1, len(close)):
            if mov[i] != 0 and mov[i] != mov[i-1]:
                trend.iloc[i] = mov[i]
            else:
                trend.iloc[i] = trend.iloc[i-1] if not np.isnan(trend.iloc[i-1]) else 0

        def is_trending_func(window):
            if len(window) < trend_period:
                return 0
            return 1 if (window[-1] > window[0]) or (window[-1] < window[0]) else 0
        
        is_trending = close.rolling(window=trend_period, min_periods=1).apply(is_trending_func, raw=True).fillna(0).astype(bool)

        for i in range(1, len(close)):
            if trend.iloc[i] != wave.iloc[i-1] and is_trending.iloc[i]:
                wave.iloc[i] = trend.iloc[i]
            else:
                wave.iloc[i] = wave.iloc[i-1] if not np.isnan(wave.iloc[i-1]) else 0

    else:  # Impulse Trend method
        # Calculate direction strength
        dir_strength = (close > close.shift(1)).rolling(trend_period, min_periods=1).sum() - \
                       (close < close.shift(1)).rolling(trend_period, min_periods=1).sum()

        trend = pd.Series(np.nan, index=close.index)

        for i in range(len(close)):
            if dir_strength.iloc[i] > 0:
                trend.iloc[i] = 1
            elif dir_strength.iloc[i] < 0:
                trend.iloc[i] = -1
            else:
                trend.iloc[i] = trend.iloc[i-1] if not np.isnan(trend.iloc[i-1]) else 0

        for i in range(1, len(close)):
            if trend.iloc[i] != wave.iloc[i-1]:
                wave.iloc[i] = trend.iloc[i]
            else:
                wave.iloc[i] = wave.iloc[i-1] if not np.isnan(wave.iloc[i-1]) else 0

    for i in range(1, len(close)):
        if wave.iloc[i] == wave.iloc[i-1] and not np.isnan(wave.iloc[i-1]):
            wave_total.iloc[i] = wave_total.iloc[i-1] + candle_size.iloc[i]
        else:
            wave_total.iloc[i] = candle_size.iloc[i]

    wave_up = np.where(wave == 1, wave_total, 0.0)
    wave_down = np.where(wave == -1, wave_total, 0.0)

    return pd.DataFrame({
        'wave_up': wave_up,
        'wave_down': wave_down
    }, index=close.index)