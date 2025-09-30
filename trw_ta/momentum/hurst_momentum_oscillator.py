import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('oscillator', 'signal')
def hurst_momentum_oscillator(close: pd.Series, hurst_length: int = 20, momentum_length: int = 14, smoothing: int = 3) -> pd.DataFrame:
    """https://www.tradingview.com/script/Sc9Ls8Kx-Hurst-Momentum-Oscillator-AlphaNatt/"""
    # Input validation
    if not isinstance(close, pd.Series):
        raise TypeError("Close must be a pandas Series")
    if close.isna().any():
        raise ValueError("Close must not contain NaN values")
    if len(close) < max(hurst_length, momentum_length, 100):
        raise ValueError(f"Close must have at least {max(hurst_length, momentum_length, 100)} rows")
    if hurst_length < 10 or hurst_length > 50:
        raise ValueError("Hurst length must be between 10 and 50")
    if momentum_length < 5 or momentum_length > 30:
        raise ValueError("Momentum length must be between 5 and 30")
    if smoothing < 1 or smoothing > 10:
        raise ValueError("Smoothing must be between 1 and 10")

    # Hurst Exponent calculation
    def calc_hurst(src, length):
        hurst = pd.Series(0.0, index=src.index)
        for i in range(length, len(src)):
            total_range = 0.0
            for scale in range(2, min(length // 2 + 1, 11)):
                range_sum = 0.0
                count = 0
                for j in range(i - length + scale, i + 1):
                    if j >= 0 and j + scale <= len(src):
                        max_val = src.iloc[j:j+scale].max()
                        min_val = src.iloc[j:j+scale].min()
                        range_sum += max_val - min_val
                        count += 1
                avg_range = range_sum / count if count > 0 else 0
                stdev_val = src.iloc[max(0, i-length+1):i+1].std()
                if stdev_val > 0 and avg_range > 0:
                    total_range += np.log(avg_range / stdev_val) / np.log(scale)
            hurst.iloc[i] = total_range / 9 if total_range > 0 else 0.5
        hurst = hurst.clip(0, 1).fillna(0.5)
        return hurst

    # Calculate Hurst Exponent
    hurst = calc_hurst(close, hurst_length)

    # Calculate momentum
    momentum = (close / close.shift(momentum_length) - 1) * 100  # ROC

    # Combine Hurst with momentum
    hurst_multiplier = hurst * 2
    adjusted_momentum = momentum * hurst_multiplier

    # Normalize to oscillator range
    lookback = 100
    highest = adjusted_momentum.rolling(window=lookback, min_periods=1).max()
    lowest = adjusted_momentum.rolling(window=lookback, min_periods=1).min()
    range_val = highest - lowest
    normalized = ((adjusted_momentum - lowest) / range_val.where(range_val != 0, 1) - 0.5) * 200
    normalized = normalized.fillna(0)

    # Smooth with EMA
    cl = normalized.ewm(span=smoothing, adjust=False).mean()

    # Signal calculation
    sig = pd.Series(0, index=close.index)
    last_signal = 0
    for i in range(1, len(close)):
        if cl.iloc[i] > 0 and cl.iloc[i-1] <= 0:  # Crossover above 0
            last_signal = 1
        elif cl.iloc[i] < 0 and cl.iloc[i-1] >= 0:  # Crossunder below 0
            last_signal = -1
        sig.iloc[i] = last_signal
    sig = sig.fillna(method='ffill').fillna(0)

    return pd.DataFrame({
        'cl': cl,
        'sig': sig
    }, index=close.index)