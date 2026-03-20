import pandas as pd
import numpy as np
from trw_ta import register_outputs


def kvo_approx(hlc3: pd.Series, volume: pd.Series, fast: int = 34, slow: int = 55) -> pd.Series:
    trend = hlc3.diff()
    sv = volume * np.sign(trend)

    ema_fast = sv.ewm(span=fast, adjust=False, min_periods=fast).mean()
    ema_slow = sv.ewm(span=slow, adjust=False, min_periods=slow).mean()

    return ema_fast - ema_slow


def moving_average(src: pd.Series, length: int, ma_type: str = "SMA") -> pd.Series:
    if ma_type == "SMA":
        return src.rolling(length, min_periods=1).mean()
    elif ma_type == "EMA":
        return src.ewm(span=length, adjust=False, min_periods=1).mean()
    elif ma_type == "WMA":
        weights = np.arange(1, length + 1)
        return src.rolling(length, min_periods=1).apply(
            lambda x: np.dot(x, weights[-len(x):]) / weights[-len(x):].sum(), raw=True
        )
    elif ma_type == "HMA":
        wma_half = src.rolling(length // 2, min_periods=1).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True
        )
        wma_full = src.rolling(length, min_periods=1).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True
        )
        raw_hma = 2 * wma_half - wma_full
        return raw_hma.rolling(int(np.sqrt(length)), min_periods=1).apply(
            lambda x: np.dot(x, np.arange(1, len(x) + 1)) / np.arange(1, len(x) + 1).sum(), raw=True
        )
    else:
        return src.rolling(length, min_periods=1).mean()


@register_outputs('osc', 'sma')
def normalised_volume_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    fast_length: int = 34, slow_length: int = 55, norm_period: int = 50, ma_type: str = "SMA",
    ma_length: int = 15) -> pd.DataFrame:
    """https://www.tradingview.com/script/PobTVron-Normalised-Volume-Oscillator-BackQuant/"""
    hlc3 = (high + low + close) / 3

    kvo = kvo_approx(hlc3, volume, fast=fast_length, slow=slow_length)

    kvo_high = kvo.rolling(norm_period, min_periods=1).max()
    kvo_low  = kvo.rolling(norm_period, min_periods=1).min()
    range_   = kvo_high - kvo_low
    normalized = ((kvo - kvo_low) / range_.replace(0, np.nan)) - 0.5

    sig_ma = moving_average(normalized, length=ma_length, ma_type=ma_type)

    return pd.DataFrame({
        'osc': normalized,
        'sma': sig_ma,
    }).rename_axis(None, axis=1)