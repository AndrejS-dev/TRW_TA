import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('oscillator')
def pulsema_oscillator(close: pd.Series, base_ma_len: int = 50, pulse_ma_len: int = 20) -> pd.Series:
    """
    Pulsema oscillator.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/v/v0qLGdNp/
    
    Parameters
    ----------
    close : pd.Series
        Close price series.
    base_ma_len : int, default 50
        Lookback / smoothing period.
    pulse_ma_len : int, default 20
        Lookback / smoothing period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    base_ma = close.ewm(span=base_ma_len, adjust=False).mean()
    slope = base_ma.diff()

    bars_above = np.zeros(len(close), dtype=int)
    bars_below = np.zeros(len(close), dtype=int)
    trend_duration = np.zeros(len(close), dtype=int)

    for i in range(1, len(close)):
        if close.iloc[i] > base_ma.iloc[i]:
            bars_above[i] = bars_above[i - 1] + 1
            bars_below[i] = 0
        else:
            bars_below[i] = bars_below[i - 1] + 1
            bars_above[i] = 0

        trend_duration[i] = bars_above[i] if close.iloc[i] > base_ma.iloc[i] else -bars_below[i]

    osc_raw = trend_duration * slope * 100 * np.where(close > base_ma, 1, -1)
    osc_smoothed = pd.Series(osc_raw, index=close.index).rolling(window=pulse_ma_len).mean()

    return osc_smoothed
