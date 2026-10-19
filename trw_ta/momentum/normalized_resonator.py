import pandas as pd
import numpy as np
from trw_ta import register_outputs


@register_outputs('oscillator', 'signal')
def normalized_resonator(source: pd.Series, center_period: int = 100, bandwidth: float = 0.5,
    lookback_multiplier: float = 1.0, signal_length: int = 9) -> pd.DataFrame:
    """
    Normalized resonator.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/script/vQzh7vky-Normalized-Resonator-LuxAlgo/
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    center_period : int, default 100
        Lookback / smoothing period.
    bandwidth : float, default 0.5
        Bandwidth.
    lookback_multiplier : float, default 1.0
        Multiplier.
    signal_length : int, default 9
        Lookback / smoothing period.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    n = len(source)
    if n < 3:
        return pd.DataFrame(index=source.index, columns=['oscillator', 'signal'])

    pi = np.pi
    omega = 2 * pi / center_period
    alpha = np.tan(pi * bandwidth / center_period)
    beta  = np.cos(omega)
    r     = 1.0 / (1.0 + alpha)

    c1   = 2 * r * beta
    c2   = -(2 * r - 1)
    gain = alpha * r

    bp = pd.Series(np.nan, index=source.index, dtype=float)

    for i in range(2, n):
        term1 = gain * (source.iloc[i] - source.iloc[i - 2])
        term2 = c1 * bp.iloc[i - 1] if not pd.isna(bp.iloc[i - 1]) else 0.0
        term3 = c2 * bp.iloc[i - 2] if not pd.isna(bp.iloc[i - 2]) else 0.0
        bp.iloc[i] = term1 + term2 + term3

    lookback = max(1, int(center_period * lookback_multiplier))
    peak = bp.abs().rolling(lookback, min_periods=1).max()
    oscillator = bp / peak.where(peak != 0, np.nan)
    oscillator = oscillator.fillna(0.0)  # avoid division-by-zero artifacts

    signal = oscillator.ewm(span=signal_length, adjust=False, min_periods=1).mean()

    return pd.DataFrame({
        'oscillator': oscillator,
        'signal':     signal,
    }).rename_axis(None, axis=1)