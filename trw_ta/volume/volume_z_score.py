import pandas as pd
import numpy as np
from trw_ta import register_outputs
from trw_ta.ta_core import valid_data_prefix

@register_outputs('zs', 'signal')
def volume_z_score(volume: pd.Series, lookback: int = 50, use_log: bool = False, sma_type: str = "SMA", 
                  smoothing: int = 1, z_thresh_high: float = 2.0, z_thresh_low: float = -2.0, eps: float = 1e-9) -> pd.DataFrame:
    """
    Volume z score.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/script/wh1wA9zV-Normalized-Volume-Z-Score/
    
    Parameters
    ----------
    volume : pd.Series
        Volume series.
    lookback : int, default 50
        Lookback.
    use_log : bool, default False
        Use log.
    sma_type : str, default 'SMA'
        Sma type.
    smoothing : int, default 1
        Smoothing.
    z_thresh_high : float, default 2.0
        Z thresh high.
    z_thresh_low : float, default -2.0
        Z thresh low.
    eps : float, default 1e-09
        Eps.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    # Input validation
    if not isinstance(volume, pd.Series):
        raise TypeError("Volume must be a pandas Series")
    (volume,), full_index = valid_data_prefix(volume)
    if volume.le(0).any() and use_log:
        raise ValueError("Volume must be positive when use_log is True")
    if len(volume) < lookback:
        raise ValueError(f"Volume must have at least {lookback} non-NaN rows")
    if lookback < 2:
        raise ValueError("Lookback must be at least 2")
    if smoothing < 1:
        raise ValueError("Smoothing must be at least 1")
    if sma_type not in ["SMA", "EMA"]:
        raise ValueError("sma_type must be 'SMA' or 'EMA'")

    # Prepare volume series
    vol = np.log(volume + 1) if use_log else volume

    # Calculate rolling mean
    if sma_type == "SMA":
        mean = vol.rolling(window=lookback, min_periods=1).mean()
    else:  # EMA
        mean = vol.ewm(span=lookback, adjust=False).mean()

    # Calculate rolling standard deviation
    std = vol.rolling(window=lookback, min_periods=1).std()

    # Calculate Z-score
    z = (vol - mean) / (std + eps)

    # Apply smoothing
    zs = z.rolling(window=smoothing, min_periods=1).mean() if smoothing > 1 else z
    zs = zs.fillna(0)

    # Signal calculation
    sig = pd.Series(0, index=volume.index)
    last_signal = 0
    for i in range(1, len(volume)):
        if zs.iloc[i] >= z_thresh_high and zs.iloc[i-1] < z_thresh_high:  # Crossover above zThreshHigh
            last_signal = 1
        elif zs.iloc[i] <= z_thresh_low and zs.iloc[i-1] > z_thresh_low:  # Crossunder below zThreshLow
            last_signal = -1
        else:
            last_signal = 0  # Reset to 0 when not crossing thresholds
        sig.iloc[i] = last_signal
    sig = sig.ffill().fillna(0)

    return pd.DataFrame({
        'zs': zs,
        'sig': sig
    }, index=volume.index).reindex(full_index)
