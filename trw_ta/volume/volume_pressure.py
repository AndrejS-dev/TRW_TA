import pandas as pd
import numpy as np
from trw_ta import register_outputs, ema, sma

@register_outputs('vol_ratio', 'signal_line')
def volume_pressure(open_: pd.Series, close: pd.Series, volume: pd.Series, length: int = 25, 
                    signal_smoothing: int = 8, use_log_scale: bool = False) -> pd.DataFrame:
    """
    Volume pressure.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/script/Mt0Nb4rM-Volume-Pressure-Histogram-Normalized/
    
    Parameters
    ----------
    open_ : pd.Series
        Open .
    close : pd.Series
        Close price series.
    volume : pd.Series
        Volume series.
    length : int, default 25
        Lookback period.
    signal_smoothing : int, default 8
        Signal smoothing.
    use_log_scale : bool, default False
        Use log scale.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """

    up_vol = volume.where(close > open_, 0.0)
    down_vol = volume.where(close < open_, 0.0)

    delta_vol = up_vol - down_vol
    smoothed_vol = ema(delta_vol, length)
    avg_volume = sma(volume, length)
    vol_ratio = (smoothed_vol / avg_volume) * 100

    if use_log_scale:
        vol_ratio = np.log10(np.abs(vol_ratio) + 1) * np.sign(vol_ratio)

    signal_line = ema(vol_ratio, signal_smoothing)
    
    return pd.DataFrame({
        'vol_ratio': vol_ratio,
        'signal_line': signal_line
    })