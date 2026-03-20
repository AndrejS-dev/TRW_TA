import pandas as pd
import numpy as np
from trw_ta import register_outputs, ema, sma

@register_outputs('vol_ratio', 'signal_line')
def volume_pressure(open_: pd.Series, close: pd.Series, volume: pd.Series, length: int = 25, 
                    signal_smoothing: int = 8, use_log_scale: bool = False) -> pd.DataFrame:
    """https://www.tradingview.com/script/Mt0Nb4rM-Volume-Pressure-Histogram-Normalized/"""

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