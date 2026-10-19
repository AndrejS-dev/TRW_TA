import pandas as pd
import numpy as np
from trw_ta import register_outputs

def manual_tanh(x: pd.Series) -> pd.Series:
    """
    Manual tanh.
    
    Parameters
    ----------
    x : pd.Series
        Numeric input.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    e2x = np.exp(2 * x)
    return (e2x - 1) / (e2x + 1)

def manual_clamp(x: pd.Series, min_val: float, max_val: float) -> pd.Series:
    """
    Manual clamp.
    
    Parameters
    ----------
    x : pd.Series
        Numeric input.
    min_val : float
        Min val.
    max_val : float
        Max val.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return np.maximum(min_val, np.minimum(max_val, x))

@register_outputs('prism', 'signal')
def trend_pressure_prism(high: pd.Series, low: pd.Series, close: pd.Series,
    lookback_period: int = 20, sensitivity: float = 1.5, signal_length: int = 9) -> pd.DataFrame:
    """
    Trend pressure prism.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/script/et2oFiWR-Trend-Pressure-Prism-LuxAlgo/
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    lookback_period : int, default 20
        Lookback / smoothing period.
    sensitivity : float, default 1.5
        Sensitivity.
    signal_length : int, default 9
        Lookback / smoothing period.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    mom_roc = close.pct_change(lookback_period) * 100
    mom_std = mom_roc.rolling(lookback_period, min_periods=1).std()
    mom_drive = (mom_roc / (mom_std + 1e-10)).ewm(span=5, adjust=False, min_periods=1).mean()

    ema_fast = close.ewm(span=lookback_period, adjust=False, min_periods=1).mean()
    ema_slow = close.ewm(span=lookback_period * 2, adjust=False, min_periods=1).mean()
    struct_val = (close - ema_slow) / (ema_fast - ema_slow + 1e-10)
    struct_align = manual_clamp(struct_val, -2, 2).ewm(span=5, adjust=False, min_periods=1).mean()

    highest_high = high.rolling(lookback_period, min_periods=1).max()
    lowest_low   = low.rolling(lookback_period, min_periods=1).min()
    range_ = highest_high - lowest_low
    pullback = (close - lowest_low) / (range_ + 1e-10)
    pb_quality = (pullback - 0.5) * 2

    norm_mom   = manual_tanh(mom_drive * sensitivity)
    norm_struct = manual_tanh(struct_align * sensitivity)
    norm_pb     = manual_tanh(pb_quality * sensitivity)

    composite = (norm_mom + norm_struct + norm_pb) / 3
    comp_score = composite * 100

    sig_line = comp_score.ewm(span=signal_length, adjust=False, min_periods=1).mean()

    return pd.DataFrame({
        'prism':  comp_score,
        'signal': sig_line,
    }).rename_axis(None, axis=1)