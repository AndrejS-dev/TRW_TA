import pandas as pd
import numpy as np
from trw_ta import register_outputs

def kalman_filter_1d(src: pd.Series, measurement_noise: float = 3.0, process_noise: float = 0.01) -> pd.Series:
    """
    Kalman filter 1d.
    
    Parameters
    ----------
    src : pd.Series
        Input price or value series.
    measurement_noise : float, default 3.0
        Measurement noise.
    process_noise : float, default 0.01
        Process noise.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    n = len(src)
    if n == 0:
        return pd.Series(dtype=float)

    estimate = np.full(n, np.nan, dtype=float)
    error_cov = np.full(n, 100.0, dtype=float)

    first_valid = src.first_valid_index()
    if first_valid is None:
        return pd.Series(np.nan, index=src.index)

    idx0 = src.index.get_loc(first_valid)
    estimate[idx0] = src.iloc[idx0]
    error_cov[idx0] = 1.0

    for i in range(idx0 + 1, n):
        if pd.isna(src.iloc[i]):
            estimate[i] = estimate[i-1]
            error_cov[i] = error_cov[i-1]
            continue

        pred_estimate = estimate[i-1]
        pred_cov = error_cov[i-1] + process_noise

        kg = pred_cov / (pred_cov + measurement_noise)

        estimate[i] = pred_estimate + kg * (src.iloc[i] - pred_estimate)
        error_cov[i] = (1 - kg) * pred_cov

    return pd.Series(estimate, index=src.index)

def kalman_hull_ma(src: pd.Series, length: float = 3.0, process_noise: float = 0.01) -> pd.Series:
    """
    Kalman hull ma.
    
    Parameters
    ----------
    src : pd.Series
        Input price or value series.
    length : float, default 3.0
        Lookback period.
    process_noise : float, default 0.01
        Process noise.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    if length < 1:
        return src

    half = length / 2
    sqrt_len = np.round(np.sqrt(length))

    kalman_half   = kalman_filter_1d(src, measurement_noise=half,   process_noise=process_noise)
    kalman_full   = kalman_filter_1d(src, measurement_noise=length, process_noise=process_noise)
    diff          = 2 * kalman_half - kalman_full
    khma          = kalman_filter_1d(diff, measurement_noise=sqrt_len, process_noise=process_noise)

    return khma

def donchian_midpoint(src: pd.Series, period: int = 26) -> pd.Series:
    """
    Donchian midpoint.
    
    Parameters
    ----------
    src : pd.Series
        Input price or value series.
    period : int, default 26
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    if period < 1:
        return src

    high_roll = src.rolling(period, min_periods=1).max()
    low_roll  = src.rolling(period, min_periods=1).min()
    return (high_roll + low_roll) / 2

@register_outputs('kijun', 'trend')
def kalman_hull_kijun(source: pd.Series, base_period: int = 26, measurement_noise: float = 3.0,
    process_noise: float = 0.01) -> pd.DataFrame:
    """
    Kalman hull kijun.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/script/0E5xrn6O-Kalman-Hull-Kijun-BackQuant/
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    base_period : int, default 26
        Lookback / smoothing period.
    measurement_noise : float, default 3.0
        Measurement noise.
    process_noise : float, default 0.01
        Process noise.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    khma = kalman_hull_ma(source, length=measurement_noise, process_noise=process_noise)
    kijun = donchian_midpoint(khma, period=base_period)

    trend = pd.Series(np.nan, index=source.index, dtype=float)
    valid = (source > kijun).astype(float) * 2 - 1
    trend = valid.where(pd.notna(kijun) & pd.notna(source))

    return pd.DataFrame({
        'kijun': kijun,
        'trend': trend.astype('Int64'),
    }).rename_axis(None, axis=1)