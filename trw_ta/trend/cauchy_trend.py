import numpy as np
import pandas as pd
from trw_ta import register_outputs

def cauchy_pdf(offset: int, gamma: float) -> float:
    """
    Cauchy pdf.
    
    Parameters
    ----------
    offset : int
        Offset parameter.
    gamma : float
        Gamma.
    
    Returns
    -------
    float
        Scalar result.
    """
    numerator = gamma ** 2
    denominator = offset ** 2 + gamma ** 2
    return (1 / (np.pi * gamma)) * (numerator / denominator)

def calc_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    """
    Calc atr.
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    period : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    high = pd.Series(high)
    low = pd.Series(low)
    close = pd.Series(close)
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def supertrend(cwm_avg, src, atr, factor):
    """
    Supertrend.
    
    Parameters
    ----------
    cwm_avg : Any
        Cwm avg.
    src : Any
        Input price or value series.
    atr : Any
        Atr.
    factor : Any
        Factor.
    """
    n = len(src)
    upper_band = cwm_avg + factor * atr
    lower_band = cwm_avg - factor * atr

    upper_band = upper_band.values
    lower_band = lower_band.values
    src = src.values
    atr = atr.values

    st_vals = [np.nan]
    dir_vals = [0]

    # Find first valid ATR index
    first_valid = np.where(~np.isnan(atr))[0]
    if len(first_valid) == 0:
        raise ValueError("ATR is completely NaN. Check inputs.")
    start = first_valid[0] + 1  # start after first valid ATR

    for i in range(1, n):
        if i < start:
            st_vals.append(np.nan)
            dir_vals.append(0)
            continue

        if lower_band[i] > lower_band[i - 1] or src[i - 1] < lower_band[i - 1]:
            pass
        else:
            lower_band[i] = lower_band[i - 1]

        if upper_band[i] < upper_band[i - 1] or src[i - 1] > upper_band[i - 1]:
            pass
        else:
            upper_band[i] = upper_band[i - 1]

        prev_st = st_vals[-1]
        if np.isnan(prev_st):
            # Initialize direction to 1 (up) or -1 based on src vs. middle band
            direction = 1 if src[i] >= cwm_avg[i] else -1
        elif prev_st == upper_band[i - 1]:
            direction = 1 if src[i] > upper_band[i] else -1
        else:
            direction = -1 if src[i] < lower_band[i] else 1

        st_val = lower_band[i] if direction == 1 else upper_band[i]
        st_vals.append(st_val)
        dir_vals.append(direction)

    return pd.Series(st_vals), pd.Series(dir_vals)


@register_outputs('cauchyTrend', 'trend', 'cwm_avg')
def cauchy_trend(high: pd.Series, low: pd.Series, close: pd.Series, input_src: pd.Series, length: int = 28,
                             gamma: float = 0.5, atr_len: int = 14, atr_mult: float = 2.0) -> pd.DataFrame:
    """
    Cauchy trend.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/script/XuPnhy3w-CauchyTrend-InvestorUnknown/
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    input_src : pd.Series
        Input src.
    length : int, default 28
        Lookback period.
    gamma : float, default 0.5
        Gamma.
    atr_len : int, default 14
        Lookback / smoothing period.
    atr_mult : float, default 2.0
        Multiplier.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    df_len = len(close)

    # Calculate Cauchy-weighted average
    cwm_avg = []
    for i in range(df_len):
        values, weights = [], []
        for j in range(length):
            idx = i - j if i - j >= 0 else 0
            val = input_src.iloc[idx]
            w = cauchy_pdf(j, gamma)
            values.append(val)
            weights.append(w)
        norm_weights = np.array(weights) / np.sum(weights)
        cwm_avg.append(np.dot(values, norm_weights))

    # Calculate ATR and Bands
    atr = calc_atr(high, low, close, atr_len)
    cwm_avg = pd.Series(cwm_avg)

    # SuperTrend Logic
    cauchy_trend, trend = supertrend(cwm_avg, input_src, atr, atr_mult)

    return pd.DataFrame({
        "CauchyTrend": cauchy_trend,
        "Trend": trend,
        "cwm_avg": cwm_avg
    })
