import pandas as pd
import numpy as np
from trw_ta import register_outputs
from trw_ta.ta_core import valid_data_prefix

@register_outputs('kama', 'signal')
def kama_trend_flip(src: pd.Series, er_len: int = 50, fast_pow: int = 2, slow_pow: int = 30) -> pd.DataFrame:
    """
    Kama trend flip.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/script/fNAmA2Gz-KAMA-Trend-Flip-SightLing-Labs/
    
    Parameters
    ----------
    src : pd.Series
        Input price or value series.
    er_len : int, default 50
        Lookback / smoothing period.
    fast_pow : int, default 2
        Fast pow.
    slow_pow : int, default 30
        Slow pow.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    if not isinstance(src, pd.Series):
        raise TypeError("Source must be a pandas Series")
    (src,), full_index = valid_data_prefix(src)
    if len(src) < er_len:
        raise ValueError(f"Source must have at least {er_len} non-NaN rows")
    if er_len < 1 or fast_pow < 1 or slow_pow < 1:
        raise ValueError("er_len, fast_pow, and slow_pow must be >= 1")

    change = abs(src - src.shift(er_len))
    volatility = abs(src - src.shift(1)).rolling(window=er_len, min_periods=1).sum()
    er = change / volatility.where(volatility != 0, 1)
    fast_sc = 2 / (fast_pow + 1)
    slow_sc = 2 / (slow_pow + 1)
    sc = er * (fast_sc - slow_sc) + slow_sc

    kama = pd.Series(np.nan, index=src.index)
    kama.iloc[0] = src.iloc[0]  # Initialize first value
    for i in range(1, len(src)):
        kama.iloc[i] = kama.iloc[i-1] + (sc.iloc[i] ** 2) * (src.iloc[i] - kama.iloc[i-1]) if not np.isnan(kama.iloc[i-1]) else src.iloc[i]
    kama = kama.ffill()

    trend_up = kama > kama.shift(1)
    trend_dn = kama < kama.shift(1)
    signal = pd.Series(0, index=src.index)
    last_signal = 0  # 0: neutral, 1: uptrend, -1: downtrend
    for i in range(1, len(src)):
        if trend_up.iloc[i] and not trend_up.iloc[i-1]:
            last_signal = 1
        elif trend_dn.iloc[i] and not trend_dn.iloc[i-1]:
            last_signal = -1
        signal.iloc[i] = last_signal
    signal = signal.ffill().fillna(0)

    return pd.DataFrame({
        'kama': kama,
        'signal': signal
    }, index=src.index).reindex(full_index)
