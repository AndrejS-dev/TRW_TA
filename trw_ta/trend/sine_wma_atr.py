import pandas as pd
import numpy as np
from trw_ta import register_outputs


def sine_weights(length: int) -> np.ndarray:
    """
    Sine weights.
    
    Parameters
    ----------
    length : int
        Lookback period.
    
    Returns
    -------
    np.ndarray
        Function result.
    """
    weights = np.array([np.sin(np.pi * (i + 1) / length) for i in range(length)])
    return weights / weights.sum()


def sine_weighted_ma(series: pd.Series, length: int) -> pd.Series:
    """
    Sine weighted ma.
    
    Parameters
    ----------
    series : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    weights = sine_weights(length)
    swma = np.full(len(series), np.nan)
    for i in range(length - 1, len(series)):
        window = series.iloc[i - length + 1:i + 1]
        swma[i] = np.dot(window, weights)
    return pd.Series(swma, index=series.index)


def sine_weighted_atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    """
    Sine weighted atr.
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    tr = pd.concat([
        (high - low).abs(),
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    weights = sine_weights(length)
    swatr = np.full(len(tr), np.nan)
    for i in range(length - 1, len(tr)):
        window = tr.iloc[i - length + 1:i + 1]
        swatr[i] = np.dot(window, weights)
    return pd.Series(swatr, index=tr.index)

@register_outputs('swma', 'signal')
def swma_atr_signals(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, 
                     ma_length: int = 14, atr_len: int = 14, atr_mult: float = 1.2, 
                     swma_src: str = "close", src_long: str = "close", src_short: str = "close") -> pd.DataFrame:
    """
    Swma atr signals.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/script/Ku1jOXK0-Sine-Weighted-MA-ATR-InvestorUnknown/
    
    Parameters
    ----------
    open : pd.Series
        Open price series.
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    ma_length : int, default 14
        Lookback / smoothing period.
    atr_len : int, default 14
        Lookback / smoothing period.
    atr_mult : float, default 1.2
        Multiplier.
    swma_src : str, default 'close'
        Swma src.
    src_long : str, default 'close'
        Src long.
    src_short : str, default 'close'
        Src short.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    src_map = {
        "open": open,
        "high": high,
        "low": low,
        "close": close,
        "oc2": (open + close) / 2,
        "hl2": (high + low) / 2,
        "occ3": (open + close + close) / 3,
        "hlc3": (high + low + close) / 3,
        "ohlc4": (open + high + low + close) / 4,
        "hlcc4": (high + low + close + close) / 4,
    }

    src = src_map[swma_src]
    src_l = src_map[src_long]
    src_s = src_map[src_short]

    swma = sine_weighted_ma(src, ma_length)

    atr = sine_weighted_atr(high, low, close, atr_len)

    swma_up = swma + (atr * atr_mult)
    swma_dn = swma - (atr * atr_mult)

    signal = pd.Series(0, index=close.index)

    for i in range(1, len(close)):
        if src_l.iloc[i - 1] < swma_up.iloc[i - 1] and src_l.iloc[i] >= swma_up.iloc[i]:
            signal.iloc[i] = 1
        elif src_s.iloc[i - 1] > swma_dn.iloc[i - 1] and src_s.iloc[i] <= swma_dn.iloc[i]:
            signal.iloc[i] = -1
        else:
            signal.iloc[i] = signal.iloc[i - 1]

    return pd.DataFrame({'swma': swma, 'signal': signal})
