import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('osc', 'signal')
def firefly_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, lookback_length: int = 10, signal_smoothing: int = 3, double_smooth: bool = False, use_zlema: bool = False) -> pd.DataFrame:
    """
    Firefly oscillator.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/script/pgvRuoUi-Firefly-Oscillator-LazyBear/
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    lookback_length : int, default 10
        Lookback / smoothing period.
    signal_smoothing : int, default 3
        Signal smoothing.
    double_smooth : bool, default False
        Double smooth.
    use_zlema : bool, default False
        Use zlema.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    def calc_zlema(src, length):
        ema1 = src.ewm(span=length, adjust=False).mean()
        ema2 = ema1.ewm(span=length, adjust=False).mean()
        d = ema1 - ema2
        return ema1 + d
    
    def ma(src, length, use_zlema):
        return calc_zlema(src, length) if use_zlema else src.ewm(span=length, adjust=False).mean()
    
    v2 = (high + low + close * 2) / 4
    v3 = ma(v2, lookback_length, use_zlema)
    v4 = v2.rolling(window=lookback_length).std()
    v4 = v4.where(v4 != 0, 1)
    v5 = (v2 - v3) * 100 / v4
    v6 = ma(v5, signal_smoothing, use_zlema)
    v7 = ma(v6, signal_smoothing, use_zlema) if double_smooth else v6
    ww = (ma(v7, lookback_length, use_zlema) + 100) / 2 - 4
    mm = ww.rolling(window=signal_smoothing).max()
    
    return pd.DataFrame({
        'oscillator': ww,
        'signal': mm
    }, index=high.index)