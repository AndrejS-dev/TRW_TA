import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('osc', 'signal')
def firefly_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, lookback_length: int = 10, signal_smoothing: int = 3, double_smooth: bool = False, use_zlema: bool = False) -> pd.DataFrame:
    """https://www.tradingview.com/script/pgvRuoUi-Firefly-Oscillator-LazyBear/"""
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