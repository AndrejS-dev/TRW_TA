import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('fast_osc', 'slow_osc')
def hurst_cycle_channel_oscillator(src: pd.Series, short_cycle_length: int = 10, medium_cycle_length: int = 30, short_cycle_multiplier: float = 1.0, medium_cycle_multiplier: float = 3.0) -> pd.DataFrame:
    """https://www.tradingview.com/script/3yAQDB3h-Cycle-Channel-Oscillator-LazyBear/"""
    scl = short_cycle_length // 2
    mcl = medium_cycle_length // 2
    
    def rma(series, length):
        return series.ewm(alpha=1/length, adjust=False).mean()
    
    def atr(series, length):
        high_low = series.rolling(window=length).max() - series.rolling(window=length).min()
        return high_low.ewm(alpha=1/length, adjust=False).mean()
    
    ma_scl = rma(src, scl)
    ma_mcl = rma(src, mcl)
    
    scm_off = short_cycle_multiplier * atr(src, scl)
    mcm_off = medium_cycle_multiplier * atr(src, mcl)
    
    scl_2 = scl // 2
    mcl_2 = mcl // 2
    
    sct = ma_scl.shift(scl_2).fillna(src) + scm_off
    scb = ma_scl.shift(scl_2).fillna(src) - scm_off
    mct = ma_mcl.shift(mcl_2).fillna(src) + mcm_off
    mcb = ma_mcl.shift(mcl_2).fillna(src) - mcm_off
    
    scmm = (sct + scb) / 2
    omed = (scmm - mcb) / (mct - mcb).where((mct - mcb) != 0, 1)
    oshort = (src - mcb) / (mct - mcb).where((mct - mcb) != 0, 1)
    
    return pd.DataFrame({
        'fast_osc': oshort,
        'slow_osc': omed
    }, index=src.index)