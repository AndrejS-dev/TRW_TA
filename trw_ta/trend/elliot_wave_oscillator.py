import pandas as pd
from trw_ta import register_outputs

@register_outputs('smadif')
def elliot_wave_oscillator(src: pd.Series, sma1_length: int = 5, sma2_length: int = 35, use_percent: bool = True) -> pd.Series:
    """https://www.tradingview.com/v/VNJP71tP/"""
    sma_fast = src.rolling(window=sma1_length).mean()
    sma_slow = src.rolling(window=sma2_length).mean()
    
    if use_percent:
        smadif = (sma_fast - sma_slow) / src * 100
    else:
        smadif = sma_fast - sma_slow

    return smadif
