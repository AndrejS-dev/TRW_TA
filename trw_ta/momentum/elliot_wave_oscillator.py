import pandas as pd
from trw_ta import register_outputs

@register_outputs('smadif')
def elliot_wave_oscillator(src: pd.Series, sma1_length: int = 5, sma2_length: int = 35, use_percent: bool = True) -> pd.Series:
    """
    Elliot wave oscillator.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/v/VNJP71tP/
    
    Parameters
    ----------
    src : pd.Series
        Input price or value series.
    sma1_length : int, default 5
        Lookback / smoothing period.
    sma2_length : int, default 35
        Lookback / smoothing period.
    use_percent : bool, default True
        Use percent.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    sma_fast = src.rolling(window=sma1_length).mean()
    sma_slow = src.rolling(window=sma2_length).mean()
    
    if use_percent:
        smadif = (sma_fast - sma_slow) / src * 100
    else:
        smadif = sma_fast - sma_slow

    return smadif
