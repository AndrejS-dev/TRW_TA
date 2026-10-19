import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('m_kijun_sen')
def median_kijun_sen(source: pd.Series, median_length: int, kijun_length: int) -> pd.Series:
    """
    Median kijun sen.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/script/EjXXXaVk-Median-Kijun-Sen-InvestorUnknown/
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    median_length : int
        Lookback / smoothing period.
    kijun_length : int
        Lookback / smoothing period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    def kijun_sen(source: pd.Series, length: int) -> pd.Series:
        return (ta.highest(source, length) + ta.lowest(source, length)) / 2
    return kijun_sen(ta.median(source, median_length), kijun_length)
