import pandas as pd
import trw_ta as ta

def median_kijun_sen(source: pd.Series, median_length: int, kijun_length: int) -> pd.Series:
    """https://www.tradingview.com/script/EjXXXaVk-Median-Kijun-Sen-InvestorUnknown/"""
    def kijun_sen(source: pd.Series, length: int) -> pd.Series:
        return (ta.highest(source, length) + ta.lowest(source, length)) / 2
    return kijun_sen(ta.median(source, median_length), kijun_length)
