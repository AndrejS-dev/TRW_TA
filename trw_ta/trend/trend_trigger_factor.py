import pandas as pd
import numpy as np
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('TTF')
def trend_trigger_factor(high: pd.Series, low: pd.Series, length: int = 15) -> pd.DataFrame:
    """
    Trend trigger factor.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/script/wSMZKu7B-Indicator-Trend-Trigger-Factor/
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    length : int, default 15
        Lookback period.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    highest_high = ta.highest(high, length)
    lowest_low = ta.lowest(low, length)
    bp = highest_high - lowest_low.shift(length)
    sp = highest_high.shift(length) - lowest_low

    ttf_val = 100 * (bp - sp) / (0.5 * (bp + sp))
    
    return pd.DataFrame({"TTF": ttf_val})
