import pandas as pd
from trw_ta import register_outputs

@register_outputs('wpr')
def williams_percent_r(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    highest_high = high.rolling(window=length).max()
    lowest_low = low.rolling(window=length).min()
    percent_r = 100 * (close - highest_high) / (highest_high - lowest_low)
    return percent_r
