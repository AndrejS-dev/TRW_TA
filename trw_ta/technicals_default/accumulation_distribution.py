import numpy as np
import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('ad')
def accumulation_distribution(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    cond = ((close == high) & (close == low)) | (high == low)
    money_flow = np.where(cond, 0, ((2 * close - low - high) / (high - low)) * volume)
    return ta.cum(pd.Series(money_flow, index=close.index))  # ensure it's a Series with matching index
