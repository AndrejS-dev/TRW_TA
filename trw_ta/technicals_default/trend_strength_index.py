import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('tsi')
def trend_strength_index(close: pd.Series, length: int = 14) -> pd.Series:
    time_index = pd.Series(np.arange(len(close)), index=close.index)
    tsi_series = close.rolling(length).corr(time_index)
    return tsi_series
