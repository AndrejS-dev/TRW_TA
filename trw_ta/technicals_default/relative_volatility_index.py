import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('rvi')
def relative_volatility_index(src: pd.Series, length: int = 10, ema_len: int = 14) -> pd.Series:
    stddev = src.rolling(length).std()
    change = src.diff()

    upper = np.where(change <= 0, 0, stddev)
    lower = np.where(change > 0, 0, stddev)

    upper_ema = pd.Series(upper, index=src.index).ewm(span=ema_len, adjust=False).mean()
    lower_ema = pd.Series(lower, index=src.index).ewm(span=ema_len, adjust=False).mean()

    rvi = (upper_ema / (upper_ema + lower_ema)) * 100
    return rvi
