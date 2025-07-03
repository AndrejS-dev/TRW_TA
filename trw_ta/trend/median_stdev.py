import pandas as pd
import numpy as np
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('signal')
def median_stdev(high: pd.Series, low: pd.Series, close: pd.Series, len_dema: int = 7,
    median_len: int = 61, atr_len: int = 6, atr_mul: float = 0.6, stdev_len: int = 27) -> pd.Series:
    """https://www.tradingview.com/script/ywl57T7A-Median-Standard-Deviation-viResearch/"""

    dema = ta.dema(close, length=len_dema)

    median = dema.rolling(window=median_len).median()

    atr = ta.atr1(high, low, close, length=atr_len) * atr_mul

    upper = median + atr
    lower = median - atr

    meu = close > lower
    mel = close < upper

    sd = median.rolling(window=stdev_len).std()
    sdd = median + sd

    sd_l = close >= sdd

    L = meu & sd_l
    S = mel

    signal = pd.Series(np.nan, index=close.index)
    signal.loc[L & ~S] =  1
    signal.loc[S] = -1
    signal = signal.ffill().fillna(0).astype(int)

    return signal