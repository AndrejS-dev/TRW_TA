import pandas as pd
import numpy as np
import trw_ta as ta

def alma_vii(high: pd.Series, low: pd.Series, close: pd.Series, entry_src: pd.Series,
    alma_length: int = 40, hma_length: int = 20, offset: float = 0.7, sigma: float = 4.0) -> pd.Series:
    """https://www.tradingview.com/script/YIm6QQ4I-vii-Alma/"""
    alma_line = ta.alma(close, alma_length, offset, sigma)

    e1 = ta.hma(low, hma_length)
    e2 = ta.hma(e1, hma_length)
    dhma = 2 * e1 - e2

    hlcc4 = (high + low + close + close) / 4

    almal = (entry_src > alma_line) & (low > dhma)
    almas = (entry_src < alma_line) & (hlcc4 < dhma)

    alma_signal = []
    last = 0

    for i in range(len(close)):
        if almal.iloc[i]:
            last = 1
        elif almas.iloc[i]:
            last = -1
        alma_signal.append(last)

    return pd.Series(alma_signal, index=close.index)
