import numpy as np
import pandas as pd
from trw_ta import register_outputs

@register_outputs('vma','signal')
def variable_moving_average(close: pd.Series, length: int = 6) -> pd.DataFrame:
    """https://www.tradingview.com/v/6Ix0E5Yr/"""
    k = 1.0 / length

    pdm = np.maximum(close - close.shift(1), 0)
    mdm = np.maximum(close.shift(1) - close, 0)

    pdmS = pdm.ewm(alpha=k, adjust=False).mean()
    mdmS = mdm.ewm(alpha=k, adjust=False).mean()

    s = pdmS + mdmS
    pdi = pdmS / s.replace(0, np.nan)
    mdi = mdmS / s.replace(0, np.nan)

    pdiS = pdi.ewm(alpha=k, adjust=False).mean()
    mdiS = mdi.ewm(alpha=k, adjust=False).mean()

    d = (pdiS - mdiS).abs()
    s1 = pdiS + mdiS
    iS = (d / s1.replace(0, np.nan)).ewm(alpha=k, adjust=False).mean()

    hhv = iS.rolling(window=length).max()
    llv = iS.rolling(window=length).min()
    d1 = (hhv - llv).replace(0, np.nan)
    vI = (iS - llv) / d1

    vma = pd.Series(index=close.index, dtype='float64')
    for i in range(len(close)):
        if i == 0 or pd.isna(vI.iloc[i]) or pd.isna(close.iloc[i]):
            vma.iloc[i] = close.iloc[i]
        else:
            vma.iloc[i] = (1 - k * vI.iloc[i]) * vma.iloc[i - 1] + k * vI.iloc[i] * close.iloc[i]

    signal = np.where(vma > vma.shift(1), 1,
             np.where(vma < vma.shift(1), -1, 0))

    return pd.DataFrame({
        "vma": vma,
        "signal": signal
    })
