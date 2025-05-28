import pandas as pd
import numpy as np
import trw_ta as ta

def dema_sma_stdev(high: pd.Series, low: pd.Series, source: pd.Series, dema_len: int = 5, sma_len: int = 60, sd_len: int = 20) -> pd.Series:
    """https://www.tradingview.com/script/WexMT4w3-DEMA-SMA-Standard-Deviation/"""
    srcl = high
    srcs = low

    dema_val = ta.dema(source, dema_len)

    sma = dema_val.rolling(window=sma_len).mean()
    sd = sma.rolling(window=sd_len).std()

    upper = sma + sd
    lower = sma - sd

    ma_sd_l = srcl > lower
    ma_sd_s = srcs < upper

    vii_list = []
    last_vii = 0

    for i in range(len(high)):
        L = ma_sd_l.iloc[i] if not pd.isna(ma_sd_l.iloc[i]) else False
        S = ma_sd_s.iloc[i] if not pd.isna(ma_sd_s.iloc[i]) else False

        if L and not S:
            vii = 1
        elif S:
            vii = -1
        else:
            vii = last_vii

        vii_list.append(vii)
        last_vii = vii

    return pd.Series(vii_list, index=high.index)
