import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('upb', 'lpd', 'mid')
def projection_bands(high: pd.Series, low: pd.Series, length: int = 14) -> pd.DataFrame:
    """https://www.tradingview.com/script/yAKD6bQA-3-projection-Indicators-PBands-PO-PB/"""
    cum_idx = np.arange(1, len(high) + 1)

    sum_c = pd.Series(cum_idx).rolling(length).sum()
    psum_c = sum_c ** 2
    sump_c = pd.Series(cum_idx ** 2).rolling(length).sum()
    lsump_c = length * sump_c
    denom = lsump_c - psum_c

    # rlh and rll
    rlh = ((length * pd.Series(cum_idx * high).rolling(length).sum()) -
           (sum_c * high.rolling(length).sum())) / denom

    rll = ((length * pd.Series(cum_idx * low).rolling(length).sum()) -
           (sum_c * low.rolling(length).sum())) / denom

    # Upper Projection Band (upb)
    upb_list = []
    for i in range(len(high)):
        vals = []
        for shift in range(length):
            if i - shift >= 0:
                vals.append(high.iloc[i - shift] + shift * rlh.iloc[i])
        upb_list.append(max(vals) if vals else np.nan)
    upb = pd.Series(upb_list, index=high.index)

    # Lower Projection Band (lpb)
    lpb_list = []
    for i in range(len(low)):
        vals = []
        for shift in range(length):
            if i - shift >= 0:
                vals.append(low.iloc[i - shift] + shift * rll.iloc[i])
        lpb_list.append(min(vals) if vals else np.nan)
    lpb = pd.Series(lpb_list, index=low.index)

    mid = (upb + lpb) / 2

    return pd.DataFrame({
        "upb": upb,
        "lpb": lpb,
        "mid": mid
    })