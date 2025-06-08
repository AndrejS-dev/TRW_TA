import pandas as pd
import numpy as np
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('signal')
def hull_for_loop(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, hma_len: int = 15,
    loop_from: int = 1, loop_to: int = 50, threshold_l: int = 49, threshold_s: int = -10) -> pd.Series:
    """https://www.tradingview.com/script/CMsOR65t-Hull-For-Loop-viResearch/"""
    src = (open_ + high + low + close) / 4

    a = ta.hma(src, hma_len)
    a = a.reset_index(drop=True)

    score_list = []
    vii_list = []
    last_vii = 0
    prev_score = None

    for i in range(len(a)):
        score = 0
        for j in range(loop_from, loop_to + 1):
            if i - j >= 0 and not np.isnan(a[i]) and not np.isnan(a[i - j]):
                score += -1 if a[i - j] > a[i] else 1
        score_list.append(score)

        # Cross logic
        L = prev_score is not None and score > threshold_l and prev_score <= threshold_l
        S = prev_score is not None and score < threshold_s and prev_score >= threshold_s

        if L and not S:
            vii = 1
        elif S:
            vii = -1
        else:
            vii = last_vii

        vii_list.append(vii)
        last_vii = vii
        prev_score = score

    return pd.Series(vii_list, index=close.index)
