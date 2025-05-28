import pandas as pd
import numpy as np

def median_for_loop(source: pd.Series, median_len: int = 2, a: int = 10, b: int = 60,
    threshold_l: int = 40, threshold_s: int = 15) -> pd.Series:
    """https://www.tradingview.com/script/V63HaPE4-Median-For-Loop-viResearch/"""
    subject = source.rolling(window=median_len, center=False).median()

    subject = subject.reset_index(drop=True)
    score_series = []
    vii_series = []
    last_vii = 0

    for i in range(len(subject)):
        score = 0.0

        for j in range(a, b + 1):
            if i - j >= 0 and not np.isnan(subject[i]) and not np.isnan(subject[i - j]):
                score += 1 if subject[i] > subject[i - j] else -1

        score_series.append(score)

        if score > threshold_l and score < threshold_s:
            vii = last_vii
        elif score > threshold_l:
            vii = 1
        elif score < threshold_s:
            vii = -1
        else:
            vii = last_vii

        vii_series.append(vii)
        last_vii = vii

    return pd.Series(vii_series)
