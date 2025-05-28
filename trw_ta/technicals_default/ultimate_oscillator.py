import pandas as pd

def ultimate_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, length1: int = 7, length2: int = 14, length3: int = 28) -> pd.Series:
    prev_close = close.shift(1)
    high_ = high.combine(prev_close, max)
    low_ = low.combine(prev_close, min)

    bp = close - low_
    tr_ = high_ - low_

    def avg(bp_, tr_, length):
        return bp_.rolling(length).sum() / tr_.rolling(length).sum()

    avg1 = avg(bp, tr_, length1)
    avg2 = avg(bp, tr_, length2)
    avg3 = avg(bp, tr_, length3)

    uo = 100 * ((4 * avg1) + (2 * avg2) + avg3) / 7
    return uo
