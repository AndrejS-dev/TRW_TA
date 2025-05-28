import pandas as pd

def stochastic(high: pd.Series, low: pd.Series, close: pd.Series, period_k: int = 14, smooth_k: int = 1, period_d: int = 3) -> pd.DataFrame:
    lowest_low = low.rolling(window=period_k).min()
    highest_high = high.rolling(window=period_k).max()

    raw_k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    k = raw_k.rolling(window=smooth_k).mean()
    d = k.rolling(window=period_d).mean()

    return pd.DataFrame({
        "k": k,
        "d": d
    })
