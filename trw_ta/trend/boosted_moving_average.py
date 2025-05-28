import pandas as pd
import numpy as np

def boosted_moving_average(close: pd.Series, length: int = 43, boost_factor: float = 1.5) -> pd.DataFrame:
    """https://www.tradingview.com/v/eashAKsX/"""
    ema1 = close.ewm(span=length, adjust=False).mean()
    ema2 = close.ewm(span=length / 2, adjust=False).mean()


    boosted_value = ema2 + boost_factor * (ema2 - ema1)

    smoothing_len = int(round(length / boost_factor))
    smoothed_value = boosted_value.ewm(span=smoothing_len, adjust=False).mean()

    shifted_value = smoothed_value.shift(1)
    wma_window = int(round(boost_factor))

    if wma_window < 1:
        wma_window = 1  # Avoid zero window

    def wma(x):
        weights = np.arange(1, len(x) + 1)
        return np.dot(x, weights) / weights.sum()

    reference = shifted_value.rolling(window=wma_window).apply(wma, raw=True)

    signal = np.where(smoothed_value > reference, 1, -1)

    return pd.DataFrame({
        "boosted_ma": smoothed_value,
        "signal": signal
    })
