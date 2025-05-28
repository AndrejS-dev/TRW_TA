import pandas as pd
import numpy as np

def deb_supertrend(close: pd.Series, length: int = 20, mult: float = 2.0) -> pd.Series:
    """https://www.tradingview.com/v/Ja7PuNY9/"""
    price = close
    ma = price.rolling(window=length).mean()
    std = price.rolling(window=length).std()
    normalized_price = (price - ma) / std

    inverted_gaussian = 1 - np.exp(-(normalized_price ** 2))

    upper_band = ma + (std * mult)
    lower_band = ma - (std * mult)

    is_long = np.full(len(close), np.nan)

    for i in range(len(close)):
        if i < length:
            is_long[i] = np.nan
            continue

        if close[i] > upper_band[i]:
            is_long[i] = 1
        elif close[i] < lower_band[i]:
            is_long[i] = -1
        else:
            is_long[i] = is_long[i - 1] if i > 0 else -1

    signal = pd.Series(np.where(is_long == 1, 1, -1), index=close.index)
    return signal
