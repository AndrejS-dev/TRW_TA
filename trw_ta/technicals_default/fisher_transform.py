import pandas as pd
import numpy as np

def fisher_transform(high: pd.Series, low: pd.Series, length: int = 9) -> pd.DataFrame:
    hl2 = (high + low) / 2
    high_ = hl2.rolling(window=length).max()
    low_ = hl2.rolling(window=length).min()

    # Initialize arrays
    value = np.zeros(len(hl2))
    fish1 = np.zeros(len(hl2))

    for i in range(1, len(hl2)):
        if pd.isna(high_[i]) or pd.isna(low_[i]):
            value[i] = value[i-1]
            fish1[i] = fish1[i-1]
            continue

        raw = 0.66 * ((hl2[i] - low_[i]) / (high_[i] - low_[i] + 1e-10) - 0.5)
        clipped = max(min(raw + 0.67 * value[i-1], 0.999), -0.999)
        value[i] = clipped

        fish = 0.5 * np.log((1 + clipped) / (1 - clipped + 1e-10)) + 0.5 * fish1[i-1]
        fish1[i] = fish

    fish1_series = pd.Series(fish1, index=hl2.index)
    fish2_series = fish1_series.shift(1)

    return pd.DataFrame({
        'fish1': fish1_series,
        'fish2': fish2_series
    })
