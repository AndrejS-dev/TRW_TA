import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('up', 'down')
def aroon(high: pd.Series, low: pd.Series, length: int = 14) -> pd.DataFrame:
    def bars_since_max(x):
        return length - 1 - np.argmax(x)

    def bars_since_min(x):
        return length - 1 - np.argmin(x)

    high_bars_ago = high.rolling(window=length).apply(bars_since_max, raw=True)
    low_bars_ago = low.rolling(window=length).apply(bars_since_min, raw=True)

    aroon_up = 100 * (length - high_bars_ago) / length
    aroon_down = 100 * (length - low_bars_ago) / length

    return pd.DataFrame({
        "aroon_up": aroon_up,
        "aroon_down": aroon_down
    })

