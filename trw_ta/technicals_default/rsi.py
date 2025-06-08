import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('rsi')
def rsi(src: pd.Series, length: int = 14) -> pd.Series:
    change = src.diff()
    gain = change.clip(lower=0)
    loss = -change.clip(upper=0)

    avg_gain = gain.ewm(alpha=1/length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, adjust=False).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    rsi = np.where(avg_loss == 0, 100, np.where(avg_gain == 0, 0, rsi))

    return pd.Series(rsi, index=src.index)
