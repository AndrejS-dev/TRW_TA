import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('trix')
def trix(close: pd.Series, length: int = 18) -> pd.Series:
    log_close = np.log(close)

    ema1 = log_close.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()

    trix_val = ema3.diff() * 10000  # Equivalent to `ta.change(...) * 10000`
    return trix_val
