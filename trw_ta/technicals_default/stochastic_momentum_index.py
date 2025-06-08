import pandas as pd
from trw_ta import register_outputs

def double_ema(series: pd.Series, length: int) -> pd.Series:
    ema1 = series.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    return ema2

@register_outputs('smi')
def stochastic_momentum_index(high: pd.Series, low: pd.Series, close: pd.Series, length_k: int = 10, length_d: int = 3) -> pd.Series:
    hh = high.rolling(window=length_k).max()
    ll = low.rolling(window=length_k).min()
    mid = (hh + ll) / 2
    range_ = hh - ll
    rel = close - mid

    smi_val = 200 * (double_ema(rel, length_d) / double_ema(range_, length_d))
    return smi_val
