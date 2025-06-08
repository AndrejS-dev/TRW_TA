import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('kvo', 'sig')
def klinger_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, fast=34, slow=55, signal=13) -> pd.Series:
    hlc3 = (high + low + close) / 3
    sv = volume.where(hlc3.diff() >= 0, -volume)
    kvo = sv.ewm(span=fast, adjust=False).mean() - sv.ewm(span=slow, adjust=False).mean()
    signal_line = kvo.ewm(span=signal, adjust=False).mean()

    return pd.DataFrame({
        'kvo': kvo,
        'sig': signal_line
    })