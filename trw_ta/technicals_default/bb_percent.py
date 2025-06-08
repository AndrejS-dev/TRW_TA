import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('bb percent')
def BB_percent(source: pd.Series, length: int = 20, mult: float = 2.0) -> pd.Series:
    basis = ta.sma(source, length)
    dev = mult * ta.stdev(source, length)
    upper = basis + dev
    lower = basis - dev
    return (source - lower)/(upper - lower)