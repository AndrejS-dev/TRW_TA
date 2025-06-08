import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('co')
def chaikin_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, short_len: int = 3, long_len: int = 10) -> pd.Series:
    return ta.ema(ta.accumulation_distribution(high, low, close, volume), short_len) - ta.ema(ta.accumulation_distribution(high, low, close, volume), long_len)