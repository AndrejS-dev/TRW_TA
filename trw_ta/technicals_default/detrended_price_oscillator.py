import pandas as pd
from trw_ta import register_outputs

@register_outputs('dpo')
def detrended_price_oscillator(source: pd.Series, length: int = 21, centered: bool = False) -> pd.Series:
    barsback = int(length / 2 + 1)
    ma = source.rolling(window=length).mean()
    
    if centered:
        shifted_close = source.shift(-barsback)
        dpo = shifted_close - ma
    else:
        shifted_ma = ma.shift(barsback)
        dpo = source - shifted_ma
    
    return dpo

