import pandas as pd
from trw_ta import register_outputs

@register_outputs('tsi')
def true_strength_index(series: pd.Series, short_len: int = 13, long_len: int = 25) -> pd.Series:

    momentum = series.diff()
    abs_momentum = momentum.abs()

    ema1 = momentum.ewm(span=short_len, adjust=False).mean()
    ema2 = ema1.ewm(span=long_len, adjust=False).mean()

    abs_ema1 = abs_momentum.ewm(span=short_len, adjust=False).mean()
    abs_ema2 = abs_ema1.ewm(span=long_len, adjust=False).mean()

    tsi = 100 * ema2 / abs_ema2
    return tsi