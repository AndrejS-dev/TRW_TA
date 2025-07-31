import numpy as np
import pandas as pd
import trw_ta
from trw_ta import register_outputs

@register_outputs('gau_ma', 'osc', 'trend')
def lowess_trend(close: pd.Series, length: int = 100) -> pd.DataFrame:
    """https://www.tradingview.com/v/hyeoDyZn/"""
    gau_ma = trw_ta.gaussian_ma(close, length)
    osc = close - gau_ma
    trend = np.where(gau_ma > gau_ma.shift(1), 1,
             np.where(gau_ma < gau_ma.shift(1), -1, 0))
    return pd.DataFrame({
        'gau_ma': gau_ma,
        'osc': osc,
        'trend': trend
    }, index=close.index)
