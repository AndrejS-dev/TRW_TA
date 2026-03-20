import pandas as pd
import numpy as np
from trw_ta import register_outputs, hma

@register_outputs('plf_smooth')
def price_lag_factor(close: pd.Series, long_period: int = 50, short_period: int = 14, smoothing: int = 3) -> pd.DataFrame:
    """https://www.tradingview.com/script/mSwWP8uu-Price-Lag-Factor-PLF/"""

    def zscore_component(src: pd.Series, length: int) -> pd.Series:
        mean = src.rolling(length).mean()
        return src - mean

    z_long  = zscore_component(close, long_period)
    z_short = zscore_component(close, short_period)

    plf_raw = z_long - z_short
    plf_normalized = plf_raw / plf_raw.rolling(long_period).std(ddof=0)

    plf_smooth = hma(plf_normalized, smoothing)

    return pd.DataFrame({
        'plf_smooth': plf_smooth
    }).rename_axis(None, axis=1)