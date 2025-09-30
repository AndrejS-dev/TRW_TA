import pandas as pd
from trw_ta import register_outputs

@register_outputs('wt1', 'wt2', 'hist')
def wavetrend(high: pd.Series, low: pd.Series, close: pd.Series, channel_len: int = 10, avg_len: int = 21, smooth_len: int = 4) -> pd.DataFrame:
    """https://www.tradingview.com/v/jFQn4jYZ/"""
    ap = (high + low + close) / 3
    esa = ap.ewm(span=channel_len, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=channel_len, adjust=False).mean()
    ci = (ap - esa) / (0.015 * d)
    tci = ci.ewm(span=avg_len, adjust=False).mean()

    wt1 = tci
    wt2 = wt1.rolling(window=smooth_len).mean()

    hist = wt1 - wt2

    return pd.DataFrame({
        'wt1': wt1,
        'wt2': wt2,
        'hist': hist
    })
