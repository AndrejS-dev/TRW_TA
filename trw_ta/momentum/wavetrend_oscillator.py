import numpy as np
import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('cl', 'sig')
def wavetrend_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    n1: int = 10,
    n2: int = 21,
    reaction_wt: int = 1,
    smooth: bool = False,
    ma_type: str = "EMA",
    ma_len: int = 200,
) -> pd.DataFrame:
    """https://www.tradingview.com/script/LlQl3rFf-WaveTrend-Oscillator-Divergence-Direction-Detection-Alerts/"""

    for s, name in zip([high, low, close], ["high", "low", "close"]):
        if not isinstance(s, pd.Series):
            raise TypeError(f"{name} must be a pandas Series")
        if s.isna().any():
            raise ValueError(f"{name} contains NaN")
    if len(high) != len(low) or len(high) != len(close):
        raise ValueError("high, low, close must have the same length")
    if len(high) < max(n1, n2, 50):
        raise ValueError("Not enough data (need at least max(n1,n2,50) rows)")


    ap = (high + low + close) / 3.0                     # hlc3
    esa = ap.ewm(span=n1, adjust=False).mean()
    d = (ap - esa).abs().ewm(span=n1, adjust=False).mean()
    ci = (ap - esa) / (0.015 * (d + 1e-9))
    wt1 = ci.ewm(span=n2, adjust=False).mean()          # main line
    wt2 = wt1.rolling(window=4, min_periods=1).mean()   # signal line


    rising = wt1.diff(reaction_wt) > 0
    falling = wt1.diff(reaction_wt) < 0

    direction = pd.Series(0, index=wt1.index)
    direction = direction.mask(rising, 1)
    direction = direction.mask(falling, -1)
    direction = direction.fillna(method="ffill").fillna(0).astype(int)

    if smooth:
        wt1 = ta.ma(wt1, ma_len, ma_type)

    return pd.DataFrame(
        {
            "cl": wt1,       # the oscillator line
            "sig": direction # +1 / -1 / 0
        },
        index=high.index,
    )