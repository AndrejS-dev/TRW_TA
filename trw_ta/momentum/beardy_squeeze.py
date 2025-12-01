# trw_ta/momentum/beardy_squeeze.py
import pandas as pd
import numpy as np
from trw_ta import register_outputs


@register_outputs('bsq_mom')
def beardy_squeeze(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 20,
) -> pd.DataFrame:
    """https://www.tradingview.com/script/X6j1hgwQ-AKS-Squeeze-Pro/"""

    length = int(max(1, length))

    hh = high.rolling(length, min_periods=1).max()
    ll = low.rolling(length, min_periods=1).min()

    # Average of (HH+LL)/2 and SMA(close)
    mid_price = (hh + ll) / 2
    sma_price = close.rolling(length, min_periods=1).mean()
    avg_price = (mid_price + sma_price) / 2

    diff = close - avg_price

    x = np.arange(length)
    x_mean = x.mean()

    def linreg_slope(window: np.ndarray) -> float:
        if len(window) < 2:
            return 0.0
        cov = np.cov(x[:len(window)], window, bias=True)[0, 1]
        var_x = x[:len(window)].var()
        return cov / var_x if var_x != 0 else 0.0

    mom = diff.rolling(length, min_periods=1).apply(linreg_slope, raw=True)

    return pd.DataFrame({
        "bsq_mom": mom
    }, index=close.index)