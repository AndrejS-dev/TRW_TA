import numpy as np
import pandas as pd

import trw_ta as ta
from trw_ta import register_outputs


def _pivothigh(series: pd.Series, left: int, right: int) -> pd.Series:
    """TradingView ``ta.pivothigh(left, right)`` — value appears on the confirmation bar."""
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        center = i - right
        window = values[center - left : center + right + 1]
        if values[center] == np.nanmax(window):
            out[i] = values[center]
    return pd.Series(out, index=series.index)


def _pivotlow(series: pd.Series, left: int, right: int) -> pd.Series:
    """TradingView ``ta.pivotlow(left, right)`` — value appears on the confirmation bar."""
    values = series.to_numpy(dtype=float)
    n = len(values)
    out = np.full(n, np.nan)
    for i in range(left + right, n):
        center = i - right
        window = values[center - left : center + right + 1]
        if values[center] == np.nanmin(window):
            out[i] = values[center]
    return pd.Series(out, index=series.index)


@register_outputs("trend")
def swing_flow(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    len_period: int = 10,
    band_mult: float = 2.0,
    volatility_length: int = 50,
) -> pd.DataFrame:
    """
    Swing Flow Indicator [ChartPrime].

    Builds a pivot-informed midline, wraps it with ATR(200)-scaled bands, and
    flips trend state when price breaks those bands.

    Python conversion of the TradingView / Pine Script indicator.

    Source: https://www.tradingview.com/script/BtGC9TGm-Swing-Flow-Indicator-ChartPrime/

    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    len_period : int, default 10
        Pivot left/right strength and SMA smoothing length.
    band_mult : float, default 2.0
        Multiplier applied to ATR(200) for band width (Pine ``Bands``, min 1).
    volatility_length : int, default 50
        Unused; kept for backward compatibility. Bands use ATR(200), as in Pine.

    Returns
    -------
    pd.DataFrame
        Column ``trend``: ``1`` bullish, ``-1`` bearish (``NaN`` until the first
        band cross).
    """
    _ = volatility_length  # deprecated / unused (Pine uses ATR(200))
    length = int(len_period)
    mult = float(band_mult)

    ph = _pivothigh(high, length, length)
    pl = _pivotlow(low, length, length)

    # Persist last structural anchors. On confirmation Pine assigns current bar
    # high/low (not the pivot price itself).
    h1 = pd.Series(np.nan, index=high.index, dtype=float)
    l1 = pd.Series(np.nan, index=low.index, dtype=float)
    last_h = np.nan
    last_l = np.nan
    ph_vals = ph.to_numpy()
    pl_vals = pl.to_numpy()
    high_vals = high.to_numpy(dtype=float)
    low_vals = low.to_numpy(dtype=float)

    for i in range(len(high)):
        if not np.isnan(ph_vals[i]):
            last_h = high_vals[i]
        if not np.isnan(pl_vals[i]):
            last_l = low_vals[i]
        h1.iloc[i] = last_h
        l1.iloc[i] = last_l

    atr = ta.atr(high, low, close, 200) * mult
    mid = ((h1 + l1) / 2).rolling(length, min_periods=1).mean()
    upper = (mid + atr).rolling(length * 2, min_periods=1).mean()
    lower = (mid - atr).rolling(length * 2, min_periods=1).mean()

    # Pine: ta.crossover(high, h) / ta.crossunder(low, l)
    cross_up = (high > upper) & (high.shift(1) <= upper.shift(1))
    cross_dn = (low < lower) & (low.shift(1) >= lower.shift(1))

    trend = pd.Series(np.nan, index=close.index, dtype=float)
    state = np.nan  # matches Pine ``var trend = bool(na)`` until first cross
    for i in range(len(close)):
        if bool(cross_up.iloc[i]):
            state = 1.0
        elif bool(cross_dn.iloc[i]):
            state = -1.0
        trend.iloc[i] = state

    # After first signal, forward-fill; bars before first cross stay NaN then
    # become the first established state for a stable series (optional fill).
    # Keep NaN until first cross so early bars are not falsely bearish.
    return pd.DataFrame({"trend": trend})
