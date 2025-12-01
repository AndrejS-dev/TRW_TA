import pandas as pd
import numpy as np
from trw_ta import register_outputs


@register_outputs('zro_osc')
def zeiierman_range_oscillator(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    length: int = 50,
    mult: float = 2.0,
) -> pd.DataFrame:
    """https://www.tradingview.com/script/mlL8CpJq-Range-Oscillator-Zeiierman/"""

    length = int(max(1, length))
    mult = float(mult)

    delta = close.diff().abs()
    w = delta / close.shift(1)                 # weight = |Δclose| / prev_close
    w.iloc[0] = 0.0                            # first bar has no prev

    weighted_close = (close * w).rolling(length, min_periods=1).sum()
    sum_weights = w.rolling(length, min_periods=1).sum()

    ma = weighted_close / (sum_weights + 1e-10)

    # --- Dynamic range based on ATR ---
    atr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr_long = atr.rolling(2000, min_periods=200).mean()
    atr_long = atr_long.fillna(atr.rolling(200, min_periods=1).mean())

    range_atr = atr_long * mult

    osc = np.where(
        range_atr != 0,
        100 * (close - ma) / range_atr,
        np.nan
    )

    return pd.DataFrame({
        "zro_osc": pd.Series(osc, index=close.index)
    }, index=close.index)