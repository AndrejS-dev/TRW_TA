import pandas as pd
import numpy as np
from trw_ta import register_outputs
from trw_ta import hma

@register_outputs('trendspeed')
def trend_speed_analyzer(open_: pd.Series, close: pd.Series, max_length: int = 50, accel_multiplier: float = 5.0) -> pd.DataFrame:
    """https://www.tradingview.com/script/YUAV6jit-Trend-Speed-Analyzer-Zeiierman/"""
    counts_diff = close.diff().fillna(0)
    max_abs_diff = counts_diff.abs().rolling(200, min_periods=1).max()
    counts_diff_norm = (counts_diff + max_abs_diff) / (2 * max_abs_diff + 1e-10)
    dyn_length = 5 + counts_diff_norm * (max_length - 5)
    dyn_length = dyn_length.clip(lower=5, upper=max_length)

    prev_diff = counts_diff.shift(1).fillna(0)
    delta_counts_diff = np.abs(counts_diff - prev_diff)
    max_delta = delta_counts_diff.rolling(200, min_periods=1).max()
    max_delta = max_delta.replace(0, 1.0)
    accel_factor = delta_counts_diff / max_delta

    alpha_base = 2.0 / (dyn_length + 1)
    alpha = alpha_base * (1 + accel_factor * accel_multiplier)
    alpha = alpha.clip(upper=1.0)

    dyn_ema = pd.Series(np.nan, index=close.index, dtype=float)
    dyn_ema.iloc[0] = close.iloc[0]

    for i in range(1, len(close)):
        if pd.isna(close.iloc[i]):
            dyn_ema.iloc[i] = dyn_ema.iloc[i-1]
            continue
        a = alpha.iloc[i]
        dyn_ema.iloc[i] = a * close.iloc[i] + (1 - a) * dyn_ema.iloc[i-1]

    c = close.ewm(span=10, adjust=False, min_periods=1).mean()
    o = open_.ewm(span=10, adjust=False, min_periods=1).mean()
    speed_delta = c - o

    speed = speed_delta.cumsum()
    trendspeed = hma(speed, 5)

    return pd.DataFrame({
        'trendspeed': trendspeed,
    }).rename_axis(None, axis=1)