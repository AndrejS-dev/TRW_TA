import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('basis', 'upper', 'lower', 'trend')
def adaptive_trend_flow(high: pd.Series, low: pd.Series, close: pd.Series,
                        length: int = 10, smooth_len: int = 14, sensitivity: float = 2.0) -> pd.DataFrame:
    """https://www.tradingview.com/v/1ttpw8M3/"""
    typical = (high + low + close) / 3

    fast_ema = typical.ewm(span=length, adjust=False).mean()
    slow_ema = typical.ewm(span=length * 2, adjust=False).mean()
    basis = (fast_ema + slow_ema) / 2

    vol = typical.rolling(window=length).std()
    smooth_vol = vol.ewm(span=smooth_len, adjust=False).mean()

    upper = basis + smooth_vol * sensitivity
    lower = basis - smooth_vol * sensitivity

    trend = pd.Series(index=close.index, dtype=int)
    prev_level = np.nan

    for i in range(len(close)):
        if np.isnan(prev_level):
            trend.iloc[i] = 1 if close.iloc[i] > basis.iloc[i] else -1
            prev_level = lower.iloc[i] if trend.iloc[i] == 1 else upper.iloc[i]
        else:
            if trend.iloc[i - 1] == 1:
                if close.iloc[i] < lower.iloc[i]:
                    trend.iloc[i] = -1
                    prev_level = upper.iloc[i]
                else:
                    trend.iloc[i] = 1
                    prev_level = lower.iloc[i]
            else:
                if close.iloc[i] > upper.iloc[i]:
                    trend.iloc[i] = 1
                    prev_level = lower.iloc[i]
                else:
                    trend.iloc[i] = -1
                    prev_level = upper.iloc[i]

    return pd.DataFrame({
        "basis": basis,
        "upper": upper,
        "lower": lower,
        "trend": trend
    })
