import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('cl')
def continuation_index(src: pd.Series, gamma: float = 0.8, order: int = 8, length: int = 40) -> pd.DataFrame:
    """https://www.tradingview.com/script/5ZrOut79-TASC-2025-09-The-Continuation-Index/"""
    if not isinstance(src, pd.Series):
        raise TypeError("Source must be a pandas Series")
    if src.isna().any():
        raise ValueError("Source must not contain NaN values")
    if len(src) < max(length, 4):
        raise ValueError(f"Source must have at least {max(length, 4)} rows")
    if not 0 <= gamma <= 1:
        raise ValueError("Gamma must be between 0 and 1")
    if not 1 <= order <= 10:
        raise ValueError("Order must be between 1 and 10")
    if length < 3:
        raise ValueError("Length must be at least 3")

    def ultimate_smoother(src, period):
        a1 = np.exp(-1.414 * np.pi / period)
        c2 = 2.0 * a1 * np.cos(1.414 * np.pi / period)
        c3 = -a1 * a1
        c1 = (1.0 + c2 - c3) / 4.0
        us = pd.Series(np.nan, index=src.index)
        us.iloc[0] = src.iloc[0]  # Initialize first value
        for i in range(1, len(src)):
            if i < 4:
                us.iloc[i] = src.iloc[i]
            else:
                us.iloc[i] = (1.0 - c1) * src.iloc[i] + \
                             (2.0 * c1 - c2) * src.iloc[i-1] - \
                             (c1 + c3) * src.iloc[i-2] + \
                             c2 * (us.iloc[i-1] if not np.isnan(us.iloc[i-1]) else src.iloc[i-1]) + \
                             c3 * (us.iloc[i-2] if not np.isnan(us.iloc[i-2]) else src.iloc[i-2])
        return us.fillna(method='ffill')

    def laguerre_filter(src, gamma, order, length):
        lg = [ultimate_smoother(src, length)]  # L0 term
        for j in range(1, order):
            prev = lg[j-1]
            curr = pd.Series(np.nan, index=src.index)
            curr.iloc[0] = prev.iloc[0]
            for i in range(1, len(src)):
                curr.iloc[i] = gamma * (curr.iloc[i-1] - prev.iloc[i-1]) + prev.iloc[i-1]
            lg.append(curr.fillna(method='ffill'))
        return sum(lg) / order

    us = ultimate_smoother(src, int(length / 2))
    lg = laguerre_filter(src, gamma, order, length)
    diff = us - lg
    abs_diff = diff.abs()
    sma_abs_diff = abs_diff.rolling(window=length, min_periods=1).mean()
    # Add small epsilon to avoid division by zero
    ref = 2.0 * diff / (sma_abs_diff + 1e-10).where(sma_abs_diff > 1e-10, 1.0)
    # Clip ref to prevent extreme values
    ref = ref.clip(-10, 10)
    ci = (np.exp(2.0 * ref) - 1.0) / (np.exp(2.0 * ref) + 1.0)

    return pd.DataFrame({
        'cl': ci
    }, index=src.index)