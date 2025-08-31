import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('mid', 'top', 'bot')
def pure_price_zone_flow(high: pd.Series, low: pd.Series, close: pd.Series, len: int = 20, atrMult: float = 1.0) -> pd.DataFrame:
    """https://www.tradingview.com/script/87oQcPHN-Pure-Price-Zone-Flow/"""
    if not all(isinstance(s, pd.Series) for s in [high, low, close]):
        raise TypeError("Inputs must be pandas Series")
    if not (high.index.equals(low.index) and low.index.equals(close.index)):
        raise ValueError("Input Series must have the same index")
    if high.isna().any() or low.isna().any() or close.isna().any():
        raise ValueError("Input Series must not contain NaN values")
    if len < 2 or atrMult < 0.1:
        raise ValueError("len must be >= 2, atrMult must be >= 0.1")

    # Custom ATR calculation
    def atr(high, low, close, period):
        tr = pd.DataFrame({
            'hl': high - low,
            'hc': (high - close.shift(1)).abs(),
            'lc': (low - close.shift(1)).abs()
        }).max(axis=1)
        return tr.rolling(window=period, min_periods=1).mean()

    swing_high = high.rolling(window=len, min_periods=1).max()
    swing_low = low.rolling(window=len, min_periods=1).min()
    atr_value = atr(high, low, close, len)

    mid = (swing_high + swing_low) / 2
    top = swing_high + atr_value * atrMult
    bot = swing_low - atr_value * atrMult

    return pd.DataFrame({
        'mid': mid,
        'top': top,
        'bot': bot
    }, index=high.index)