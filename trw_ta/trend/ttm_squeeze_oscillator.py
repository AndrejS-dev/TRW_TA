import pandas as pd
import numpy as np
import trw_ta as ta

def ttm_squeeze_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20) -> pd.Series:
    """https://www.tradingview.com/v/eFs5kf8F/"""
    e1 = (high.rolling(length).max() + low.rolling(length).min()) / 2 + close.rolling(length).mean()
    osc_input = close - e1 / 2
    osc = ta.linear_regression(osc_input, length)

    return osc
