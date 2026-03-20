import pandas as pd
import numpy as np
from trw_ta import register_outputs


@register_outputs('mcd', 'smooth_mcd', 'sig')
def dynamic_median_momentum_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, basis_length: int = 21,
    smooth_length: int = 21, signal_length: int = 14, standardize: bool = True) -> pd.DataFrame:
    """https://www.tradingview.com/script/EDgeowAw-Dynamic-Median-Momentum-Oscillator-AlgoAlpha/"""
    hlc3 = (high + low + close) / 3
    basis = hlc3.rolling(basis_length, min_periods=1).median()
    range_basis = (high - low).ewm(span=basis_length, adjust=False, min_periods=1).mean()

    raw_mcd = close - basis

    if standardize and range_basis.ne(0).any():
        mcd = raw_mcd / range_basis * 100
    else:
        mcd = raw_mcd

    smooth_mcd = mcd.ewm(span=smooth_length, adjust=False, min_periods=1).mean()

    sig = smooth_mcd.ewm(span=signal_length, adjust=False, min_periods=1).mean()

    return pd.DataFrame({
        'mcd':        mcd,
        'smooth_mcd': smooth_mcd,
        'sig':        sig,
    }).rename_axis(None, axis=1)