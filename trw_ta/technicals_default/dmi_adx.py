from .. import ta_core as ta
import pandas as pd
import numpy as np

def dmi(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = ta.true_range(high, low, close)
    tr_smoothed = ta.rma(tr, length)

    plus_di = 100 * ta.rma(pd.Series(plus_dm, index=high.index), length) / tr_smoothed
    minus_di = 100 * ta.rma(pd.Series(minus_dm, index=high.index), length) / tr_smoothed

    return pd.DataFrame({
        "plus_di": plus_di,
        "minus_di": minus_di
    })

def adx(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14, adx_smoothing: int = 14) -> pd.DataFrame:
    up_move = high.diff()
    down_move = -low.diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = ta.true_range(high, low, close)
    tr_smoothed = ta.rma(tr, length)

    plus_di = 100 * ta.rma(pd.Series(plus_dm, index=high.index), length) / tr_smoothed
    minus_di = 100 * ta.rma(pd.Series(minus_dm, index=high.index), length) / tr_smoothed

    di_sum = plus_di + minus_di
    di_diff = (plus_di - minus_di).abs()
    dx = 100 * di_diff / di_sum.replace(0, np.nan)
    adx = ta.rma(dx, adx_smoothing)

    return pd.DataFrame(adx)