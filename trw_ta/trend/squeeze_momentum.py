import pandas as pd
import numpy as np
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('val')
def squeeze_momentum(high: pd.Series, low: pd.Series, close: pd.Series, bb_length: int = 20, bb_mult: float = 2.0, kc_length: int = 20, kc_mult: float = 1.5, use_tr: bool = True) -> pd.DataFrame:
    """https://www.tradingview.com/v/nqQ1DT5a/"""
    source = close
    basis = close.rolling(window=bb_length).mean()
    dev = close.rolling(window=bb_length).std() * bb_mult
    upper_bb = basis + dev
    lower_bb = basis - dev

    ma = close.rolling(window=kc_length).mean()
    if use_tr:
        range_ = ta.tr(high, low, close)
    else:
        range_ = high - low
    rangema = range_.rolling(window=kc_length).mean()
    upper_kc = ma + rangema * kc_mult
    lower_kc = ma - rangema * kc_mult

    # Squeeze conditions
    sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
    no_sqz = ~(sqz_on | sqz_off)

    # Momentum value
    avg_hl = (high.rolling(kc_length).max() + low.rolling(kc_length).min()) / 2
    avg_all = ((avg_hl + close.rolling(kc_length).mean()) / 2)
    val = ta.linear_regression(source - avg_all, kc_length)

    # Bar color logic
    val_prev = val.shift(1)
    bcolor = np.where(val > 0,
                      np.where(val > val_prev, "lime", "green"),
                      np.where(val < val_prev, "red", "maroon"))

    scolor = np.where(no_sqz, "blue",
              np.where(sqz_on, "black", "gray"))

    return pd.DataFrame({
        "squeeze_momentum": val,
    }, index=close.index)
