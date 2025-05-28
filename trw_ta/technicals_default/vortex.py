import pandas as pd
import numpy as np
import trw_ta as ta

def vortex(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.DataFrame:
    vm_plus = (high - low.shift(1)).abs()
    vm_minus = (low - high.shift(1)).abs()
    tr = ta.atr2(high, low, close, 1)

    vmp = vm_plus.rolling(length).sum()
    vmm = vm_minus.rolling(length).sum()
    str_sum = tr.rolling(length).sum()

    vip = vmp / str_sum
    vim = vmm / str_sum

    return pd.DataFrame({
        'VIP': vip,
        'VIM': vim
    }, index=high.index)
