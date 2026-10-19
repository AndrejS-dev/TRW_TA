import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('mfi')
def money_flow_index(src: pd.Series, volume: pd.Series, length: int = 14) -> pd.Series:
    """
    Money Flow Index (MFI).
    
    Parameters
    ----------
    src : pd.Series
        Input price or value series.
    volume : pd.Series
        Volume series.
    length : int, default 14
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    delta = src.diff()
    upper = volume * np.where(delta <= 0, 0.0, src)
    lower = volume * np.where(delta >= 0, 0.0, src)

    upper_sum = upper.rolling(length).sum()
    lower_sum = lower.rolling(length).sum()

    mfi = 100.0 - (100.0 / (1.0 + (upper_sum / lower_sum.replace(0, np.nan))))
    return mfi
