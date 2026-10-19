import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('strength index')
def strength_index(series: pd.Series, length: int) -> pd.Series:
    """
    Strength index.
    
    Parameters
    ----------
    series : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    def corr_with_time(x):
        x = np.array(x)
        t = np.arange(len(x))
        if np.std(x) == 0:
            return 0  # avoid division by zero
        return np.corrcoef(t, x)[0, 1]

    return series.rolling(length).apply(corr_with_time, raw=False)
