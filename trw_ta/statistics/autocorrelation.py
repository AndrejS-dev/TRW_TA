import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('autocorrelation')
def autocorrelation(series: pd.Series, length: int, lag: int = 1) -> pd.Series:
    """
    Autocorrelation.
    
    Parameters
    ----------
    series : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    lag : int, default 1
        Lag.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    def autocorr(x):
        x = np.array(x)
        if np.std(x) == 0:
            return 0  # Avoid division by zero
        return np.corrcoef(x[:-lag], x[lag:])[0, 1]
    
    return series.rolling(length).apply(autocorr, raw=False)
