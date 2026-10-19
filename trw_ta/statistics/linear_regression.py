import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('linear regression')
def linear_regression(series: pd.Series, length: int) -> pd.Series:
    """
    Linear regression.
    
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
    def linreg(x):
        x = np.array(x)
        t = np.arange(len(x))
        A = np.vstack([t, np.ones(len(t))]).T
        m, b = np.linalg.lstsq(A, x, rcond=None)[0]
        return m * t[-1] + b
    return series.rolling(length).apply(linreg, raw=False)