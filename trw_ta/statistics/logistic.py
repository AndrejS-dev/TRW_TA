import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('log_prob')
def to_normalized_logistic_prob(series: pd.Series, lookback: int = 100, steepness: float = 1.0, use_rolling_mean: bool = True) -> pd.Series:
    """
    To normalized logistic prob.
    
    Parameters
    ----------
    series : pd.Series
        Input price or value series.
    lookback : int, default 100
        Lookback.
    steepness : float, default 1.0
        Steepness.
    use_rolling_mean : bool, default True
        Use rolling mean.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    if use_rolling_mean:
        mean = series.rolling(lookback, min_periods=1).mean()
    else:
        mean = series.mean()
    
    deviation = series - mean
    z = steepness * deviation
    prob = 1.0 / (1.0 + np.exp(-z))
    
    return pd.Series(prob, index=series.index, name='logistic_prob')