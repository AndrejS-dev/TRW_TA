import pandas as pd
import numpy as np
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('garch')
def garch_volatility(close: pd.Series, alpha_start=0.10, beta_start=0.80, ema_length=20) -> pd.Series:
    """
    Garch volatility.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/v/a6jqpmu4/
    
    Parameters
    ----------
    close : pd.Series
        Close price series.
    alpha_start : Any, default 0.1
        Alpha start.
    beta_start : Any, default 0.8
        Beta start.
    ema_length : Any, default 20
        Lookback / smoothing period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    ema = ta.ema(close, ema_length)
    variance = np.full_like(close.values, np.nan, dtype=float)

    for i in range(len(close)):
        if i == 0 or np.isnan(ema[i]):
            continue
        prev_var = variance[i - 1] if not np.isnan(variance[i - 1]) else 0
        squared_error = (close[i] - ema[i]) ** 2
        variance[i] = alpha_start * squared_error + beta_start * prev_var

    volatility = np.sqrt(variance)
    return pd.Series(volatility, index=close.index, name="garch_volatility")
