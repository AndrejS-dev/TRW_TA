import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('CG', 'Trigger')
def ehlers_center_of_gravity(source: pd.Series, length: int = 10) -> pd.DataFrame:
    """
    Ehlers center of gravity.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/script/gFkGeFH4-Ehlers-Center-of-Gravity-Oscillator-LazyBear/
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    length : int, default 10
        Lookback period.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    if not isinstance(source, pd.Series):
        raise TypeError("Source must be a pandas Series")
    if not isinstance(length, int) or length < 1:
        raise ValueError("Length must be a positive integer")
        
    cg = pd.Series(np.zeros(len(source)), index=source.index)
    
    for i in range(len(source)):
        if i < length - 1:
            cg.iloc[i] = 0
        else:
            window = source.iloc[i - length + 1:i + 1]
            nm = sum((j + 1) * price for j, price in enumerate(window[::-1]))
            dm = window.sum()
            cg.iloc[i] = -nm / dm + (length + 1) / 2.0 if dm != 0 else 0
    
    trigger = cg.shift(1).fillna(0)
    
    return pd.DataFrame({'cg': cg, 'trigger': trigger})