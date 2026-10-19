import pandas as pd
from trw_ta import register_outputs

@register_outputs('bop')
def balance_of_power(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    Balance of Power.
    
    Parameters
    ----------
    open : pd.Series
        Open price series.
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return ((close - open) / (high - low))