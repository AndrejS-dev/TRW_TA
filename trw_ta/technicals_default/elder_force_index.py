import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('efi')
def elder_force_index(close: pd.Series, volume: pd.Series, length: int = 13) -> pd.Series:
    """
    Elder Force Index.
    
    Parameters
    ----------
    close : pd.Series
        Close price series.
    volume : pd.Series
        Volume series.
    length : int, default 13
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return ta.ema(close.diff()* volume, length)
