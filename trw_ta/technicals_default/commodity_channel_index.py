import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('cci')
def commodity_channel_index(source: pd.Series, length: int = 20) -> pd.Series:
    """
    Commodity Channel Index (CCI).
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    length : int, default 20
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    ma = ta.sma(source, length)
    return (source - ma) / (0.015 * ta.dev(source, length))
