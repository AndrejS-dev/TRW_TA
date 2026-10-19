import pandas as pd
from trw_ta import register_outputs

@register_outputs('cmf')
def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 20) -> pd.Series:
    """
    Chaikin Money Flow (CMF).
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    volume : pd.Series
        Volume series.
    length : int, default 20
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    ad = ((2*close-low-high)/(high-low))*volume
    mf = ad.rolling(window=length).sum() / volume.rolling(window=length).sum()
    return mf