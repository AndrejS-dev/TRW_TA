import pandas as pd
from trw_ta import register_outputs

@register_outputs('volume_summer')
def volume_summer(open_: pd.Series, close: pd.Series, volume: pd.Series, sum_len: int = 30, ema_len: int = 3, offset_bars: int = 0) -> pd.Series:
    """
    Volume summer.
    
    Parameters
    ----------
    open_ : pd.Series
        Open .
    close : pd.Series
        Close price series.
    volume : pd.Series
        Volume series.
    sum_len : int, default 30
        Lookback / smoothing period.
    ema_len : int, default 3
        Lookback / smoothing period.
    offset_bars : int, default 0
        Offset bars.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    signed_vol = volume.where(close > open_, -volume)
    vol_sum = signed_vol.rolling(window=sum_len).sum()
    volume_summer = vol_sum.ewm(span=ema_len, adjust=False).mean()

    if offset_bars != 0:
        volume_summer = volume_summer.shift(offset_bars)

    return volume_summer
