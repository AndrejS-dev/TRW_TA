import pandas as pd
from trw_ta import register_outputs

@register_outputs('signal')
def momentum_acceleration(series: pd.Series, length: int = 13) -> pd.Series:
    """
    Momentum acceleration.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/v/BmXmwmnE/
    
    Parameters
    ----------
    series : pd.Series
        Input price or value series.
    length : int, default 13
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    velocity = series.diff(periods=length) / length
    velocity_sma = velocity.rolling(window=3).mean()
    acceleration = velocity_sma.diff(periods=length) / length
    psg = velocity_sma - acceleration

    # Directional signal
    signal = psg.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return signal
