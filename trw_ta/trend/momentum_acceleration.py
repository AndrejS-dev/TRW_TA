import pandas as pd
from trw_ta import register_outputs

@register_outputs('signal')
def momentum_acceleration(series: pd.Series, length: int = 13) -> pd.Series:
    """https://www.tradingview.com/v/BmXmwmnE/"""
    velocity = series.diff(periods=length) / length
    velocity_sma = velocity.rolling(window=3).mean()
    acceleration = velocity_sma.diff(periods=length) / length
    psg = velocity_sma - acceleration

    # Directional signal
    signal = psg.apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return signal
