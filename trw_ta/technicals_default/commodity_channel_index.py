import pandas as pd
import trw_ta as ta

def commodity_channel_index(source: pd.Series, length: int = 20) -> pd.Series:
    ma = ta.sma(source, length)
    return (source - ma) / (0.015 * ta.dev(source, length))
