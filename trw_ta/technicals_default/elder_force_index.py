import pandas as pd
import trw_ta as ta

def elder_force_index(close: pd.Series, volume: pd.Series, length: int = 13) -> pd.Series:
    return ta.ema(close.diff()* volume, length)
