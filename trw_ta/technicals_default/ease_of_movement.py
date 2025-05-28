import pandas as pd
import trw_ta as ta

def ease_of_movement(high: pd.Series, low: pd.Series, volume: pd.Series, length: int = 14, div: int = 10000) -> pd.Series:
    hl2 = (high + low) / 2
    return ta.sma(div * hl2.diff() * (high- low) / volume, length)