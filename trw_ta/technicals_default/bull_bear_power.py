import pandas as pd
import trw_ta as ta

def bull_bear_power(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 13) -> pd.Series:
    bullPower = high - ta.ema(close, length)
    bearPower = low - ta.ema(close, length)
    return bullPower + bearPower