import pandas as pd
import trw_ta as ta

def donchian_channels(high: pd.Series, low: pd.Series, length: int = 20) -> pd.Series:
    lower = ta.lowest(low, length)
    upper = ta.highest(high, length)
    basis = (lower + upper) / 2

    return pd.DataFrame({
        "lower": lower,
        "basis": basis,
        "upper": upper
    })