import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('lower', 'basis', 'upper')
def donchian_channels(high: pd.Series, low: pd.Series, length: int = 20) -> pd.Series:
    lower = ta.lowest(low, length)
    upper = ta.highest(high, length)
    basis = (lower + upper) / 2

    return pd.DataFrame({
        "lower": lower,
        "basis": basis,
        "upper": upper
    })