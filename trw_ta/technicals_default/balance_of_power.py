import pandas as pd
from trw_ta import register_outputs

@register_outputs('bop')
def balance_of_power(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return ((close - open) / (high - low))