import pandas as pd

def balance_of_power(open: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return ((close - open) / (high - low))