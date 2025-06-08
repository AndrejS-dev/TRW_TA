import pandas as pd
from trw_ta import register_outputs

@register_outputs('cmf')
def chaikin_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 20) -> pd.Series:
    ad = ((2*close-low-high)/(high-low))*volume
    mf = ad.rolling(window=length).sum() / volume.rolling(window=length).sum()
    return mf