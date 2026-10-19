import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('signal')
def hma_swing_points(high: pd.Series, low: pd.Series, x: int = 30, length: int = 5) -> pd.Series:
    """
    Hma swing points.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/script/5uXm51B0-Hma-Swing-Points-viResearch/
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    x : int, default 30
        Numeric input.
    length : int, default 5
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    xy = ta.hma(high, length)
    xz = ta.hma(low, length)

    highest = xy.rolling(window=x).max()
    lowest = xz.rolling(window=x).min()

    L = xy == highest
    S = xz == lowest

    vii = []
    last_vii = 0

    for i in range(len(xy)):
        if S.iloc[i]:
            last_vii = -1
        elif L.iloc[i] and not S.iloc[i]:
            last_vii = 1
        vii.append(last_vii)

    return pd.Series(vii, index=high.index)