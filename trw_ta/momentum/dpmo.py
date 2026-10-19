import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('PMO', 'PMO_signal', 'histogram')
def dpmo(close: pd.Series, length1: int = 35, length2: int = 20, siglength: int = 10,) -> pd.DataFrame:
    """
    Dpmo.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/v/5e9WBJwE/
    
    Parameters
    ----------
    close : pd.Series
        Close price series.
    length1 : int, default 35
        Length1.
    length2 : int, default 20
        Length2.
    siglength : int, default 10
        Siglength.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    def csf(src: pd.Series, length: int) -> pd.Series:
        alpha = 2 / length
        out = pd.Series(index=src.index, dtype=float)
        out.iloc[0] = 0
        for i in range(1, len(src)):
            prev = out.iloc[i - 1] if not np.isnan(out.iloc[i - 1]) else 0
            out.iloc[i] = (src.iloc[i] - prev) * alpha + prev
        return out

    i = (close / close.shift(1).fillna(close)) * 100
    pmol2 = csf(i - 100, length1)
    pmol = csf(10 * pmol2, length2)
    pmols = pmol.ewm(span=siglength, adjust=False).mean()
    histogram = pmol - pmols

    return pd.DataFrame({
        "PMO": pmol,
        "PMO_signal": pmols,
        "histogram": histogram
    })
