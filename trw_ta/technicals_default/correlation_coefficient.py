import pandas as pd
from trw_ta import register_outputs

@register_outputs('cc')
def correlation_coefficient(source1: pd.Series, source2: pd.Series, length: int = 20) -> pd.Series:
    """
    Correlation Coefficient.
    
    Parameters
    ----------
    source1 : pd.Series
        Source1.
    source2 : pd.Series
        Source2.
    length : int, default 20
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    aligned = pd.concat([source1, source2], axis=1).dropna()
    s1 = aligned.iloc[:, 0]
    s2 = aligned.iloc[:, 1]

    # Rolling correlation
    corr = s1.rolling(window=length).corr(s2)
    return corr.reindex(source1.index)
