import pandas as pd

def correlation_coefficient(source1: pd.Series, source2: pd.Series, length: int = 20) -> pd.Series:
    aligned = pd.concat([source1, source2], axis=1).dropna()
    s1 = aligned.iloc[:, 0]
    s2 = aligned.iloc[:, 1]

    # Rolling correlation
    corr = s1.rolling(window=length).corr(s2)
    return corr.reindex(source1.index)
