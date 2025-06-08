import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('rci')
def rank_correlation_index(src: pd.Series, length: int = 10) -> pd.Series:
    rci_vals = []

    for i in range(len(src)):
        if i < length - 1:
            rci_vals.append(np.nan)
            continue
        
        # Subset of data
        window = src.iloc[i - length + 1:i + 1]

        # Price ranks (descending)
        price_ranks = window.rank(method="first", ascending=False)
        # Time ranks (fixed 1 to length)
        time_ranks = pd.Series(range(length, 0, -1), index=window.index)

        # Sum of squared rank differences
        d_sq = ((price_ranks - time_ranks) ** 2).sum()

        # RCI formula
        rci = (1 - (6 * d_sq) / (length * (length**2 - 1))) * 100
        rci_vals.append(rci)

    return pd.Series(rci_vals, index=src.index)
