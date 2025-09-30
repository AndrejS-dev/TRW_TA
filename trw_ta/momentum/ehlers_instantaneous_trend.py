import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('it', 'lag')
def ehlers_instantaneous_trend(high: pd.Series, low: pd.Series, alpha: float = 0.07) -> pd.DataFrame:
    """https://www.tradingview.com/v/DaHLcICg/"""
    src = (high + low) / 2
    it = np.full(len(src), np.nan)

    for i in range(2, len(src)):
        src0 = src.iloc[i]
        src1 = src.iloc[i - 1]
        src2 = src.iloc[i - 2]
        
        it_1 = it[i - 1] if not np.isnan(it[i - 1]) else (src0 + 2 * src1 + src2) / 4
        it_2 = it[i - 2] if not np.isnan(it[i - 2]) else (src0 + 2 * src1 + src2) / 4

        it[i] = (
            (alpha - (alpha ** 2) / 4.0) * src0 +
            0.5 * alpha ** 2 * src1 -
            (alpha - 0.75 * alpha ** 2) * src2 +
            2 * (1 - alpha) * it_1 -
            (1 - alpha) ** 2 * it_2
        )

    it_series = pd.Series(it, index=src.index)
    lag = 2 * it_series - it_series.shift(2)

    return pd.DataFrame({
        'it': it_series,
        'lag': lag
    })
