import pandas as pd
import numpy as np
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('psar', 'score')
def dynamic_score_psar(high: pd.Series, low: pd.Series, close: pd.Series,
                       psar_start: float = 0.02, psar_increment: float = 0.0005, psar_max: float = 0.2,
                       window_len: int = 60, uptrend_threshold: int = 30, downtrend_threshold: int = -20) -> pd.DataFrame:
    """https://www.tradingview.com/v/KAWg6T4j/"""
    psar_out = ta.sar(high, low, close, psar_start, psar_increment, psar_max)
    psar = close - psar_out
    ema_range = (high - low).ewm(span=21, adjust=False).mean()
    normalized_psar = psar / ema_range * 100

    trend_score = pd.Series(index=normalized_psar.index, dtype=float)
    for i in range(window_len, len(normalized_psar)):
        score = 0
        for j in range(1, window_len + 1):
            score += 1 if normalized_psar.iloc[i] > normalized_psar.iloc[i - j] else -1
        trend_score.iloc[i] = score

    trend_score.fillna(0, inplace=True)

    long_condition = (normalized_psar > 0) & (trend_score > uptrend_threshold)
    short_condition = (normalized_psar < 0) & (trend_score < downtrend_threshold)

    return pd.DataFrame({
        "psar": normalized_psar,
        "score": trend_score
    })
