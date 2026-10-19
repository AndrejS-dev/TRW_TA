import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('score', 'signal', 'trend_state')
def quartile_for_loop(close: pd.Series, q_length: int = 14, sl: int = 5, el: int = 55, thr_up: float = 35.0, 
                      thr_down: float = -5.0, type_sig: bool = False) -> pd.DataFrame:
    """
    Quartile for loop.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/script/6OezfUK6-Quartile-For-Loop-SeerQuant/
    
    Parameters
    ----------
    close : pd.Series
        Close price series.
    q_length : int, default 14
        Lookback / smoothing period.
    sl : int, default 5
        Sl.
    el : int, default 55
        El.
    thr_up : float, default 35.0
        Thr up.
    thr_down : float, default -5.0
        Thr down.
    type_sig : bool, default False
        Type sig.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    # Weighted quartile function
    def f_quartile(src: pd.Series, length: int) -> pd.Series:
        q1 = src.rolling(length).quantile(0.25)
        median = src.rolling(length).quantile(0.5)
        q3 = src.rolling(length).quantile(0.75)
        return (q1 + 2 * median + q3) / 4

    # For-loop based score calculation
    def calc_score(src: pd.Series, val: pd.Series, alt: bool, start: int, end: int) -> pd.Series:
        score = pd.Series(0.0, index=src.index)
        for i in range(end, len(src)):
            sum_score = 0
            for j in range(start, end + 1):
                if alt:
                    sum_score += 1 if val.iloc[i] > val.iloc[i - j] else -1
                else:
                    sum_score += 1 if src.iloc[i] > val.iloc[i - j] else -1
            score.iloc[i] = sum_score
        return score

    adaptive_quartile = f_quartile(close, q_length)
    score = calc_score(close, adaptive_quartile, type_sig, sl, el)

    signal = pd.Series(0, index=close.index)
    signal[(score > thr_up) & ~(score < thr_down)] = 1
    signal[score < thr_down] = -1

    trend_state = pd.Series(0, index=close.index)
    for i in range(1, len(score)):
        if score.iloc[i] > thr_up:
            trend_state.iloc[i] = 1
        elif score.iloc[i] < thr_down:
            trend_state.iloc[i] = -1
        else:
            trend_state.iloc[i] = trend_state.iloc[i - 1]

    return pd.DataFrame({
        "score": score,
        "signal": signal,
        "trend_state": trend_state
    })
