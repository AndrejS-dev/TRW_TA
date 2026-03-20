import pandas as pd
import numpy as np
from trw_ta import register_outputs

def adaptive_momentum_raw(source: pd.Series, length: int = 14) -> pd.Series:
    n = len(source)
    amo = pd.Series(np.nan, index=source.index, dtype=float)

    for i in range(length, n):
        window = source.iloc[i - length:i + 1]
        deltas = source.iloc[i] - window[:-1]
        abs_deltas = np.abs(deltas)
        max_abs = abs_deltas.max()
        if max_abs > 0:
            idx = np.argmax(abs_deltas)
            amo.iloc[i] = deltas.iloc[idx]
        else:
            amo.iloc[i] = 0.0

    return amo

def linreg_end_value(s: pd.Series) -> float:
    if len(s) < 2 or s.isna().all():
        return np.nan
    valid = s.dropna()
    if len(valid) < 2:
        return valid.iloc[-1] if len(valid) > 0 else np.nan
    x = np.arange(len(valid))
    y = valid.values
    slope, intercept = np.polyfit(x, y, 1)
    return slope * (len(valid) - 1) + intercept

def adaptive_moving_average(series: pd.Series, length: int = 9) -> pd.Series:
    n = len(series)
    ama = pd.Series(np.nan, index=series.index, dtype=float)

    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        return ama

    start_i = series.index.get_loc(first_valid_idx)
    ama.iloc[start_i] = series.iloc[start_i]

    for i in range(start_i + 1, n):
        if pd.isna(series.iloc[i]):
            ama.iloc[i] = ama.iloc[i - 1]
            continue

        window = series.iloc[max(0, i - length + 1):i + 1].dropna()
        if len(window) < 2:
            ama.iloc[i] = ama.iloc[i - 1] if not pd.isna(ama.iloc[i - 1]) else series.iloc[i]
            continue

        change = np.abs(series.iloc[i] - window.iloc[0])
        volatility = np.abs(window.diff()).sum()
        er = change / volatility if volatility > 0 else 0.0

        prev_ama = ama.iloc[i - 1] if not pd.isna(ama.iloc[i - 1]) else series.iloc[i]
        ama.iloc[i] = prev_ama + er * (series.iloc[i] - prev_ama)

    return ama


@register_outputs('amo', 'ama')
def adaptive_momentum_oscillator(source: pd.Series, length: int = 14, smoothing_length: int = 9) -> pd.DataFrame:
    """https://www.tradingview.com/script/Uqk8zpzT-Adaptive-Momentum-Oscillator-LuxAlgo/"""
    raw_amo = adaptive_momentum_raw(source, length=length)
    amo = raw_amo.rolling(smoothing_length, min_periods=1).apply(linreg_end_value, raw=False)
    ama = adaptive_moving_average(amo, length=smoothing_length)

    return pd.DataFrame({
        'amo': amo,
        'ama': ama,
    }).rename_axis(None, axis=1)