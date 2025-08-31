import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('signalLine', 'support', 'resistance')
def trendlines_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 5, memory: int = 10, 
                                           data_smoothing: str = 'NONE', data_smoothing_length: int = 10, 
                                           signal_smoothing: bool = True, signal_smoothing_type: str = 'TMA', 
                                           signal_smoothing_length: int = 10) -> pd.DataFrame:
    """https://www.tradingview.com/script/3oeha5F1-Trendlines-Oscillator-LuxAlgo/"""
    if not all(isinstance(s, pd.Series) for s in [high, low, close]):
        raise TypeError("Inputs must be pandas Series")
    if not (high.index.equals(low.index) and low.index.equals(close.index)):
        raise ValueError("Input Series must have the same index")
    if high.isna().any() or low.isna().any() or close.isna().any():
        raise ValueError("Input Series must not contain NaN values")
    if len(high) < length * 2 + 1:
        raise ValueError(f"Input Series must have at least {length * 2 + 1} rows")
    if length < 1 or memory < 1 or data_smoothing_length < 2 or signal_smoothing_length < 2:
        raise ValueError("Length parameters must be positive, smoothing lengths >= 2")

    def smooth(data, smoothing, length):
        if smoothing == 'RMA':
            return data.ewm(alpha=1/length, adjust=False).mean()
        elif smoothing == 'SMA':
            return data.rolling(window=length, min_periods=1).mean()
        elif smoothing == 'TMA':
            sma1 = data.rolling(window=length, min_periods=1).mean()
            return sma1.rolling(window=length, min_periods=1).mean()
        elif smoothing == 'EMA':
            return data.ewm(span=length, adjust=False).mean()
        elif smoothing == 'DEMA':
            ema1 = data.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            return 2 * ema1 - ema2
        elif smoothing == 'TEMA':
            ema1 = data.ewm(span=length, adjust=False).mean()
            ema2 = ema1.ewm(span=length, adjust=False).mean()
            ema3 = ema2.ewm(span=length, adjust=False).mean()
            return 3 * ema1 - 3 * ema2 + ema3
        elif smoothing == 'HMA':
            half_period = max(2, length // 2)
            wma_half = data.rolling(window=half_period, min_periods=1).apply(
                lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)
            wma_full = data.rolling(window=length, min_periods=1).apply(
                lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)
            hma_raw = 2 * wma_half - wma_full
            sqrt_period = max(2, int(np.sqrt(length)))
            return hma_raw.rolling(window=sqrt_period, min_periods=1).apply(
                lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)
        elif smoothing == 'WMA':
            return data.rolling(window=length, min_periods=1).apply(
                lambda x: np.average(x, weights=np.arange(1, len(x) + 1)), raw=True)
        elif smoothing == 'SWMA':
            weights = [0.119, 0.238, 0.381, 0.262]
            len_weights = min(length, len(weights))
            return data.rolling(window=len_weights, min_periods=1).apply(
                lambda x: np.average(x[-len_weights:], weights=weights[-len_weights:]), raw=True)
        elif smoothing == 'VWMA':
            volume = pd.Series(1, index=data.index)  # Placeholder; requires volume data
            return (data * volume).rolling(window=length, min_periods=1).sum() / volume.rolling(window=length, min_periods=1).sum()
        else:
            return data

    def pivothigh(high, left, right):
        window = left + right + 1
        rolling_max = high.rolling(window=window, center=True, min_periods=1).max()
        ph = pd.Series(np.nan, index=high.index)
        ph[high == rolling_max] = high[high == rolling_max]
        return ph

    def pivotlow(low, left, right):
        window = left + right + 1
        rolling_min = low.rolling(window=window, center=True, min_periods=1).min()
        pl = pd.Series(np.nan, index=low.index)
        pl[low == rolling_min] = low[low == rolling_min]
        return pl

    res_lines = []
    sup_lines = []
    phx1 = 0
    plx1 = 0

    ph = pivothigh(high, length, length).fillna(method='ffill').fillna(high)
    pl = pivotlow(low, length, length).fillna(method='ffill').fillna(low)

    n = pd.Series(range(len(high)), index=high.index)
    sup_sum = pd.Series(0.0, index=high.index)
    sup_den = pd.Series(0.0, index=high.index)
    res_sum = pd.Series(0.0, index=high.index)
    res_den = pd.Series(0.0, index=high.index)

    for i in range(len(high)):
        if i > 0 and ph.iloc[i] < ph.iloc[i-1]:
            slope = (ph.iloc[i] - ph.iloc[i-1]) / (i - length - phx1)
            res_lines.insert(0, {'intercept': ph.iloc[i-1] - slope * phx1, 'slope': slope})
            phx1 = i - length

        if i > 0 and pl.iloc[i] > pl.iloc[i-1]:
            slope = (pl.iloc[i] - pl.iloc[i-1]) / (i - length - plx1)
            sup_lines.insert(0, {'intercept': pl.iloc[i-1] - slope * plx1, 'slope': slope})
            plx1 = i - length

        if len(res_lines) > memory:
            res_lines.pop()
        if len(sup_lines) > memory:
            sup_lines.pop()

        for sup in sup_lines:
            point = sup['slope'] * i + sup['intercept']
            if close.iloc[i] > point:
                sup_sum.iloc[i] += close.iloc[i] - point
            sup_den.iloc[i] += abs(close.iloc[i] - point)

        for res in res_lines:
            point = res['slope'] * i + res['intercept']
            if close.iloc[i] < point:
                res_sum.iloc[i] += point - close.iloc[i]
            res_den.iloc[i] += abs(close.iloc[i] - point)

    support_line = (sup_sum / sup_den.where(sup_den != 0, 1)) * 100
    resistance_line = (res_sum / res_den.where(res_den != 0, 1)) * 100

    smooth_support = smooth(support_line, data_smoothing, data_smoothing_length).fillna(0)
    smooth_resistance = smooth(resistance_line, data_smoothing, data_smoothing_length).fillna(0)

    signal_line = abs(smooth_support - smooth_resistance)
    signal_line = smooth(signal_line, signal_smoothing_type, signal_smoothing_length).fillna(0) if signal_smoothing else pd.Series(0.0, index=close.index)

    return pd.DataFrame({
        'signalLine': signal_line,
        'support': smooth_support,
        'resistance': smooth_resistance
    }, index=close.index)