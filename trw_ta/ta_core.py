import pandas as pd
import numpy as np

def sma(series: pd.Series, length: int) -> pd.Series:
    """Simple Moving Average"""
    return series.rolling(window=length, min_periods=1).mean()

def ema(series: pd.Series, length: int) -> pd.Series:
    """Exponential Moving Average"""
    return series.ewm(span=length, adjust=False).mean()

def wma(series: pd.Series, length: int) -> pd.Series:
    """Weighted Moving Average"""
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hma(source: pd.Series, period: int) -> pd.Series:
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))

    wma_half = wma(source, half_length)
    wma_full = wma(source, period)

    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_length)
    return hma

def dema(source: pd.Series, length: int) -> pd.Series:
    e1 = ema(source, length)
    e2 = ema(e1, length)
    return 2 * e1 - e2

def tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    return pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

def atr1(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=length).mean()

def atr2(high: pd.Series, low: pd.Series, close: pd.Series, length: int) -> pd.Series:
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(alpha=1/length, adjust=False).mean()

def cum(series: pd.Series) -> pd.Series:
    """Cumulative sum"""
    return series.cumsum()

def nz(series: pd.Series, value: float = 0.0) -> pd.Series:
    """Replace NaN values with a given value (default is 0.0)"""
    return series.fillna(value)

def highest(series: pd.Series, length: int) -> pd.Series:
    """Return the highest value over a rolling window"""
    return series.rolling(window=length, min_periods=1).max()

def lowest(series: pd.Series, length: int) -> pd.Series:
    """Return the lowest value over a rolling window"""
    return series.rolling(window=length, min_periods=1).min()

def average_day_range(high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    return sma(high - low, length)

def rma(series: pd.Series, length: int) -> pd.Series:
    alpha = 1 / length
    return series.ewm(alpha=alpha, adjust=False).mean()

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

def stdev(source: pd.Series, length: int) -> pd.Series:
    return source.rolling(window=length).std()

def dev(source: pd.Series, length: int) -> pd.Series:
    mean = source.rolling(window=length).mean()

    def rolling_abs_dev(x):
        return np.mean(np.abs(x - x.mean()))

    dev = source.rolling(window=length).apply(rolling_abs_dev, raw=False)
    return dev

def lsma(source: pd.Series, length: int = 25, offset: int = 0) -> pd.Series:
    x = np.arange(length)

    def linear_regression(y):
        if len(y) < length:
            return np.nan
        # Linear regression coefficients
        A = np.vstack([x, np.ones(length)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return m * (length - 1 + offset) + b

    return source.rolling(length).apply(linear_regression, raw=True)

def median(source: pd.Series, length: int) -> pd.Series:
    return source.rolling(window=length).median()

def smma(src: pd.Series, length: int = 7) -> pd.Series:
    smma = pd.Series(index=src.index, dtype='float64')
    sma = src.rolling(window=length).mean()

    for i in range(len(src)):
        if i < length - 1:
            smma.iloc[i] = np.nan
        elif i == length - 1:
            smma.iloc[i] = sma.iloc[i]
        else:
            smma.iloc[i] = (smma.iloc[i - 1] * (length - 1) + src.iloc[i]) / length
    return smma

def tema(close: pd.Series, length: int = 9) -> pd.Series:
    ema1 = close.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()

    tema = 3 * (ema1 - ema2) + ema3
    return tema

def alma(series: pd.Series, length: int, offset: float, sigma: float) -> pd.Series:
    m = offset * (length - 1)
    s = length / sigma
    weights = np.array([np.exp(-((i - m) ** 2) / (2 * s ** 2)) for i in range(length)])
    weights /= weights.sum()
    return series.rolling(length).apply(lambda x: np.dot(x, weights), raw=True)

def ma(source: pd.Series, len: int, maType: str) -> pd.Series:
    match maType.upper():
        case "SMA":
            return sma(source, len)
        case "EMA":
            return ema(source, len)
        case "WMA":
            return wma(source, len)
        case "RMA":
            return rma(source, len)
        case "DEMA":
            return dema(source, len)
        case "HMA":
            return hma(source, len)
        case "LSMA":
            return lsma(source, len)
        case "SMMA":
            return smma(source, len)
        case "TEMA":
            return tema(source, len)
        case _:
            raise ValueError(f"Unsupported moving average type: {maType}")


def bollinger_bands(source: pd.Series, length: int = 20, mult: float = 2.0, maType: str = "SMA") -> pd.Series:
    basis = ma(source, length, maType)
    dev = mult * stdev(source, length)
    upper = basis + dev
    lower = basis - dev
    return pd.DataFrame({
         "upper_band": upper,
         "middle_band": basis,
         "lower_band": lower})

def rate_of_change(source: pd.Series, period: int) -> pd.Series:
    return source.diff(period) / source.shift(period) * 100

def momentum(source: pd.Series, length: int) -> pd.Series:
    return source.diff(periods=length)

def mad(source: pd.Series, length: int) -> pd.Series:
    median_ = median(source, length)
    abs_dev = abs(source - median_)
    return median(abs_dev, length)

def aad(source: pd.Series, length: int, avg_type: str) -> pd.Series:
    avg = ma(source, length, avg_type)
    abs_dev = abs(source - avg)
    return ma(abs_dev, length, avg_type)

def rmsd(source: pd.Series, benchmark: pd.Series, length: int) -> pd.Series:
    diff_squared = (source - benchmark) ** 2
    return diff_squared.rolling(window=length).mean().apply(np.sqrt)

def emd(source: pd.Series, benchmark: pd.Series, length: int) -> pd.Series:
    abs_dev = abs(source - benchmark)
    return ema(abs_dev, length)

def price_volume_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
    pvt = [0.0]
    for i in range(1, len(close)):
        prev_close = close.iloc[i - 1]
        if prev_close == 0:
            pvt_change = 0
        else:
            pvt_change = ((close.iloc[i] - prev_close) / prev_close) * volume.iloc[i]
        pvt.append(pvt[-1] + pvt_change)
    return pd.Series(pvt, index=close.index)

def keltner(close: pd.Series, high: pd.Series, low: pd.Series, length: int, mult: float) -> pd.Series:
    tr = tr(high, low, close)
    return close.ewm(span=length, adjust=False).mean() + mult * tr.ewm(span=length, adjust=False).mean()
