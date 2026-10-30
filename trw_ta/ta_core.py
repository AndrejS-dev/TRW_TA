import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('sma')
def sma(series: pd.Series, length: int) -> pd.Series:
    """
    Simple Moving Average (SMA).
    
    Parameters
    ----------
    series : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return series.rolling(window=length, min_periods=1).mean()

@register_outputs('ema')
def ema(series: pd.Series, length: int) -> pd.Series:
    """
    Exponential Moving Average (EMA).
    
    Parameters
    ----------
    series : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return series.ewm(span=length, adjust=False).mean()

@register_outputs('wma')
def wma(series: pd.Series, length: int) -> pd.Series:
    """
    Weighted Moving Average (WMA).
    
    Parameters
    ----------
    series : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    weights = np.arange(1, length + 1)
    return series.rolling(length).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

@register_outputs('hma')
def hma(source: pd.Series, period: int) -> pd.Series:
    """
    Hull Moving Average (HMA).
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    period : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    half_length = int(period / 2)
    sqrt_length = int(np.sqrt(period))

    wma_half = wma(source, half_length)
    wma_full = wma(source, period)

    diff = 2 * wma_half - wma_full
    hma = wma(diff, sqrt_length)
    return hma

@register_outputs('dema')
def dema(source: pd.Series, length: int) -> pd.Series:
    """
    Double Exponential Moving Average (DEMA).
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    e1 = ema(source, length)
    e2 = ema(e1, length)
    return 2 * e1 - e2

@register_outputs('tr')
def tr(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    True Range.
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)

@register_outputs('atr_sma')
def atr1(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """
    Average True Range (SMA of True Range).
    
    ATR using simple moving average of true range.
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    length : int, default 14
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    tr_ = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low - close.shift()).abs()
    ], axis=1).max(axis=1)
    return tr_.rolling(window=length).mean()

@register_outputs('atr_rma')
def atr2(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """
    Average True Range (RMA / Wilder of True Range).
    
    ATR using Wilder/RMA smoothing of true range.
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    length : int, default 14
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr_ = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr_.ewm(alpha=1/length, adjust=False).mean()

@register_outputs('atr')
def atr(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> pd.Series:
    """
    Average True Range (ATR).
    
    Default ATR (SMA of true range). Prefer atr_sma / atr_rma for explicit intent.
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    length : int, default 14
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return atr1(high, low, close, length)

# Stable aliases
atr_sma = atr1
atr_rma = atr2

@register_outputs('cum')
def cum(series: pd.Series) -> pd.Series:
    """
    Cumulative Sum.
    
    Parameters
    ----------
    series : pd.Series
        Input price or value series.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return series.cumsum()

@register_outputs('nz')
def nz(series: pd.Series, value: float = 0.0) -> pd.Series:
    """
    Replace NaN Values.
    
    Replace NaN values with a given value (default is 0.0).
    
    Parameters
    ----------
    series : pd.Series
        Input price or value series.
    value : float, default 0.0
        Replacement value for NaNs.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return series.fillna(value)

@register_outputs('highest')
def highest(series: pd.Series, length: int) -> pd.Series:
    """
    Rolling Highest.
    
    Return the highest value over a rolling window.
    
    Parameters
    ----------
    series : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return series.rolling(window=length, min_periods=1).max()

@register_outputs('lowest')
def lowest(series: pd.Series, length: int) -> pd.Series:
    """
    Rolling Lowest.
    
    Return the lowest value over a rolling window.
    
    Parameters
    ----------
    series : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return series.rolling(window=length, min_periods=1).min()

@register_outputs('average_day_range')
def average_day_range(high: pd.Series, low: pd.Series, length: int) -> pd.Series:
    """
    Average Day Range.
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return sma(high - low, length)

@register_outputs('rma')
def rma(series: pd.Series, length: int) -> pd.Series:
    """
    Wilder's Moving Average (RMA).
    
    Parameters
    ----------
    series : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    alpha = 1 / length
    return series.ewm(alpha=alpha, adjust=False).mean()

@register_outputs('tr')
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """
    True Range.
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr

@register_outputs('stdev')
def stdev(source: pd.Series, length: int) -> pd.Series:
    """
    Rolling Standard Deviation.
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return source.rolling(window=length).std()

@register_outputs('dev')
def dev(source: pd.Series, length: int) -> pd.Series:
    """
    Mean Absolute Deviation.
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    def rolling_abs_dev(x):
        return np.mean(np.abs(x - x.mean()))

    dev = source.rolling(window=length).apply(rolling_abs_dev, raw=False)
    return dev

@register_outputs('lsma')
def lsma(source: pd.Series, length: int = 25, offset: int = 0) -> pd.Series:
    """
    Least Squares Moving Average (LSMA).
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    length : int, default 25
        Lookback period.
    offset : int, default 0
        Offset parameter.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    x = np.arange(length)

    def linear_regression(y):
        if len(y) < length:
            return np.nan
        # Linear regression coefficients
        A = np.vstack([x, np.ones(length)]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return m * (length - 1 + offset) + b

    return source.rolling(length).apply(linear_regression, raw=True)

@register_outputs('median')
def median(source: pd.Series, length: int) -> pd.Series:
    """
    Rolling Median.
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return source.rolling(window=length).median()

@register_outputs('smma')
def smma(src: pd.Series, length: int = 7) -> pd.Series:
    """
    Smoothed Moving Average (SMMA).
    
    Parameters
    ----------
    src : pd.Series
        Input price or value series.
    length : int, default 7
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
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

@register_outputs('tema')
def tema(close: pd.Series, length: int = 9) -> pd.Series:
    """
    Triple Exponential Moving Average (TEMA).
    
    Parameters
    ----------
    close : pd.Series
        Close price series.
    length : int, default 9
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    ema1 = close.ewm(span=length, adjust=False).mean()
    ema2 = ema1.ewm(span=length, adjust=False).mean()
    ema3 = ema2.ewm(span=length, adjust=False).mean()

    tema = 3 * (ema1 - ema2) + ema3
    return tema

@register_outputs('alma')
def alma(
    source: pd.Series,
    length: int = 9,
    offset: float = 0.85,
    sigma: float = 6.0,
    *,
    window: int | None = None,
    series: pd.Series | None = None,
) -> pd.Series:
    """
    Arnaud Legoux Moving Average (ALMA).

    Parameters
    ----------
    source : pd.Series
        Input series. ``series`` is accepted as a deprecated alias.
    length : int, default 9
        Window length. ``window`` is accepted as a deprecated alias.
    offset : float, default 0.85
        ALMA offset parameter in ``[0, 1]``.
    sigma : float, default 6.0
        Gaussian width parameter.
    window : int or None, optional
        Deprecated alias for ``length``.
    series : pd.Series or None, optional
        Deprecated alias for ``source``.

    Returns
    -------
    pd.Series
        Series of computed values.
    """
    if series is not None:
        source = series
    if window is not None:
        length = window
    m = offset * (length - 1)
    s = length / sigma
    weights = np.array([np.exp(-((i - m) ** 2) / (2 * s ** 2)) for i in range(length)])
    weights /= weights.sum()
    return source.rolling(length).apply(lambda x: np.dot(x, weights), raw=True)

@register_outputs('zlema')
def zlema(src: pd.Series, length: int = 14) -> pd.Series:
    """
    Zero-Lag Exponential Moving Average (ZLEMA).
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/v/Oxizu1k7/
    
    Parameters
    ----------
    src : pd.Series
        Input price or value series.
    length : int, default 14
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    lag = (length - 1) // 2
    shifted_src = src.shift(lag)
    adjusted_src = src + (src - shifted_src)
    return adjusted_src.ewm(span=length, adjust=False).mean()

@register_outputs('moving average')
def ma(
    source: pd.Series,
    length: int | None = None,
    ma_type: str = "SMA",
    *,
    len: int | None = None,
    maType: str | None = None,
) -> pd.Series:
    """
    Moving Average Dispatcher.
    
    Moving-average dispatcher.
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    length : int | None, default None
        Lookback period.
    ma_type : str, default 'SMA'
        Moving-average type (e.g. SMA, EMA, WMA).
    len : int | None, default None
        Lookback period (deprecated alias for length).
    maType : str | None, default None
        Deprecated alias for ma_type.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    if length is None:
        length = len
    if length is None:
        raise TypeError("ma() missing required argument: 'length'")
    if maType is not None:
        ma_type = maType

    match ma_type.upper():
        case "SMA":
            return sma(source, length)
        case "EMA":
            return ema(source, length)
        case "WMA":
            return wma(source, length)
        case "RMA":
            return rma(source, length)
        case "DEMA":
            return dema(source, length)
        case "HMA":
            return hma(source, length)
        case "LSMA":
            return lsma(source, length)
        case "SMMA":
            return smma(source, length)
        case "TEMA":
            return tema(source, length)
        case "ZLEMA":
            return zlema(source, length)
        case _:
            raise ValueError(f"Unsupported moving average type: {ma_type}")

@register_outputs('upper_band', 'middle_band', 'lower_band')
def bollinger_bands(
    source: pd.Series,
    length: int = 20,
    mult: float = 2.0,
    ma_type: str = "SMA",
    *,
    maType: str | None = None,
) -> pd.DataFrame:
    """
    Bollinger Bands.
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    length : int, default 20
        Lookback period.
    mult : float, default 2.0
        Band or ATR multiplier.
    ma_type : str, default 'SMA'
        Moving-average type (e.g. SMA, EMA, WMA).
    maType : str | None, default None
        Deprecated alias for ma_type.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
    if maType is not None:
        ma_type = maType
    basis = ma(source, length, ma_type)
    deviation = mult * stdev(source, length)
    upper = basis + deviation
    lower = basis - deviation
    return pd.DataFrame({
        "upper_band": upper,
        "middle_band": basis,
        "lower_band": lower,
    })

@register_outputs('roc')
def rate_of_change(source: pd.Series, period: int) -> pd.Series:
    """
    Rate of Change (ROC).
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    period : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return source.diff(period) / source.shift(period) * 100

@register_outputs('momentum')
def momentum(source: pd.Series, length: int) -> pd.Series:
    """
    Momentum.
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    return source.diff(periods=length)

@register_outputs('mad')
def mad(source: pd.Series, length: int) -> pd.Series:
    """
    Median Absolute Deviation.
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    median_ = median(source, length)
    abs_dev = abs(source - median_)
    return median(abs_dev, length)

@register_outputs('aad')
def aad(source: pd.Series, length: int, avg_type: str) -> pd.Series:
    """
    Average Absolute Deviation.
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    avg_type : str
        Averaging method name.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    avg = ma(source, length, avg_type)
    abs_dev = abs(source - avg)
    return ma(abs_dev, length, avg_type)

@register_outputs('rmsd')
def rmsd(source: pd.Series, benchmark: pd.Series, length: int) -> pd.Series:
    """
    Root Mean Square Deviation.
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    benchmark : pd.Series
        Benchmark series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    diff_squared = (source - benchmark) ** 2
    return diff_squared.rolling(window=length).mean().apply(np.sqrt)

@register_outputs('emd')
def emd(source: pd.Series, benchmark: pd.Series, length: int) -> pd.Series:
    """
    Exponential Mean Deviation.
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    benchmark : pd.Series
        Benchmark series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    abs_dev = abs(source - benchmark)
    return ema(abs_dev, length)

@register_outputs('pvt')
def price_volume_trend(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    Price Volume Trend (PVT).
    
    Parameters
    ----------
    close : pd.Series
        Close price series.
    volume : pd.Series
        Volume series.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    pvt = [0.0]
    for i in range(1, len(close)):
        prev_close = close.iloc[i - 1]
        if prev_close == 0:
            pvt_change = 0
        else:
            pvt_change = ((close.iloc[i] - prev_close) / prev_close) * volume.iloc[i]
        pvt.append(pvt[-1] + pvt_change)
    return pd.Series(pvt, index=close.index)

@register_outputs('keltner')
def keltner(close: pd.Series, high: pd.Series, low: pd.Series, length: int, mult: float) -> pd.Series:
    """
    Keltner Channel Midline Offset.
    
    Parameters
    ----------
    close : pd.Series
        Close price series.
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    length : int
        Lookback period.
    mult : float
        Band or ATR multiplier.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    tr = true_range(high, low, close)
    return close.ewm(span=length, adjust=False).mean() + mult * tr.ewm(span=length, adjust=False).mean()

@register_outputs('zlag_dema')
def zlag_dema(src: pd.Series, length: int) -> pd.Series:
    """
    Zero-lag Double Exponential Moving Average (ZLag DEMA).
    
    Parameters
    ----------
    src : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    zdema1 = ema(src, length)
    zdema2 = ema(zdema1, length)
    dema1 = 2 * zdema1 - zdema2
    zdema12 = ema(dema1, length)
    zdema22 = ema(zdema12, length)
    return 2 * zdema12 - zdema22

@register_outputs('zlag_ma')
def zlag_ma(src: pd.Series, length: int) -> pd.Series:
    """
    Zero-lag Moving Average (ZLag MA).
    
    Parameters
    ----------
    src : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    alpha = 2.0 / (1.0 + length)
    per = int(np.ceil((length - 1.0) / 2.0))
    zlagma = pd.Series(index=src.index, dtype='float64')
    zlagma.iloc[0] = src.iloc[0]

    for i in range(1, len(src)):
        prev = zlagma.iloc[i - 1]
        prev_per = zlagma.iloc[i - per] if i - per >= 0 else src.iloc[i]
        zlagma.iloc[i] = prev + alpha * (2.0 * src.iloc[i] - prev_per - prev)

    return zlagma

@register_outputs('zlag_tema')
def zlag_tema(src: pd.Series, length: int) -> pd.Series:
    """
    Zero-lag Triple Exponential Moving Average (ZLag TEMA).
    
    Parameters
    ----------
    src : pd.Series
        Input price or value series.
    length : int
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    ema1 = ema(src, length)
    ema2 = ema(ema1, length)
    ema3 = ema(ema2, length)
    out = 3 * (ema1 - ema2) + ema3
    ema1a = ema(out, length)
    ema2a = ema(ema1a, length)
    ema3a = ema(ema2a, length)
    return 3 * (ema1a - ema2a) + ema3a

@register_outputs('gau_ma')
def gaussian_ma(src: pd.Series, length: int = 100) -> pd.Series:
    """
    Gaussian Moving Average.
    
    Parameters
    ----------
    src : pd.Series
        Input price or value series.
    length : int, default 100
        Lookback period.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    result = []
    for i in range(len(src)):
        if i < length:
            result.append(np.nan)
            continue
        window = src[i - length + 1:i + 1]
        mad = (window - window.mean()).abs().mean()
        std = window.std()
        sigma = (mad + std) / 2 if (mad + std) != 0 else 1
        weights = np.exp(-0.5 * ((np.arange(length) - (length - 1)) / sigma) ** 2)
        weights /= weights.sum()
        smoothed = np.dot(window, weights)
        result.append(smoothed)
    return pd.Series(result, index=src.index)

def valid_data_prefix(*series: pd.Series) -> tuple[tuple[pd.Series, ...], pd.Index]:
    """
    Truncate inputs to the last bar where all series are non-NaN.

    Charting tools often append future bars with NaN prices for projection.
    Compute on the returned truncated series, then ``.reindex(full_index)``
    so trailing future bars remain NaN in the output.
    """
    if not series:
        raise ValueError("At least one series is required")
    full_index = series[0].index
    for s in series[1:]:
        if not s.index.equals(full_index):
            raise ValueError("All series must share the same index")
    valid = pd.concat(list(series), axis=1).notna().all(axis=1)
    if not valid.any():
        raise ValueError("No non-NaN rows available for calculation")
    last_valid = valid[valid].index[-1]
    truncated = tuple(s.loc[:last_valid].copy() for s in series)
    return truncated, full_index


@register_outputs('rescaled')
def rescale(src: pd.Series, old_min: float, old_max: float, new_min: float, new_max: float) -> pd.Series:
    """
    Rescale Series.
    
    Rescale a source series from one bounded range to another.
    
    Parameters
    ----------
    src : pd.Series
        Input price or value series.
    old_min : float
        Original range minimum.
    old_max : float
        Original range maximum.
    new_min : float
        Target range minimum.
    new_max : float
        Target range maximum.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    denominator = np.maximum(old_max - old_min, 1e-10)
    rescaled = new_min + (new_max - new_min) * (src - old_min) / denominator
    
    return rescaled