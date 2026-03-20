import pandas as pd
import numpy as np
from trw_ta import register_outputs

def get_ols_weights(length: int, degree: int) -> np.ndarray:

    if length < 2 or degree < 1:
        return np.ones(length) / length

    x = np.arange(length).reshape(-1, 1)
    X = np.hstack([x ** j for j in range(degree + 1)])

    Xt = X.T
    XtX = Xt @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        # Fallback: pseudo-inverse if singular
        XtX_inv = np.linalg.pinv(XtX)

    H = XtX_inv @ Xt

    x_last = np.array([(length - 1) ** j for j in range(degree + 1)]).reshape(1, -1)

    weights = (x_last @ H).flatten()

    return weights


def apply_smoothing_once(src: pd.Series, length: int, method: str = "EMA") -> pd.Series:
    if method == "None" or length <= 1:
        return src

    if method == "SMA":
        return src.rolling(length, min_periods=1).mean()
    elif method == "EMA":
        return src.ewm(span=length, adjust=False, min_periods=1).mean()
    elif method == "WMA":
        weights = np.arange(1, length + 1)
        def wma_func(x):
            valid = x[-len(weights):]
            w = weights[-len(valid):]
            return np.dot(valid, w) / w.sum() if w.sum() > 0 else np.nan
        return src.rolling(length, min_periods=1).apply(wma_func, raw=True)
    elif method == "RMA":
        return src.ewm(alpha=1/length, adjust=False, min_periods=1).mean()
    elif method == "HMA":
        half = length // 2
        sqrt_len = int(np.sqrt(length))
        wma_half = src.rolling(half, min_periods=1).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum() if len(x) > 0 else np.nan, raw=True
        )
        wma_full = src.rolling(length, min_periods=1).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum() if len(x) > 0 else np.nan, raw=True
        )
        raw_hma = 2 * wma_half - wma_full
        return raw_hma.rolling(sqrt_len, min_periods=1).apply(
            lambda x: np.dot(x, np.arange(1, len(x)+1)) / np.arange(1, len(x)+1).sum() if len(x) > 0 else np.nan, raw=True
        )
    elif method == "DEMA":
        e1 = src.ewm(span=length, adjust=False, min_periods=1).mean()
        e2 = e1.ewm(span=length, adjust=False, min_periods=1).mean()
        return 2 * e1 - e2
    elif method == "TEMA":
        e1 = src.ewm(span=length, adjust=False, min_periods=1).mean()
        e2 = e1.ewm(span=length, adjust=False, min_periods=1).mean()
        e3 = e2.ewm(span=length, adjust=False, min_periods=1).mean()
        return 3 * (e1 - e2) + e3
    elif method == "VWMA":
        return src.rolling(length, min_periods=1).mean()
    elif method == "Gaussian":
        sigma = length / 3.0
        def gauss(x):
            weights = np.exp(-0.5 * ((np.arange(len(x)) / sigma) ** 2))
            return np.dot(x, weights) / weights.sum() if weights.sum() > 0 else np.nan
        return src.rolling(length, min_periods=1).apply(gauss, raw=True)
    else:
        return src.rolling(length, min_periods=1).mean()


@register_outputs('prma')
def polynomial_regression_ma(source: pd.Series, period: int = 100, degree: float = 4.0,
    smooth_type: str = "EMA", smooth_len: int = 5, smooth_iterations: int = 1) -> pd.DataFrame:
    """https://www.tradingview.com/script/3aez8sIq-Polynomial-Regression-Moving-Average-PRMA/"""
    n = len(source)
    if n < period:
        return pd.DataFrame({'prma': np.nan}, index=source.index)

    deg_f = int(np.floor(degree))
    deg_c = int(np.ceil(degree))
    weight = degree - deg_f

    kernel_floor = get_ols_weights(period, deg_f)
    if deg_f == deg_c:
        kernel = kernel_floor
    else:
        kernel_ceil = get_ols_weights(period, deg_c)
        kernel = (1 - weight) * kernel_floor + weight * kernel_ceil

    prma_raw = pd.Series(np.nan, index=source.index, dtype=float)

    for i in range(period - 1, n):
        window = source.iloc[i - period + 1 : i + 1].values
        if len(window) == period and not np.any(np.isnan(window)):
            prma_raw.iloc[i] = np.dot(kernel, window[::-1])  # reverse because weights are for oldest to newest

    result = prma_raw.copy()
    if smooth_type != "None" and smooth_len > 1:
        for _ in range(smooth_iterations):
            result = apply_smoothing_once(result, smooth_len, smooth_type)

    return pd.DataFrame({
        'prma': result
    }).rename_axis(None, axis=1)