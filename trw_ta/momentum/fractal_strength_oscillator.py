# trw_ta/momentum/fractal_strength_oscillator.py
import pandas as pd
import numpy as np
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('cl', 'sig')
def fractal_strength_oscillator(
    close: pd.Series,
    volume: pd.Series | None = None,
    N: int = 20,
    FDI_threshold: float = 1.45,
    rsi_length: int = 14,
    use_smoothing: bool = False,
    ma_type: str = "EMA",
    ma_length: int = 14,
) -> pd.DataFrame:
    """https://www.tradingview.com/script/4pd82lq7-Fractal-Strength-Oscillator/"""

    if not isinstance(close, pd.Series):
        raise TypeError("close must be a pandas Series")
    if close.isna().any():
        raise ValueError("close contains NaN")
    if N < 2 or len(close) < N:
        raise ValueError(f"close must contain at least {N} rows")
    if rsi_length < 1 or ma_length < 1:
        raise ValueError("Lengths must be >= 1")
    if volume is not None and len(volume) != len(close):
        raise ValueError("volume must have the same length as close")

    hh = close.rolling(window=N, min_periods=1).max()
    ll = close.rolling(window=N, min_periods=1).min()
    rng = hh - ll
    rng = rng.replace(0, np.nan)  # avoid div-by-zero later

    # normalised positions inside the N-bar window
    norm = (close - ll) / rng

    # pre-compute the N differences (same order as Pine array)
    diffs = pd.concat(
        [norm.shift(i) for i in range(N)], axis=1
    ).iloc[:, ::-1]  # newest on the right
    diffs = diffs.fillna(0)

    # length = Σ sqrt( (diff_i - diff_{i+1})² + (1/N)² )
    delta = diffs.iloc[:, 1:].values - diffs.iloc[:, :-1].values
    fdi_len = np.sqrt(delta ** 2 + (1.0 / N) ** 2).sum(axis=1)
    fdi_len = pd.Series(fdi_len, index=close.index)

    # final FDI
    fdi = 1 + (np.log(fdi_len.replace(0, np.nan)) + np.log(2)) / np.log(2 * N)
    fdi = fdi.fillna(method="ffill").fillna(1)  # first values → neutral

    delta = close.diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    roll_up = up.rolling(window=rsi_length, min_periods=1).mean()
    roll_down = down.rolling(window=rsi_length, min_periods=1).mean()
    rs = roll_up / roll_down.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    rsi = rsi.fillna(50)

    if use_smoothing:
        rsi = ta.ma(rsi, ma_length, ma_type)

    RSItrend = pd.Series(0, index=close.index)
    RSItrend = np.where(rsi < 45, -1, RSItrend)
    RSItrend = np.where(rsi > 55, 1, RSItrend)

    FDItrend = pd.Series(0, index=close.index)
    FDItrend = np.where(fdi > FDI_threshold, -1, FDItrend)
    FDItrend = np.where(fdi < FDI_threshold, 1, FDItrend)

    Trend = pd.Series(0, index=close.index)
    Trend = np.where((RSItrend == 1) & (FDItrend == 1), 1, Trend)
    Trend = np.where((RSItrend == 1) & (FDItrend == -1), 1, Trend)
    Trend = np.where((RSItrend == -1) & (FDItrend == -1), -1, Trend)
    Trend = np.where((RSItrend == -1) & (FDItrend == 1), -1, Trend)

    return pd.DataFrame(
        {
            "cl": fdi,      # oscillator line
            "sig": Trend,   # 1 / -1 / 0
        },
        index=close.index,
    )