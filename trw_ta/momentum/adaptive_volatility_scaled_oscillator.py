import pandas as pd
import numpy as np
from trw_ta import register_outputs


def yang_zhang_volatility(open_: pd.Series, high: pd.Series, low: pd.Series,
    close: pd.Series, period: int = 14) -> pd.Series:

    r_oo = np.log(open_ / close.shift(1)).fillna(0)
    r_oc = np.log(close / open_).fillna(0)
    r_hl = np.log(high / low).fillna(0)

    sigma_oo_sq = r_oo.rolling(period, min_periods=1).var()
    sigma_oc_sq = r_oc.rolling(period, min_periods=1).var()
    sigma_hl_sq = r_hl.rolling(period, min_periods=1).var()

    k = 0.34
    c = 0.34
    sigma_yz_sq = k * sigma_oo_sq + (1 - k) * sigma_oc_sq + c * sigma_hl_sq
    sigma_yz_sq = sigma_yz_sq.clip(lower=0)

    return np.sqrt(sigma_yz_sq)


def adaptive_ema(src: pd.Series, base_length: int = 5) -> pd.Series:
    alpha = 2.0 / (base_length + 1.0)
    return src.ewm(alpha=alpha, adjust=False, min_periods=1).mean()


@register_outputs('zscore', 'adaptive')
def adaptive_volatility_scaled_oscillator(src: pd.Series, open_: pd.Series, high: pd.Series, low: pd.Series, volume: pd.Series,
    metric_measure: str = 'Standard Deviation', lookback: int = 20, dev_period: int = 12, yang_period: int = 14, smoothing_base: int = 5,) -> pd.DataFrame:
    """https://www.tradingview.com/script/SdMSDRLI-Adaptive-Volatility-Scaled-Oscillator-AVSO-Zeiierman/"""

    if metric_measure == 'Volume':
        metric = volume
    elif metric_measure == 'Close':
        metric = src
    elif metric_measure == 'Standard Deviation':
        metric = src.rolling(dev_period, min_periods=1).std()
    elif metric_measure == 'ATR':
        tr = pd.concat([
            high - low,
            (high - src.shift(1)).abs(),
            (low - src.shift(1)).abs()
        ], axis=1).max(axis=1)
        metric = tr.ewm(span=dev_period, adjust=False, min_periods=1).mean()
    elif metric_measure == 'Yang':
        metric = yang_zhang_volatility(open_, high, low, src, yang_period)
    else:
        raise ValueError(f"Unsupported metric_measure: {metric_measure}")

    scaled_metric = metric * src / 10.0

    mean_scaled = scaled_metric.rolling(lookback, min_periods=1).mean()
    std_scaled  = scaled_metric.rolling(lookback, min_periods=1).std(ddof=0)

    zscore = (scaled_metric - mean_scaled) / (std_scaled + 1e-10)

    abs_z = zscore.abs().fillna(0)
    dynamic_len_float = smoothing_base + (abs_z * 2)
    dynamic_len = (
        dynamic_len_float
        .round()
        .fillna(smoothing_base)
        .astype(int)
        .clip(lower=1)
    )

    adaptive = pd.Series(np.nan, index=zscore.index, dtype=float)

    for i in range(1, len(zscore)):
        if pd.isna(zscore.iloc[i]):
            adaptive.iloc[i] = adaptive.iloc[i-1] if i > 0 and not pd.isna(adaptive.iloc[i-1]) else 0.0
            continue

        length_i = dynamic_len.iloc[i]
        alpha_i = 2.0 / (length_i + 1.0)

        prev = adaptive.iloc[i-1] if not pd.isna(adaptive.iloc[i-1]) else zscore.iloc[i]
        adaptive.iloc[i] = alpha_i * zscore.iloc[i] + (1 - alpha_i) * prev

    return pd.DataFrame({
        'zscore':  zscore,
        'adaptive': adaptive,
    }).rename_axis(None, axis=1)