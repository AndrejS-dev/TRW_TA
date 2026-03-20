import pandas as pd
import numpy as np
from trw_ta import register_outputs


def dmi_components(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 14) -> tuple[pd.Series, pd.Series, pd.Series]:
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    dm_plus = (high - high.shift(1)).where((high - high.shift(1)) > (low.shift(1) - low), 0)
    dm_minus = (low.shift(1) - low).where((low.shift(1) - low) > (high - high.shift(1)), 0)

    atr = tr.ewm(span=length, adjust=False, min_periods=length).mean()
    di_plus = 100 * (dm_plus.ewm(span=length, adjust=False, min_periods=length).mean() / atr)
    di_minus = 100 * (dm_minus.ewm(span=length, adjust=False, min_periods=length).mean() / atr)

    dx = 100 * np.abs(di_plus - di_minus) / (di_plus + di_minus + 1e-10)
    adx = dx.ewm(span=length, adjust=False, min_periods=length).mean()

    return di_plus, di_minus, adx


def logistic_prob(series: pd.Series, mean_lb: int, slope: float, smooth_len: int) -> pd.Series:
    mean = series.rolling(mean_lb, min_periods=1).mean()
    z = (series - mean) * slope
    prob_raw = 1.0 / (1.0 + np.exp(-z))
    return prob_raw.ewm(span=smooth_len, adjust=False, min_periods=1).mean()


@register_outputs('dlo')
def directional_logistic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, di_length: int = 14,
    mean_lookback: int = 360, lr_slope: float = 0.18, prob_smooth_len: int = 3, osc_scale: float = 2.5,
    osc_smooth_len: int = 7) -> pd.DataFrame:
    """https://www.tradingview.com/script/fDfndNzN-Directional-Logistic-Oscillator-GainzAlgo/"""

    plus_di, minus_di, adx = dmi_components(high, low, close, length=di_length)

    prob_plus  = logistic_prob(plus_di,  mean_lookback, lr_slope, prob_smooth_len)
    prob_minus = logistic_prob(minus_di, mean_lookback, lr_slope, prob_smooth_len)
    prob_adx   = logistic_prob(adx,      mean_lookback, lr_slope, prob_smooth_len)

    net_dir = prob_plus - prob_minus
    strength_raw = net_dir * prob_adx * osc_scale

    strength_bound = np.tanh(strength_raw)
    dlo = strength_bound.ewm(span=osc_smooth_len, adjust=False, min_periods=1).mean()

    return pd.DataFrame({
        'dlo': dlo
    }).rename_axis(None, axis=1)