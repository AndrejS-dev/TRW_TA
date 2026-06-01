import pandas as pd
import numpy as np
from trw_ta import register_outputs


def bandpass_filter(src: pd.Series, period: float, bandwidth: float) -> pd.Series:
    alpha = np.cos(2 * np.pi / period)
    beta = 1.0 / np.cos(4 * np.pi * bandwidth / period)
    gamma = beta - np.sqrt(beta**2 - 1.0)
    alpha2 = 1.0 - gamma

    filt = pd.Series(np.nan, index=src.index, dtype=float)

    for i in range(2, len(src)):
        term1 = 0.5 * alpha2 * (src.iloc[i] - src.iloc[i - 2])
        term2 = gamma * (1 + alpha) * filt.iloc[i - 1] if not pd.isna(filt.iloc[i - 1]) else 0.0
        term3 = -gamma * filt.iloc[i - 2] if not pd.isna(filt.iloc[i - 2]) else 0.0
        filt.iloc[i] = term1 + term2 + term3

    return filt


def normalize(src: pd.Series, length: int) -> pd.Series:
    r_max = src.rolling(length, min_periods=1).max()
    r_min = src.rolling(length, min_periods=1).min()
    denom = np.maximum(r_max - r_min, 1e-10)
    return 100.0 * (src - r_min) / denom


@register_outputs('harmonic_resonance')
def harmonic_resonance_oscillator(close: pd.Series, ref_period: int = 30, short_mult: float = 0.5, med_mult: float = 1.0,
                                  long_mult: float = 2.0, bandwidth: float = 0.1, norm_length: int = 50) -> pd.DataFrame:
    """https://www.tradingview.com/script/fJ3khqWG-Harmonic-Resonance-Oscillator-LuxAlgo/"""
    p_short = max(ref_period * short_mult, 2.0)
    p_med   = max(ref_period * med_mult,   2.0)
    p_long  = max(ref_period * long_mult,  2.0)

    cycle_short = bandpass_filter(close, p_short, bandwidth)
    cycle_med   = bandpass_filter(close, p_med,   bandwidth)
    cycle_long  = bandpass_filter(close, p_long,  bandwidth)

    norm_short = normalize(cycle_short, norm_length)
    norm_med   = normalize(cycle_med,   norm_length)
    norm_long  = normalize(cycle_long,  norm_length)

    resonance = (norm_short + norm_med + norm_long) / 3.0

    return pd.DataFrame({
        'harmonic_resonance': resonance
    }).rename_axis(None, axis=1)