import numpy as np
import pandas as pd
from trw_ta import register_outputs

@register_outputs('reflex_osc')
def reflex_oscillator(series: pd.Series, ss_period: float = 8.0, reflex_period: int = 20, post_smooth_period: float = 33.0) -> pd.Series:
    """https://www.tradingview.com/v/NI17VkdU/"""
    SQRT2xPI = np.sqrt(8.0) * np.arcsin(1.0)
    alpha = SQRT2xPI / ss_period
    beta = np.exp(-alpha)
    gamma = -beta ** 2
    delta = 2.0 * beta * np.cos(alpha)

    super_smooth = np.zeros(len(series))
    for i in range(2, len(series)):
        super_smooth[i] = (1.0 - delta - gamma) * (series.iloc[i] + series.iloc[i-1]) * 0.5 \
                          + delta * super_smooth[i-1] + gamma * super_smooth[i-2]

    slope = (np.roll(super_smooth, reflex_period) - super_smooth) / reflex_period
    epsilon = np.zeros(len(series))

    for i in range(reflex_period, len(series)):
        E = 0.0
        for j in range(1, reflex_period + 1):
            if i - j >= 0:
                E += (super_smooth[i] + j * slope[i]) - super_smooth[i - j]
        epsilon[i] = E / reflex_period

    zeta = 2.0 / (post_smooth_period + 1.0)
    ema = np.zeros(len(series))
    for i in range(1, len(series)):
        ema[i] = zeta * epsilon[i] ** 2 + (1.0 - zeta) * ema[i - 1]

    # Avoid division by zero
    with np.errstate(divide='ignore', invalid='ignore'):
        reflex = np.where(ema == 0, 0.0, epsilon / np.sqrt(ema))

    return pd.Series(reflex, index=series.index)
