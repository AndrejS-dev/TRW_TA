import numpy as np
import pandas as pd
from trw_ta import register_outputs

@register_outputs('raw_vams', 'zscore_vams', 'smooth_vams')
def vams_oscillator(close: pd.Series, momentum_period: int = 10, volatility_period: int = 20, zscore_period: int = 100, smoothing_period: int = 3, min_volatility: float = 0.001) -> pd.DataFrame:
    """https://www.tradingview.com/v/P71BL3mF/"""
    close = pd.Series(close)
    log_returns = np.log(close / close.shift(1))
    momentum = log_returns.rolling(momentum_period).mean() * 252

    raw_volatility = log_returns.rolling(volatility_period).std() * np.sqrt(252)
    volatility = raw_volatility.clip(lower=min_volatility)
    raw_vams = momentum / volatility

    mean_vams = raw_vams.rolling(zscore_period).mean()
    std_vams = raw_vams.rolling(zscore_period).std()
    zscore_vams = (raw_vams - mean_vams) / std_vams

    zscore_vams = zscore_vams.replace([np.inf, -np.inf], np.nan).fillna(0)
    smooth_vams = zscore_vams.ewm(span=smoothing_period, adjust=False).mean()

    return pd.DataFrame({
        'raw_vams': raw_vams.fillna(0),
        'zscore_vams': zscore_vams,
        'smooth_vams': smooth_vams.fillna(0)
    })

