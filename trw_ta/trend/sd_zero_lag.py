import pandas as pd
import numpy as np
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('zlma', 'upper_base', 'lower_base', 'normalized_kijun', 'normalized_sd', 'signal')
def sd_zero_lag(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 12, sd_length: int = 27,
                        upper_sd_factor: float = 1.036, lower_sd_factor: float = 0.982,
                        ma_type: str = "Zero-Lag DEMA") -> pd.DataFrame:
    """https://www.tradingview.com/script/DUvsybcD-SD-Zero-Lag-The-Don-Killuminati/"""
    src = (high + low + close + close) / 4

    if ma_type == "Zero Lag TEMA":
        zlma = ta.zlag_tema(src, length)
    elif ma_type == "Zero-Lag DEMA":
        zlma = ta.zlag_dema(src, length)
    elif ma_type == "Zero-Lag MA":
        zlma = ta.zlag_ma(src, length)
    else:
        raise ValueError("Unsupported MA type.")

    zlma_sd = zlma.rolling(window=sd_length).std()

    upper_base = (zlma + zlma_sd) * upper_sd_factor
    lower_base = (zlma - zlma_sd) * lower_sd_factor

    base_kijun_long_signal = close > upper_base
    base_kijun_short_signal = close < lower_base

    normalized_kijun = -1 * zlma / close
    normalized_sd = normalized_kijun.rolling(window=sd_length).std()
    lower_bound = normalized_kijun - normalized_sd

    normalized_long_signal = lower_bound > -1
    normalized_short_signal = normalized_kijun < -1

    long_signal = base_kijun_long_signal & normalized_long_signal
    short_signal = base_kijun_short_signal & normalized_short_signal
    signal_raw = np.where(long_signal, 1, np.where(short_signal, -1, np.nan))
    signal = pd.Series(signal_raw).ffill().fillna(0)

    return pd.DataFrame({
        "zlma": zlma,
        "upper_base": upper_base,
        "lower_base": lower_base,
        "normalized_kijun": normalized_kijun,
        "normalized_sd": normalized_sd,
        "signal": signal
    })
