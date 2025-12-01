# trw_ta/momentum/osc_matrix.py
import pandas as pd
import numpy as np
from trw_ta import register_outputs


@register_outputs('mf', 'hw')
def oscillator_matrix(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    mf_length: int = 14, mf_smooth: int = 3, mf_multiplier: float = 2.0,
    hw_fast: int = 12, hw_slow: int = 26, hw_signal: int = 9, hw_smooth: int = 3
) -> pd.DataFrame:
    """https://www.tradingview.com/script/gfUptPZd-Oscillator-Matrix-Alpha-Extract/"""

    # --- Money Flow Oscillator ---
    typical_price = (high + low + close) / 3
    price_up = typical_price > typical_price.shift(1)
    price_down = typical_price < typical_price.shift(1)

    up_volume = volume.where(price_up, 0)
    down_volume = volume.where(price_down, 0)

    up_vol_sum = up_volume.rolling(mf_length, min_periods=1).mean()
    down_vol_sum = down_volume.rolling(mf_length, min_periods=1).mean()

    total_volume = up_vol_sum + down_vol_sum
    money_flow_ratio = np.where(total_volume > 0, (up_vol_sum - down_vol_sum) / total_volume, 0)
    money_flow = pd.Series(money_flow_ratio, index=close.index)
    money_flow = money_flow.ewm(span=mf_smooth, adjust=False, min_periods=1).mean()
    money_flow = money_flow * mf_multiplier
    money_flow = money_flow.clip(-1, 1)

    # --- Hyper Wave Oscillator ---
    fast_ma = close.ewm(span=hw_fast, adjust=False, min_periods=1).mean()
    slow_ma = close.ewm(span=hw_slow, adjust=False, min_periods=1).mean()
    macd_line = fast_ma - slow_ma
    signal_line = macd_line.ewm(span=hw_signal, adjust=False, min_periods=1).mean()

    price_base = close.rolling(50, min_periods=1).mean()
    macd_normalized = macd_line / price_base.replace(0, np.nan)

    macd_range = macd_normalized.abs().rolling(100, min_periods=1).max()
    hyper_wave = np.where(macd_range > 0, macd_normalized / macd_range, 0)
    hyper_wave = pd.Series(hyper_wave, index=close.index)
    hyper_wave = hyper_wave.ewm(span=hw_smooth, adjust=False, min_periods=1).mean()
    hyper_wave = hyper_wave.clip(-1, 1)

    # --- Output ---
    return pd.DataFrame({
        "mf": money_flow,
        "hw": hyper_wave
    }, index=close.index)