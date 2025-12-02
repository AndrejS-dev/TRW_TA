# trw_ta/momentum/mro_alpha.py
import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('mro', 'mro_hist', 'mro_sig')
def mro_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    bb_length: int = 20, bb_mult: float = 2.0,
    rsi_length: int = 14, stoch_k: int = 14, stoch_d: int = 3,
    mfi_length: int = 14, composite_smooth: int = 3,
    extreme_level: float = 80.0
) -> pd.DataFrame:
    """https://www.tradingview.com/script/fEFvlTk2-Mean-Reversion-Oscillator-Alpha-Extract/"""
    # Bollinger %B
    basis = close.rolling(bb_length).mean()
    dev = bb_mult * close.rolling(bb_length).std()
    bb_upper = basis + dev
    bb_lower = basis - dev
    bb_percent = (close - bb_lower) / (bb_upper - bb_lower) * 100

    # RSI
    delta = close.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1/rsi_length, adjust=False).mean()
    ma_down = down.ewm(alpha=1/rsi_length, adjust=False).mean()
    rs = ma_up / ma_down
    rsi = 100 - 100 / (1 + rs)

    # Stochastic
    lowest_low = low.rolling(stoch_k).min()
    highest_high = high.rolling(stoch_k).max()
    stoch_k_val = 100 * (close - lowest_low) / (highest_high - lowest_low)
    stoch_d_val = stoch_k_val.rolling(stoch_d).mean()

    # MFI
    typical = (high + low + close) / 3
    money_flow = typical * volume
    pos_flow = money_flow.where(typical > typical.shift(1), 0).rolling(mfi_length).sum()
    neg_flow = money_flow.where(typical < typical.shift(1), 0).rolling(mfi_length).sum()
    mfi = 100 - 100 / (1 + pos_flow / (neg_flow + 1e-9))

    # Williams %R
    wpr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-9)

    # Composite
    composite_raw = (bb_percent + rsi + stoch_d_val + mfi + (100 + wpr)) / 5
    mro = composite_raw.rolling(composite_smooth).mean()

    # Histogram (momentum)
    hist = mro.diff(5)

    # Signals
    bullish_entry = (mro > (100 - extreme_level)) & (mro.shift(1) < (100 - extreme_level))
    bearish_entry = (mro < extreme_level) & (mro.shift(1) > extreme_level)
    momentum_bull = hist > hist.shift(1)
    momentum_bear = hist < hist.shift(1)
    bull_sig = bullish_entry & momentum_bull
    bear_sig = bearish_entry & momentum_bear

    sig = np.where(bull_sig, 1, np.where(bear_sig, -1, 0))

    return pd.DataFrame({
        "mro": mro,
        "mro_hist": hist,
        "mro_sig": sig
    }, index=close.index)