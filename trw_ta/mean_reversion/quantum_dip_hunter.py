import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('support_line', 'trend_score', 'dip_percent', 'quality_score')
def quantum_dip_hunter(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    volume: pd.Series,
    method: str = "Dynamic",
    sensitivity: float = 1.0,
    lookback: int = 50,
    dip_depth_pct: float = 2.0,
    trend_period: int = 200,
    trend_strength_threshold: float = 0.5,
    use_trend_filter: bool = True,
    min_dip_score: float = 70.0
) -> pd.DataFrame:
    
    atr = close.rolling(14).apply(lambda x: np.mean(np.abs(np.diff(x))))
    atr_mult = atr * sensitivity

    vol_ma = volume.rolling(20).mean()
    vol_spike = volume > vol_ma * 1.5

    dynamic_support = low.rolling(lookback).min() + (atr_mult * 0.5)

    high_point = high.rolling(lookback).max()
    low_point = low.rolling(lookback).min()
    fib618 = high_point - ((high_point - low_point) * 0.618)
    fib786 = high_point - ((high_point - low_point) * 0.786)
    fib_level = pd.Series(np.where(close > fib618, fib618, fib786), index=close.index)

    bbasis = close.rolling(20).mean()
    dev = close.rolling(20).std() * 2
    vol_support = bbasis - dev

    price_step = atr * 0.1
    poc = close.rolling(lookback).apply(
        lambda x: x.value_counts().idxmax() if not x.empty else np.nan
    )

    support_line = np.select(
        [method == "Dynamic", method == "Fibonacci", method == "Volatility", method == "Volume Profile"],
        [dynamic_support, fib_level, vol_support, poc],
        default=np.minimum.reduce([dynamic_support, fib_level, vol_support])
    )

    support_line = pd.Series(support_line, index=close.index).ewm(span=3, adjust=False).mean()

    ema200 = close.ewm(span=trend_period, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    def rsi_calc(x):
        delta = np.diff(x)
        up = np.mean(delta[delta > 0]) if np.any(delta > 0) else 0
        down = np.mean(-delta[delta < 0]) if np.any(delta < 0) else 0
        rs = up / (down + 1e-9)
        return 100 - 100 / (1 + rs)

    rsi = close.rolling(15).apply(rsi_calc, raw=True)

    trend_score = (
        (close > ema200).astype(float) * 0.3 +
        (close > ema50).astype(float) * 0.2 +
        (ema50 > ema200).astype(float) * 0.3 +
        (rsi > 50).astype(float) * 0.2
    )

    dip_condition = (low < support_line) | (close < support_line)
    dip_percent = (support_line - low) / support_line * 100
    valid_dip = dip_condition & (dip_percent >= dip_depth_pct)

    momentum = close.diff(10)
    momentum_score = (momentum > momentum.shift(1)).astype(float) * 15 + (~(momentum > momentum.shift(1))).astype(float) * 5

    distance_score = np.minimum(dip_percent / 5 * 15, 15)

    quality_score = (
        np.where(use_trend_filter, (trend_score > trend_strength_threshold).astype(float) * 30, 30) +
        np.where(vol_spike, 20, 10) +
        np.where(rsi < 30, 20, np.where(rsi < 40, 10, 0)) +
        momentum_score +
        distance_score
    )

    buy_signal = valid_dip & (quality_score >= min_dip_score) & (
        (~use_trend_filter) | (trend_score >= trend_strength_threshold)
    )

    return pd.DataFrame({
        'support_line': support_line,
        'trend_score': trend_score,
        'dip_percent': dip_percent,
        'quality_score': quality_score
    })