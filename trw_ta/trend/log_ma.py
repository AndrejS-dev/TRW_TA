import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('LMA')
def logarithmic_moving_average(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    method: str = "Dynamic",
    sensitivity: float = 1.0,
    lookback: int = 50
) -> pd.DataFrame:

    atr = close.rolling(14).apply(lambda x: np.mean(np.abs(np.diff(x))))
    atr_mult = atr * sensitivity

    dynamic_support = low.rolling(lookback).min() + (atr_mult * 0.5)

    high_point = high.rolling(lookback).max()
    low_point = low.rolling(lookback).min()
    fib618 = high_point - ((high_point - low_point) * 0.618)
    fib786 = high_point - ((high_point - low_point) * 0.786)
    fib_level = pd.Series(np.where(close > fib618, fib618, fib786), index=close.index)

    bbasis = close.rolling(20).mean()
    dev = close.rolling(20).std() * 2
    vol_support = bbasis - dev

    poc = close.rolling(lookback).apply(
        lambda x: x.value_counts().idxmax() if not x.empty else np.nan
    )

    support_line = np.select(
        [method == "Dynamic", method == "Fibonacci", method == "Volatility", method == "Volume Profile"],
        [dynamic_support, fib_level, vol_support, poc],
        default=np.minimum.reduce([dynamic_support, fib_level, vol_support])
    )

    support_line = pd.Series(support_line, index=close.index).ewm(span=3, adjust=False).mean()

    return pd.DataFrame({
        'lma': support_line
    })