import pandas as pd

def tsi(series: pd.Series, short_len: int, long_len: int) -> pd.Series:
    momentum = series.diff()
    abs_momentum = momentum.abs()

    ema1 = momentum.ewm(span=short_len, adjust=False).mean()
    ema2 = ema1.ewm(span=long_len, adjust=False).mean()

    abs_ema1 = abs_momentum.ewm(span=short_len, adjust=False).mean()
    abs_ema2 = abs_ema1.ewm(span=long_len, adjust=False).mean()

    tsi = 100 * ema2 / abs_ema2
    return tsi

def smi_ergodic(close: pd.Series, short_len: int = 5, long_len: int = 20, sig_len: int = 5) -> pd.DataFrame:
    ergodic = tsi(close, short_len, long_len)
    signal = ergodic.ewm(span=sig_len, adjust=False).mean()

    return pd.DataFrame({
        "ergodic": ergodic,
        "signal": signal
    })
