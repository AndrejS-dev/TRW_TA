import pandas as pd

def stochastic_rsi(src: pd.Series, rsi_length: int = 14, stoch_length: int = 14, smooth_k: int = 3, smooth_d: int = 3) -> pd.DataFrame:
    delta = src.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/rsi_length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/rsi_length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    min_rsi = rsi.rolling(window=stoch_length).min()
    max_rsi = rsi.rolling(window=stoch_length).max()
    stoch_rsi_raw = (rsi - min_rsi) / (max_rsi - min_rsi)

    k = stoch_rsi_raw.rolling(window=smooth_k).mean() * 100
    d = k.rolling(window=smooth_d).mean()

    return pd.DataFrame({
        'k': k,
        'd': d
    })
