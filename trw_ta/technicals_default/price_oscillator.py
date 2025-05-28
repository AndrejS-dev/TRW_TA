import pandas as pd

def price_oscillator(src: pd.Series, shortlen: int = 12, longlen: int = 26, signallen: int = 9, use_exp: bool = True) -> pd.DataFrame:
    def esma(series: pd.Series, length: int) -> pd.Series:
        return series.ewm(span=length, adjust=False).mean() if use_exp else series.rolling(length).mean()

    short_ma = esma(src, shortlen)
    long_ma = esma(src, longlen)
    po = (short_ma - long_ma) / long_ma * 100
    signal = esma(po, signallen)
    hist = po - signal

    return pd.DataFrame({
        "ppo": po,
        "signal": signal,
        "hist": hist
    })
