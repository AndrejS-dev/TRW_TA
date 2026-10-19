import pandas as pd
from trw_ta import register_outputs

@register_outputs('po', 'signal', 'hist')
def price_oscillator(src: pd.Series, shortlen: int = 12, longlen: int = 26, signallen: int = 9, use_exp: bool = True) -> pd.DataFrame:
    """
    Price Oscillator.
    
    Parameters
    ----------
    src : pd.Series
        Input price or value series.
    shortlen : int, default 12
        Shortlen.
    longlen : int, default 26
        Longlen.
    signallen : int, default 9
        Signallen.
    use_exp : bool, default True
        Use exp.
    
    Returns
    -------
    pd.DataFrame
        DataFrame of indicator columns.
    """
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
