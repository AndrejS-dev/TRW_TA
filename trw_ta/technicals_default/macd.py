import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('macd', 'signal', 'hist')
def macd(source: pd.Series, fast_len: int = 12, slow_len: int = 26, sig_len: int = 9, oscillator_ma_type: str = "EMA", signal_ma_type: str = "EMA") -> pd.DataFrame:
    if slow_len <= fast_len:
        raise ValueError("slow_len must be greater than fast_len")
    
    fast_ma = ta.ma(source, fast_len, oscillator_ma_type)
    slow_ma = ta.ma(source, slow_len, oscillator_ma_type)
    macd_line = fast_ma - slow_ma
    signal_line = ta.ma(macd_line, sig_len, signal_ma_type)
    hist = macd_line - signal_line
    
    return pd.DataFrame({
        "macd": macd_line,
        "signal": signal_line,
        "hist": hist
    })
