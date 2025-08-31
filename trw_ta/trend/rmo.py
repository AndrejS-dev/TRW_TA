import pandas as pd
import trw_ta
from trw_ta import register_outputs

@register_outputs('RMO')
def rmo(close: pd.Series) -> pd.Series:
    """https://www.tradingview.com/script/efHJsedw-Indicator-Rahul-Mohindar-Oscillator-RMO/"""
    def sma2(x):
        return trw_ta.sma(x, 2)

    ma = close.copy()
    for _ in range(10):
        ma = sma2(ma)

    avg_ma = sum([sma2(close) if i == 0 else sma2(ma) for i in range(10)]) / 10
    swing_trd1 = 100 * (close - avg_ma) / (close.rolling(10).max() - close.rolling(10).min())
    rmo_val = trw_ta.ema(swing_trd1, 81)

    return rmo_val