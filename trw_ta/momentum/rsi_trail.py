import pandas as pd
import numpy as np
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('line', 'signal')
def rsi_trail(open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, ma_type: str = "EMA", rsi_lower: float = 40.0, rsi_upper: float = 60.0) -> pd.DataFrame:
    """https://www.tradingview.com/v/PUGvtsEu/"""
    ohlc4 = (open_ + high + low + close) / 4
    vol = ta.atr2(high, low, close, 27)

    if ma_type == "SMA":
        ma = ta.sma(ohlc4, 27)
    elif ma_type == "WMA":
        ma = ta.wma(ohlc4, 27)
    elif ma_type == "RMA":
        ma = ta.rma(ohlc4, 27)
    elif ma_type == "McGinley":
        ma = ta.mcginley(ohlc4, 27)
    else:
        ma = ta.ema(ohlc4, 27)

    upper_bound = ma + ((rsi_upper - 50) / 10) * vol
    lower_bound = ma - ((50 - rsi_lower) / 10) * vol

    signal = pd.Series(0, index=close.index)
    trend = pd.Series(0, index=close.index)
    line = pd.Series(np.nan, index=close.index)

    for i in range(1, len(close)):
        if ohlc4.iloc[i - 1] <= upper_bound.iloc[i - 1] and ohlc4.iloc[i] > upper_bound.iloc[i]:
            trend.iloc[i] = 1
        elif close.iloc[i - 1] >= lower_bound.iloc[i - 1] and close.iloc[i] < lower_bound.iloc[i]:
            trend.iloc[i] = -1
        else:
            trend.iloc[i] = trend.iloc[i - 1]

        signal.iloc[i] = 1 if trend.iloc[i] == 1 else -1 if trend.iloc[i] == -1 else 0
        line.iloc[i] = lower_bound.iloc[i] if trend.iloc[i] == 1 else upper_bound.iloc[i] if trend.iloc[i] == -1 else np.nan

    return pd.DataFrame({
        "line": line,
        "signal": signal
    })
