import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('stc', 'trend')
def schaff_trend_cycle(close: pd.Series, length: int = 12, fast_len: int = 26, slow_len: int = 50,
                       smoothing: float = 0.5) -> pd.DataFrame:
    """https://www.tradingview.com/v/WhRRThMI/"""

    def ema(series, length):
        return series.ewm(span=length, adjust=False).mean()

    macd = ema(close, fast_len) - ema(close, slow_len)

    min_macd = macd.rolling(length).min()
    max_macd = macd.rolling(length).max()
    k1 = np.where((max_macd - min_macd) > 0,
                  (macd - min_macd) / (max_macd - min_macd) * 100,
                  np.nan)

    d1 = np.zeros_like(k1)
    for i in range(1, len(k1)):
        if np.isnan(k1[i]):
            d1[i] = d1[i-1]
        else:
            d1[i] = d1[i-1] + smoothing * (k1[i] - d1[i-1])

    d1_series = pd.Series(d1, index=close.index)
    min_d1 = d1_series.rolling(length).min()
    max_d1 = d1_series.rolling(length).max()
    k2 = np.where((max_d1 - min_d1) > 0,
                  (d1_series - min_d1) / (max_d1 - min_d1) * 100,
                  np.nan)

    stc = np.zeros_like(k2)
    for i in range(1, len(k2)):
        if np.isnan(k2[i]):
            stc[i] = stc[i-1]
        else:
            stc[i] = stc[i-1] + smoothing * (k2[i] - stc[i-1])
    stc_series = pd.Series(stc, index=close.index)

    stc_shifted = stc_series.shift(1)
    stc_2 = stc_series.shift(2)
    stc_3 = stc_series.shift(3)

    signal = pd.Series(0, index=close.index)

    for i in range(3, len(close)):
        if (stc_3[i] <= stc_2[i]) and (stc_2[i] > stc_shifted[i]) and (stc_series[i] > 75):
            signal[i] = -1
        elif (stc_3[i] >= stc_2[i]) and (stc_2[i] < stc_shifted[i]) and (stc_series[i] < 25):
            signal[i] = 1
        else:
            signal[i] = signal[i-1]

    return pd.DataFrame({
        'stc': stc_series,
        'signal': signal
    })
