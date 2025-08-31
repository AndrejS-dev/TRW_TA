import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('PSO')
def premier_stochastic_oscillator(high: pd.Series, low: pd.Series, close: pd.Series, stochlen=8, smoothlen=25):
    """https://www.tradingview.com/script/xewuyTA1-Indicator-Premier-Stochastic-Oscillator/"""
    lowest_low = low.rolling(stochlen).min()
    highest_high = high.rolling(stochlen).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)

    length = int(np.round(np.sqrt(smoothlen)))

    nsk = 0.1 * (stoch_k - 50)
    ss = nsk.ewm(span=length, adjust=False).mean()
    ss = ss.ewm(span=length, adjust=False).mean()

    expss = np.exp(ss)
    pso = (expss - 1) / (expss + 1)

    return pd.DataFrame({"PSO": pso})