import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('forward_backward', 'backward', 'histogram')
def forward_backward_exponential_oscillator(src: pd.Series, length: int = 20, smooth: int = 10) -> pd.DataFrame:
    """https://www.tradingview.com/script/hMctOIlX-Forward-Backward-Exponential-Oscillator-LuxAlgo/"""
    def backward_ema():
        ema1 = src.ewm(span=smooth, adjust=False).mean()
        ema2 = ema1.ewm(span=smooth, adjust=False).mean()
        delta = ema2.diff()
        num = delta.rolling(window=length).sum()
        den = delta.abs().rolling(window=length).sum()
        return num / den.where(den != 0, 1) * 50 + 50

    def forward_backward():
        ema1 = src.ewm(span=smooth, adjust=False).mean()
        result = pd.Series(np.zeros(len(src)), index=src.index)
        for i in range(len(src)):
            if i < length - 1:
                continue
            num, den = 0, 0
            ema2 = ema1.iloc[i]
            prev = ema2
            for j in range(1, length):
                idx = i - j
                if idx < 0:
                    break
                ema2 += 2 / (smooth + 1) * (ema1.iloc[idx] - ema2)
                dt = prev - ema2
                num += dt
                den += abs(dt)
                prev = ema2
            result.iloc[i] = num / den * 50 + 50 if den != 0 else 50
        return result

    fb = forward_backward()
    bwrd = backward_ema()
    hist = (fb - bwrd) / 4 + 50

    return pd.DataFrame({
        'forward_backward': fb,
        'backward': bwrd,
        'histogram': hist
    }, index=src.index)