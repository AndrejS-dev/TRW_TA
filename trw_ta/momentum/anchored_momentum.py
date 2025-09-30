import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('momentum', 'signal')
def anchored_momentum(src: pd.Series, momentum_period: int = 10, signal_period: int = 8, smooth_momentum: bool = False, smoothing_period: int = 7) -> pd.DataFrame:
    """https://www.tradingview.com/script/TBTFDWDq-Anchored-Momentum-LazyBear/"""
    p = 2 * momentum_period + 1
    src_processed = src.ewm(span=smoothing_period, adjust=False).mean() if smooth_momentum else src
    amom = 100 * ((src_processed / src.rolling(window=p).mean()) - 1)
    amoms = amom.rolling(window=signal_period).mean()
    
    return pd.DataFrame({
        'momentum': amom,
        'signal': amoms
    }, index=src.index)