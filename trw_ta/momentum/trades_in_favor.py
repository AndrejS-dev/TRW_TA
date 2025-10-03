import numpy as np
import pandas as pd
from trw_ta import register_outputs

@register_outputs('tif')
def trades_in_favor(close: pd.Series, volume: pd.Series, length: int = 20, smoothing: int = 5) -> pd.Series:
    """https://www.tradingview.com/script/wvDsCkvr-Trades-in-Favor/"""
    price_change = close - close.shift(1)
    volume_weighted = volume * np.abs(price_change)

    bullish_momentum = np.where(price_change > 0, volume_weighted, 0)

    total_bullish = pd.Series(bullish_momentum, index=close.index).rolling(length).sum()
    total_volume = pd.Series(volume_weighted, index=close.index).rolling(length).sum()

    trades_in_favor_raw = np.where(total_volume > 0, (total_bullish / total_volume) * 100, 50)

    trades_in_favor = pd.Series(trades_in_favor_raw, index=close.index).rolling(smoothing).mean()

    return trades_in_favor