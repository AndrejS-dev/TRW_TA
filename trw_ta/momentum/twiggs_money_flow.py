# trw_ta/momentum/twiggs_money_flow.py
import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('cl', 'sig')
def twiggs_money_flow(high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, length: int = 21) -> pd.DataFrame:
    """https://www.tradingview.com/script/Z9jDy5JN-Twiggs-Go-Money-Flow-Enhanced-KingThies/"""
    tr_high = pd.concat([close.shift(1), high], axis=1).max(axis=1)
    tr_low = pd.concat([close.shift(1), low], axis=1).min(axis=1)
    tr_range = tr_high - tr_low
    adv = np.where(tr_range == 0, 0.0, volume * ((close - tr_low) - (tr_high - close)) / tr_range)
    wma_volume = volume.ewm(alpha=1/length, adjust=False).mean()
    wma_adv = pd.Series(adv, index=volume.index).ewm(alpha=1/length, adjust=False).mean()
    tmf = np.where(wma_volume == 0, 0.0, wma_adv / wma_volume)
    sig = np.sign(tmf).astype(int)
    return pd.DataFrame({"cl": tmf, "sig": sig}, index=high.index)