import pandas as pd
import trw_ta
from trw_ta import register_outputs

@register_outputs('bulls', 'bears')
def bears_bulls(src: pd.Series, high: pd.Series, low: pd.Series, length: int = 13) -> pd.DataFrame:
    """https://www.tradingview.com/script/0t5z9xe2-Indicator-Bears-Bulls-power/"""
    s_ma = trw_ta.sma(src, length)
    bulls = high - s_ma
    bears = low - s_ma

    return pd.DataFrame({
        'bulls': bulls,
        'bears': bears
    })
