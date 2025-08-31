import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('stress', 'd1_stoch', 'd2_stoch')
def kaufman_stress_indicator(high: pd.Series, low: pd.Series, close: pd.Series, 
                           d2_high: pd.Series, d2_low: pd.Series, d2_close: pd.Series, 
                           length: int = 60) -> pd.DataFrame:
    """https://www.tradingview.com/script/uKzpaELr-Kaufman-Stress-Indicator/"""
    def calc_range(hi, lo, len):
        return hi.rolling(window=len).max() - lo.rolling(window=len).min()
    
    r1 = calc_range(high, low, length)
    r2 = calc_range(d2_high, d2_low, length)
    
    s1 = (close - low.rolling(window=length).min()) / r1.where(r1 != 0, 1) * 100
    s2 = (d2_close - d2_low.rolling(window=length).min()) / r2.where(r2 != 0, 1) * 100
    d = s1 - s2
    r11 = calc_range(d, d, length)
    sv = (d - d.rolling(window=length).min()) / r11.where(r11 != 0, 1) * 100
    
    return pd.DataFrame({
        'stress': sv,
        'd1_stoch': s1,
        'd2_stoch': s2
    }, index=high.index)