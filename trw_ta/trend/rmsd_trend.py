import pandas as pd
import numpy as np
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('direction', 'avg', 'avg_p', 'avg_m')
def rmsd_trend(source: pd.Series, length: int, aad_mult: float, avg_type: str) -> pd.DataFrame:
    """https://www.tradingview.com/script/7GzOiU0f-RMSD-Trend-InvestorUnknown/"""
    avg = ta.ma(source, length, avg_type)
    rmsd = ta.rmsd(source, avg, length)

    avg_p = avg + (rmsd * aad_mult)
    avg_m = avg - (rmsd * aad_mult)

    direction = [0]

    for i in range(1, len(source)):
        prev_dir = direction[-1]
        if pd.isna(source.iloc[i]) or pd.isna(avg_p.iloc[i]) or pd.isna(avg_m.iloc[i]):
            direction.append(prev_dir)
            continue

        crossed_above = source.iloc[i - 1] < avg_p.iloc[i - 1] and source.iloc[i] >= avg_p.iloc[i]
        crossed_below = source.iloc[i - 1] > avg_m.iloc[i - 1] and source.iloc[i] <= avg_m.iloc[i]

        if crossed_above:
            direction.append(1)
        elif crossed_below:
            direction.append(-1)
        else:
            direction.append(prev_dir)

    return pd.DataFrame({
        'direction': direction,
        'avg': avg,
        'avg_p': avg_p,
        'avg_m': avg_m
    }, index=source.index)