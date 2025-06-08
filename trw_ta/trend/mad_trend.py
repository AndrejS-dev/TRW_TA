import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('direction', 'median', 'med_p', 'med_m')
def mad_trend(source: pd.Series, length: int, mad_mult: float) -> pd.DataFrame:
    """https://www.tradingview.com/script/nhyCYGKn-MadTrend-InvestorUnknown/"""
    median_ = ta.median(source, length)
    mad = ta.mad(source, length)

    med_p = median_ + (mad * mad_mult)
    med_m = median_ - (mad * mad_mult)

    direction = [0]

    for i in range(1, len(source)):
        prev_dir = direction[-1]
        if pd.isna(source.iloc[i]) or pd.isna(med_p.iloc[i]) or pd.isna(med_m.iloc[i]):
            direction.append(prev_dir)
            continue

        crossed_above = source.iloc[i - 1] < med_p.iloc[i - 1] and source.iloc[i] >= med_p.iloc[i]
        crossed_below = source.iloc[i - 1] > med_m.iloc[i - 1] and source.iloc[i] <= med_m.iloc[i]

        if crossed_above:
            direction.append(1)
        elif crossed_below:
            direction.append(-1)
        else:
            direction.append(prev_dir)

    return pd.DataFrame({
        'direction': direction,
        'median': median_,
        'med_p': med_p,
        'med_m': med_m
    }, index=source.index)
