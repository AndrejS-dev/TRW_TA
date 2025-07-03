import numpy as np
import pandas as pd
from trw_ta import register_outputs

@register_outputs('cpm_scaled', 'trend')
def fdi_cumulative_price_momentum(source: pd.Series, period: int = 48, speed_factor: float = 42,
            upper_threshold: float = 0.65, lower_threshold: float = 0.30) -> pd.DataFrame:
    """https://www.tradingview.com/script/NI2rdYTo-FDI-Cumulative-Price-Momentum/"""

    df = pd.DataFrame(index=source.index)
    df['source'] = source

    fdi_values = []
    for i in range(len(source)):
        if i + 1 < period:
            fdi_values.append(np.nan)
            continue

        window = source.iloc[i - period + 1:i + 1]
        high_val = window.max()
        low_val = window.min()
        norm_range = high_val - low_val

        if norm_range == 0:
            fdi_values.append(np.nan)
            continue

        sum_length = 0.0
        for j in range(1, period):
            diff_curr = (window.iloc[j - 1] - low_val) / norm_range
            diff_next = (window.iloc[j] - low_val) / norm_range
            sum_length += np.sqrt((diff_curr - diff_next) ** 2 + (1 / period) ** 2)

        if sum_length <= 0 or np.isnan(sum_length):
            fdi = np.nan
        else:
            fdi = 1 + (np.log(sum_length) + np.log(2)) / np.log(2 * period)
        fdi_values.append(fdi)

    df['fdi'] = fdi_values

    df['trail_dim'] = 1 / (2 - df['fdi'])
    df['adaptive_speed'] = df['trail_dim'] / 2
    df['len'] = (speed_factor * df['adaptive_speed']).round().clip(lower=1).astype('Int64')

    diff = df['source'].diff()
    df['scaled_diff'] = diff * np.sqrt(df['len'])
    df['cpm_raw'] = df['scaled_diff'].cumsum()

    # CPM Scaling
    cpm_scaled = []
    for i in range(len(df)):
        l = df['len'].iloc[i]
        if pd.isna(l) or i < l - 1:
            cpm_scaled.append(np.nan)
            continue
        window = df['cpm_raw'].iloc[i - l + 1:i + 1]
        min_val = window.min()
        max_val = window.max()
        norm = (df['cpm_raw'].iloc[i] - min_val) / (max_val - min_val) if max_val != min_val else np.nan
        cpm_scaled.append(norm)
    df['cpm_scaled'] = cpm_scaled

    df['long'] = (df['cpm_scaled'] > upper_threshold) & (df['cpm_scaled'].shift(1) <= upper_threshold)
    df['short'] = (df['cpm_scaled'] < lower_threshold) & (df['cpm_scaled'].shift(1) >= lower_threshold)

    trend = [0]
    for i in range(1, len(df)):
        if df['long'].iloc[i] and not df['short'].iloc[i]:
            trend.append(1)
        elif df['short'].iloc[i]:
            trend.append(-1)
        else:
            trend.append(trend[-1])
    df['trend'] = trend

    return df[['cpm_scaled', 'trend']]
