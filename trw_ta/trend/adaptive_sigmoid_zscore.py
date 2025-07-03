import numpy as np
import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('sigmoid_wma', 'zscore', 'z_average', 'trend')
def adaptive_sigmoid_zscore(high: pd.Series, low: pd.Series, source: pd.Series, lookback: int = 57, volatility_period: int = 41,
    base_steepness: float = 5.0, base_midpoint: float = 0.1, z_period: int = 40, ma_type: str = "HMA", ma_lookback: int = 2,
    upper_threshold: float = 1.2, lower_threshold: float = -0.2) -> pd.DataFrame:
    """https://www.tradingview.com/script/aqF5Y7Wj-Adaptive-Sigmoid-Z-Score/"""

    atr = high.rolling(volatility_period).max() - low.rolling(volatility_period).min()
    atr_avg = atr.rolling(volatility_period).mean()
    vol_ratio = atr / atr_avg

    mom = source.diff(volatility_period)
    mom_std = mom.rolling(volatility_period).std()
    norm_mom = mom / mom_std

    adaptive_steepness = base_steepness * (1 / np.maximum(vol_ratio, 0.5))
    mom_factor = norm_mom.abs() / 2
    adaptive_midpoint = base_midpoint + np.sign(mom) * np.minimum(mom_factor, 0.2)

    weighted_sum = []
    weight_sum = []
    sigmoid_wma = []

    for i in range(len(source)):
        if i < lookback or pd.isna(adaptive_steepness.iloc[i]) or pd.isna(adaptive_midpoint.iloc[i]):
            weighted_sum.append(np.nan)
            weight_sum.append(np.nan)
            sigmoid_wma.append(np.nan)
            continue

        w_sum = 0
        w_weight = 0
        for j in range(1, lookback + 1):
            if i - j < 0:
                continue

            pos = int(source.iloc[i] > source.iloc[i - j])
            neg = int(source.iloc[i] < source.iloc[i - j])
            ratio = (pos - neg) / lookback

            steep = adaptive_steepness.iloc[i]
            mid = adaptive_midpoint.iloc[i]
            weight = 1 / (1 + np.exp(-steep * (np.abs(ratio) - mid)))

            w_sum += source.iloc[i - j] * weight
            w_weight += weight

        weighted_sum.append(w_sum)
        weight_sum.append(w_weight)
        sigmoid_wma.append(w_sum / w_weight if w_weight != 0 else source.iloc[i])


    df = pd.DataFrame()
    df['sigmoid_wma'] = sigmoid_wma
    df['zscore'] = (source - df['sigmoid_wma']) / source.rolling(z_period).std()
    df['z_average'] = ta.ma(df['zscore'], ma_lookback, ma_type)

    trend = [0]
    for i in range(1, len(df)):
        za_prev = df['z_average'].iloc[i - 1]
        za_curr = df['z_average'].iloc[i]

        if pd.isna(za_curr) or pd.isna(za_prev):
            trend.append(trend[-1])
        elif za_prev < upper_threshold and za_curr >= upper_threshold:
            trend.append(1)
        elif za_prev > lower_threshold and za_curr <= lower_threshold:
            trend.append(-1)
        else:
            trend.append(trend[-1])

    df['trend'] = trend

    return df[['sigmoid_wma', 'zscore', 'z_average', 'trend']]
