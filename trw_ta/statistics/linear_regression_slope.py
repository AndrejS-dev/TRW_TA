import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('slope')
def linear_regression_slope(series: pd.Series, length: int = 14) -> pd.Series:
    x = np.arange(length)
    x_mean = x.mean()
    denominator = ((x - x_mean)**2).sum()

    def calc_slope(y):
        y_mean = y.mean()
        numerator = ((x - x_mean) * (y - y_mean)).sum()
        return numerator / denominator if denominator != 0 else np.nan

    return series.rolling(length).apply(calc_slope, raw=True).rename("linreg_slope")
