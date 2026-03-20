import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('mid', 'upper', 'lower')
def recursive_ls(close: pd.Series, lambda_: float = 0.98, mult: float = 2.0) -> pd.DataFrame:
    """https://www.tradingview.com/script/ASDKkNUJ-Recursive-Least-Squares-Forecast-LuxAlgo/"""
    n = len(close)
    if n == 0:
        return pd.DataFrame(index=close.index, columns=['mid', 'upper', 'lower'])

    t = np.arange(n, dtype=float)

    # Initialize state
    w1 = close.iloc[0] * 1.0           # intercept ≈ first price
    w2 = 0.0                           # slope
    p11 = 1000.0
    p12 = 0.0
    p22 = 1000.0
    msqe = 0.0

    mid_arr   = np.full(n, np.nan, dtype=float)
    upper_arr = np.full(n, np.nan, dtype=float)
    lower_arr = np.full(n, np.nan, dtype=float)

    mid_arr[0] = w1
    rls_std = 0.0
    upper_arr[0] = w1 + mult * rls_std
    lower_arr[0] = w1 - mult * rls_std

    for i in range(1, n):
        y = close.iloc[i]
        ti = t[i]

        y_hat = w1 + w2 * ti
        error = y - y_hat

        msqe = max(0.0, lambda_ * msqe + (1 - lambda_) * error**2)
        rls_std = np.sqrt(msqe)

        p11_t_p12 = p11 + ti * p12
        p12_t_p22 = p12 + ti * p22
        denom = lambda_ + (p11_t_p12 + ti * p12_t_p22)

        if abs(denom) < 1e-12:
            k1 = 0.0
            k2 = 0.0
        else:
            k1 = p11_t_p12 / denom
            k2 = p12_t_p22 / denom

        w1 += k1 * error
        w2 += k2 * error

        # Update covariance matrix P
        p11 = (p11 - k1 * p11_t_p12) / lambda_
        p12 = (p12 - k1 * p12_t_p22) / lambda_
        p22 = (p22 - k2 * p12_t_p22) / lambda_

        current_mean = w1 + w2 * ti
        mid_arr[i]   = current_mean
        upper_arr[i] = current_mean + rls_std * mult
        lower_arr[i] = current_mean - rls_std * mult

    return pd.DataFrame({
        'mid':   mid_arr,
        'upper': upper_arr,
        'lower': lower_arr,
    }, index=close.index).rename_axis(None, axis=1)