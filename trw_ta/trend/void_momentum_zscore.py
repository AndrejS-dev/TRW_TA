import pandas as pd
import numpy as np
from trw_ta import ema
from trw_ta import register_outputs

@register_outputs('z_score', 'z_score_ema')
def void_momentum_zscore(source: pd.Series, a: int = 10, b: int = 30, regime: str = "STH") -> pd.DataFrame:

    def momentum_sum_fl(src, a, b):
        mom = pd.Series(index=src.index, dtype=float)
        for i in range(a, b + 1):
            delta = src - src.shift(i)
            mom = delta if mom.isna().all() else mom.add(delta, fill_value=0)
        return mom

    mom_sum = momentum_sum_fl(source, a, b)

    short_mean = mom_sum.rolling(b - a).mean()
    short_std = mom_sum.rolling(b - a).std()

    long_len = b * a
    long_mean = mom_sum.rolling(long_len).mean()
    long_std = mom_sum.rolling(long_len).std()

    if regime == "LTH":
        u_mean, u_std = long_mean, long_std
        s_mean, s_std = short_mean, short_std
    else:
        u_mean, u_std = short_mean, short_std
        s_mean, s_std = long_mean, long_std

    mom_sum_ot = (mom_sum - s_mean) / s_std
    mom_sum_z = (mom_sum - u_mean) / u_std
    mom_sum_ema = ema(mom_sum_z, int(np.ceil(np.sqrt((b - a) * 2))))

    mom_sig = pd.Series(index=source.index, dtype=int)
    prev_sig = 0
    for i in range(len(source)):
        val = mom_sum_ema.iloc[i]
        if pd.isna(val):
            mom_sig.iloc[i] = prev_sig
        elif val > 0:
            mom_sig.iloc[i] = 1
            prev_sig = 1
        elif val < 0:
            mom_sig.iloc[i] = -1
            prev_sig = -1
        else:
            mom_sig.iloc[i] = prev_sig

    return pd.DataFrame({
        'mom_sum_z': mom_sum_z,
        'mom_sum_ema': mom_sum_ema,
    })
