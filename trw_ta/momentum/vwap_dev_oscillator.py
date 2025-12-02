# trw_ta/momentum/vwap_dev_oscillator.py
import pandas as pd
import numpy as np
from trw_ta import register_outputs


@register_outputs('vdo', 'vdo_sigma', 'vdo_sig')
def vwap_dev_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series,
    lookback: int = 20, use_close: bool = False,
    dev_mode: str = "Absolute", z_win: int = 50,
    pct_vol_lookback: int = 100, pct_min_sigma: float = 0.1,
    abs_vol_lookback: int = 100
) -> pd.DataFrame:

    # --- Input Price Reference ---
    price_ref = close if use_close else (high + low + close) / 3

    # --- VWAP Calculation ---
    cum_vol_price = (price_ref * volume).rolling(lookback).sum()
    cum_vol = volume.rolling(lookback).sum()
    vwap = cum_vol_price / (cum_vol + 1e-9)

    # --- Deviation Calculations ---
    resid_abs = price_ref - vwap
    resid_pct = 100 * (price_ref / vwap - 1)

    mu_r = resid_abs.rolling(z_win).mean()
    sd_r = resid_abs.rolling(z_win).std()
    z_raw = (resid_abs - mu_r) / (sd_r + 1e-9)

    osc = {"Percent": resid_pct, "Absolute": resid_abs, "Z-Score": z_raw}[dev_mode]

    # --- Volatility (Sigma) Levels ---
    sigma_pct = resid_pct.rolling(pct_vol_lookback).std().clip(lower=pct_min_sigma)
    sigma_abs = resid_abs.rolling(abs_vol_lookback).std().clip(lower=1.0)
    sigma = {
        "Percent": sigma_pct,
        "Absolute": sigma_abs,
        "Z-Score": pd.Series(1.0, index=osc.index)
    }[dev_mode]

    # --- Signal (Polarity) - Handle NaNs ---
    sig_float = np.sign(osc)  # Returns -1.0, 0.0, 1.0 or NaN
    sig = pd.Series(sig_float, index=osc.index).fillna(0).astype(int)

    # --- Output ---
    return pd.DataFrame({
        "vdo": osc,
        "vdo_sigma": sigma,
        "vdo_sig": sig
    }, index=high.index)