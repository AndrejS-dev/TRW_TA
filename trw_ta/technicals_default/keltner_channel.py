import pandas as pd
import trw_ta as ta

def keltner_channels(high: pd.Series, low: pd.Series, close: pd.Series, length: int = 20, mult: float = 2.0,
    atr_length: int = 10, use_ema: bool = True,bands_style: str = "Average True Range") -> pd.Series:  # Options: "Average True Range", "True Range", "Range"
    ma = ta.ema(close, length) if use_ema else ta.sma(close, length)

    if bands_style == "True Range":
        range_val = ta.tr(high, low, close)
    elif bands_style == "Average True Range":
        range_val = ta.atr(high, low, close, atr_length)
    elif bands_style == "Range":
        range_val = ta.rma(high - low, length)
    else:
        raise ValueError("Invalid bands_style. Choose from 'Average True Range', 'True Range', or 'Range'.")

    upper = ma + mult * range_val
    lower = ma - mult * range_val

    return pd.DataFrame({"upper": upper, "ma": ma, "lower": lower})