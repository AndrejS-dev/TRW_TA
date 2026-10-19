import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('bb_trend')
def bb_trend(source: pd.Series, short_length: int = 20, long_length: int = 50, stdev_mult: float = 2.0) -> pd.Series:
    """
    Bollinger Band Trend.
    
    Parameters
    ----------
    source : pd.Series
        Input price or value series.
    short_length : int, default 20
        Short lookback period.
    long_length : int, default 50
        Long lookback period.
    stdev_mult : float, default 2.0
        Standard-deviation multiplier.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    df_short = pd.DataFrame()
    df_long = pd.DataFrame()

    df_short[["Upper", "Middle", "Lower"]] = ta.bollinger_bands(source, short_length, stdev_mult)
    df_long[["Upper", "Middle", "Lower"]] = ta.bollinger_bands(source, long_length, stdev_mult)

    return (abs(df_short['Lower'] - df_long['Lower']) - abs(df_short['Upper'] - df_long['Upper'])) / df_short['Middle'] * 100

# Backward-compatible alias
BBTrend = bb_trend
