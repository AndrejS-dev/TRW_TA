import pandas as pd
import trw_ta as ta

def BBTrend(source: pd.Series, short_length: int = 20, long_length: int = 50, stdev_mult: float = 2.0) -> pd.Series:
    df_short = pd.DataFrame()
    df_long = pd.DataFrame()

    df_short[["Upper", "Middle", "Lower"]] = ta.bollinger_bands(source, short_length, stdev_mult)
    df_long[["Upper", "Middle", "Lower"]] = ta.bollinger_bands(source, long_length, stdev_mult)

    return (abs(df_short['Lower'] - df_long['Lower']) - abs(df_short['Upper'] - df_long['Upper'])) / df_short['Middle'] * 100