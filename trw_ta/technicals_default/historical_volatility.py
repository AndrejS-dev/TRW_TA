import pandas as pd
import numpy as np

def historical_volatility(close: pd.Series, length: int = 10, timeframe: str = 'daily') -> pd.Series:
    annual = 365
    timeframe = timeframe.lower()
    
    if timeframe in ['1d', 'daily', 'intraday']:
        per = 1
    else:
        per = 7  # e.g., weekly or anything longer

    log_returns = np.log(close / close.shift(1))
    stdev = log_returns.rolling(window=length).std()
    hv = 100 * stdev * np.sqrt(annual / per)
    return hv
