import pandas as pd
from trw_ta import register_outputs

@register_outputs('rs')
def reversal_scalper(high: pd.Series, low: pd.Series, close: pd.Series, k_length: int = 8, k_smoothing: int = 5) -> pd.Series:
    """
    Reversal scalper.
    
    Python conversion of the TradingView / Pine Script indicator.
    
    Source: https://www.tradingview.com/script/766XyHk5-Reversal-Scalper-Adib-Noorani/
    
    Parameters
    ----------
    high : pd.Series
        High price series.
    low : pd.Series
        Low price series.
    close : pd.Series
        Close price series.
    k_length : int, default 8
        Lookback / smoothing period.
    k_smoothing : int, default 5
        K smoothing.
    
    Returns
    -------
    pd.Series
        Series of computed values.
    """
    lowest_low = low.rolling(window=k_length).min()
    highest_high = high.rolling(window=k_length).max()
    stoch_k = 100 * (close - lowest_low) / (highest_high - lowest_low)

    reversal_strength = stoch_k.rolling(window=k_smoothing).mean()
    
    return reversal_strength