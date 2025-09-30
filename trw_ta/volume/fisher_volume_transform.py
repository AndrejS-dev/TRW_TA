import numpy as np
import pandas as pd
from trw_ta import register_outputs

@register_outputs('oscillator', 'signal', 'momentum')
def fisher_volume_transform(
    close: pd.Series,
    volume: pd.Series,
    fisher_period: int = 10,
    volume_weight: float = 0.7,
    smoothing: int = 3
) -> pd.DataFrame:
    """https://www.tradingview.com/script/fpDEoqVB-Fisher-Volume-Transform-AlphaNatt/"""
    def fisher_transform(src: pd.Series, length: int) -> pd.Series:
        # Calculate highest and lowest over the period
        highest = src.rolling(length).max()
        lowest = src.rolling(length).min()
        range_vals = highest - lowest
        
        # Normalize price to -1 to 1 range
        normalized = np.where(range_vals != 0, ((src - lowest) / range_vals - 0.5) * 2, 0)
        normalized = np.clip(normalized, -0.999, 0.999)
        
        # Apply Fisher Transform formula
        fish = 0.5 * np.log((1 + normalized) / (1 - normalized))
        return pd.Series(fish, index=src.index)
    
    vol_sma = volume.rolling(fisher_period).mean()
    vw_price = close * (volume / vol_sma)
    
    blended_price = close * (1 - volume_weight) + vw_price * volume_weight
    
    fisher = fisher_transform(blended_price, fisher_period)
    fisher_smooth = fisher.ewm(span=smoothing, adjust=False).mean()
    
    # Scale to oscillator range (-100 to 100)
    scaled_fisher = fisher_smooth * 25
    
    signal = scaled_fisher.ewm(span=smoothing * 2, adjust=False).mean()
    momentum = scaled_fisher - scaled_fisher.shift(1)
    
    return pd.DataFrame({
        'oscillator': scaled_fisher,
        'signal': signal,
        'momentum': momentum
    }, index=close.index)