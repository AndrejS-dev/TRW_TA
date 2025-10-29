import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('DataFrame')
def generate_anchored_synthetic_prices(data: pd.DataFrame, num_intermediate_points: int, include_trend=True, include_volatility_shifts=False, random_seed=None):
    """
    Generate a synthetic OHLC price series that follows the statistical properties
    of a real dataset while being "anchored" to selected real data points.
    
    Parameters
    ----------
    data : pd.DataFrame
        Historical OHLC data containing 'open', 'high', 'low', 'close' columns.
    num_intermediate_points : int
        Number of randomly selected intermediate anchor points between the first and last row.
        These anchors ensure that the synthetic series reconnects with the real price path.
    include_trend : bool, default=True
        Whether to include the empirical drift (average log-return) of the original data.
    include_volatility_shifts : bool, default=False
        Whether to simulate sudden volatility regime changes during generation.
    random_seed : int, optional
        Random seed for reproducibility.
    
    Returns
    -------
    pd.DataFrame
        Synthetic OHLC time series indexed by dates.
        The generated series is continuous and respects anchor prices.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Ensure required columns exist
    required_columns = ['open', 'high', 'low', 'close']
    if not all(col in data.columns for col in required_columns):
        raise ValueError("DataFrame must contain 'open', 'high', 'low', 'close' columns")
    
    # Calculate daily returns and derive volatility and drift
    returns = np.log(data['close'] / data['close'].shift(1)).dropna()
    volatility = returns.std()
    drift = returns.mean() if include_trend else 0.0

    # Select anchor points
    n_points = len(data)
    if num_intermediate_points < 2:
        raise ValueError("Number of Intermediate points must be greater or equal to 2")
    
    # Select indices for anchor points (first, last, and random intermediates)
    anchor_indices = [0]  # First point
    if num_intermediate_points > 0:
        intermediate_indices = np.random.choice(
            range(1, n_points - 1), 
            size=num_intermediate_points, 
            replace=False
        )
        anchor_indices.extend(intermediate_indices)
    anchor_indices.append(n_points - 1)  # Last point
    anchor_indices = sorted(anchor_indices)
    
    # Initialize output DataFrame
    synthetic_data = []
    dates = data.index
    
    for i in range(len(anchor_indices) - 1):
        start_idx = anchor_indices[i]
        end_idx = anchor_indices[i + 1]
        
        # Get real data points
        start_price = data['close'].iloc[start_idx]
        end_price = data['close'].iloc[end_idx]
        segment_dates = pd.date_range(start=dates[start_idx], end=dates[end_idx], freq='D')
        n_days = len(segment_dates)
        
        # Generate synthetic data between anchor points
        current_price = start_price
        open_prices, high_prices, low_prices, close_prices = [], [], [], []
        
        for day in range(n_days):
            current_vol = volatility
            if include_volatility_shifts and np.random.rand() < 0.01:
                current_vol *= np.random.uniform(0.5, 2.5)
            
            # Calculate dynamic drift to guide price toward end_price
            if day == n_days - 1:
                new_close = end_price  # Ensure exact match at anchor point
            else:
                # Calculate remaining days and target return
                remaining_days = n_days - day - 1
                if remaining_days > 0:
                    target_return = np.log(end_price / current_price) / remaining_days
                    # Blend market drift and target drift (weighted toward target as we near end)
                    weight = min(1.0, (day / n_days) ** 2)  # Quadratic weight for smoother convergence
                    effective_drift = (1 - weight) * drift + weight * target_return
                else:
                    effective_drift = drift
                
                ret = np.random.normal(loc=effective_drift, scale=current_vol)
                new_close = current_price * np.exp(ret)
            
            new_open = current_price
            high = max(new_open, new_close) * (1 + np.random.uniform(0, 0.01))
            low = min(new_open, new_close) * (1 - np.random.uniform(0, 0.01))
            
            open_prices.append(new_open)
            high_prices.append(high)
            low_prices.append(low)
            close_prices.append(new_close)
            
            current_price = new_close
        
        # Create segment DataFrame
        segment_df = pd.DataFrame({
            'open': open_prices,
            'high': high_prices,
            'low': low_prices,
            'close': close_prices
        }, index=segment_dates)
        
        synthetic_data.append(segment_df)
    
    # Concatenate all segments
    final_df = pd.concat(synthetic_data)
    # Ensure no duplicate dates
    final_df = final_df[~final_df.index.duplicated(keep='first')]
    
    return final_df