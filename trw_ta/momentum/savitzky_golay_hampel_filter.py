import numpy as np
import pandas as pd
from trw_ta import register_outputs

@register_outputs('oscillator', 'signal', 'momentum')
def savitzky_golay_hampel_filter(
    close: pd.Series,
    poly_order: int = 2,
    window_size: int = 21,
    hampel_threshold: float = 3.0,
    smoothing_factor: int = 3
) -> pd.DataFrame:
    """https://www.tradingview.com/script/JqSLVPrd-Savitzky-Golay-Hampel-Filter-AlphaNatt/"""
    # Ensure window size is odd
    window = window_size if window_size % 2 == 1 else window_size + 1
    half_window = window // 2

    def get_sg_coeff(i: int, order: int, window_: int) -> float:
        """Calculate Savitzky-Golay coefficients for given index, order, and window."""
        center = window_ // 2
        norm = window_ * (window_ * window_ - 1) / 12

        if order == 2:  # Quadratic
            coeff = i - center
            weight = 3 * window_ * (window_ + 1) - 7 - 20 * coeff * coeff
            return weight / (4 * norm)
        elif order == 3:  # Cubic
            coeff = i - center
            h = coeff * coeff
            weight = (315 + h * (-420 + h * 48)) / 320
            return weight
        else:  # Higher order approximation
            return 1.0 / window_

    # Initialize arrays
    sg_filter = pd.Series(0.0, index=close.index)
    sum_weights = pd.Series(0.0, index=close.index)

    # Apply Savitzky-Golay Filter
    for i in range(window):
        idx = i - half_window
        weight = get_sg_coeff(i, poly_order, window)
        shifted_close = close.shift(-idx).ffill().bfill()
        sg_filter += shifted_close * weight
        sum_weights += abs(weight)

    sg_filter = np.where(sum_weights != 0, sg_filter / sum_weights, close)

    # Hampel Filter for Outlier Detection
    median = close.rolling(window, center=True).median().ffill().bfill()
    mad = close.rolling(window, center=True).apply(
        lambda x: np.median(np.abs(x - np.median(x))), raw=True
    ).ffill().bfill()
    mad = np.where(mad == 0, 0.001, mad)  # Prevent division by zero

    is_outlier = (np.abs(close - median) > hampel_threshold * 1.4826 * mad).astype(bool)
    cleaned_price = np.where(is_outlier, sg_filter, close)

    # Apply secondary Savitzky-Golay pass on cleaned data
    sg_final = pd.Series(0.0, index=close.index)
    for i in range(window):
        idx = i - half_window
        weight = get_sg_coeff(i, poly_order, window)
        shifted_is_outlier = is_outlier.shift(-idx, fill_value=False)
        shifted_sg_filter = pd.Series(sg_filter, index=close.index).shift(-idx).ffill().bfill()
        shifted_close = close.shift(-idx).ffill().bfill()
        shifted_price = pd.Series(
            np.where(shifted_is_outlier, shifted_sg_filter, shifted_close),
            index=close.index
        )
        sg_final += shifted_price * weight

    sg_final = np.where(sum_weights != 0, sg_final / sum_weights, cleaned_price)

    # Final smoothing with weighted moving average
    final_filter = pd.Series(sg_final, index=close.index).ewm(span=smoothing_factor, adjust=False).mean()

    # Calculate derivatives for trend detection
    first_derivative = final_filter - final_filter.shift(1)
    second_derivative = first_derivative - first_derivative.shift(1)

    # Calculate ATR for trend strength (14-period ATR approximation)
    high = close.rolling(14).max()
    low = close.rolling(14).min()
    tr = high - low
    atr = tr.ewm(span=14, adjust=False).mean()

    # Advanced trend detection
    trend_strength = np.abs(first_derivative) / atr * 100
    accelerating = second_derivative > 0

    # Signal logic
    strong_trend = trend_strength > 1.0
    price_above = close > final_filter
    rising = (final_filter > final_filter.shift(1)) & (final_filter.shift(1) > final_filter.shift(2))

    bullish = (rising & price_above) | (rising & strong_trend)
    bearish = ~rising | ~price_above

    # Return DataFrame
    return pd.DataFrame({
        'filtered_price': final_filter,
        'trend_strength': trend_strength,
        'bullish': bullish,
        'bearish': bearish
    }, index=close.index)