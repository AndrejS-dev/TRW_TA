import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('kama', 'signal')
def gaussian_mixture_model(close: pd.Series, training_period: int = 100, n_components: int = 3, momentum_length: int = 14, 
                          learning_rate: float = 0.3, smoothing: int = 3) -> pd.DataFrame:
    """https://www.tradingview.com/script/Ak2YzGrb-Machine-Learning-Gaussian-Mixture-Model-AlphaNatt/"""
    # Input validation
    if not isinstance(close, pd.Series):
        raise TypeError("Close must be a pandas Series")
    if close.isna().any():
        raise ValueError("Close must not contain NaN values")
    if len(close) < max(training_period, momentum_length, 20):
        raise ValueError(f"Close must have at least {max(training_period, momentum_length, 20)} rows")
    if training_period < 50 or training_period > 500:
        raise ValueError("Training period must be between 50 and 500")
    if n_components < 2 or n_components > 5:
        raise ValueError("Number of components must be between 2 and 5")
    if momentum_length < 5 or momentum_length > 30:
        raise ValueError("Momentum length must be between 5 and 30")
    if learning_rate < 0.1 or learning_rate > 1.0:
        raise ValueError("Learning rate must be between 0.1 and 1.0")
    if smoothing < 1 or smoothing > 10:
        raise ValueError("Smoothing must be between 1 and 10")

    # Calculate features
    momentum = (close / close.shift(momentum_length) - 1) * 100  # ROC
    volatility = close.rolling(window=20, min_periods=1).std()
    volume = pd.Series(1, index=close.index)  # Placeholder for volume
    volume_ratio = volume / volume.rolling(window=20, min_periods=1).mean()

    # Normalize features
    def normalize(val, length):
        min_val = val.rolling(window=length, min_periods=1).min()
        max_val = val.rolling(window=length, min_periods=1).max()
        range_val = max_val - min_val
        return (val - min_val) / range_val.where(range_val != 0, 1).replace(0, 0.5)

    norm_momentum = normalize(momentum, training_period)
    norm_volatility = normalize(volatility, training_period)
    norm_volume = normalize(volume_ratio, training_period)

    # Initialize component means
    mean_m = [0.25, 0.50, 0.75]  # Momentum: low, normal, high
    mean_v = [0.30, 0.50, 0.70]  # Volatility: low, normal, high

    # Gaussian probability density function
    def gaussian_prob(x, mean, variance):
        variance = max(variance, 0.001)  # Prevent division by zero
        exp_val = -0.5 * ((x - mean) ** 2) / variance
        return 1.0 / np.sqrt(2 * np.pi * variance) * np.exp(exp_val)

    # E-M Algorithm (simplified)
    prob_m = [gaussian_prob(norm_momentum, mean_m[i], 0.1) for i in range(n_components)]
    prob_v = [gaussian_prob(norm_volatility, mean_v[i], 0.1) for i in range(n_components)]
    prob = [prob_m[i] * prob_v[i] for i in range(n_components)]
    total_prob = sum(prob)
    total_prob = total_prob.where(total_prob > 0, 1)
    resp = [prob[i] / total_prob for i in range(n_components)]

    # Update means every 10 bars
    mean_m_series = [pd.Series(mean_m[i], index=close.index) for i in range(n_components)]
    mean_v_series = [pd.Series(mean_v[i], index=close.index) for i in range(n_components)]
    for i in range(1, len(close)):
        for j in range(n_components):
            if i % 10 == 0:
                mean_m_series[j].iloc[i] = mean_m_series[j].iloc[i-1] * (1 - learning_rate) + norm_momentum.iloc[i] * resp[j].iloc[i] * learning_rate
                mean_v_series[j].iloc[i] = mean_v_series[j].iloc[i-1] * (1 - learning_rate) + norm_volatility.iloc[i] * resp[j].iloc[i] * learning_rate
            else:
                mean_m_series[j].iloc[i] = mean_m_series[j].iloc[i-1]
                mean_v_series[j].iloc[i] = mean_v_series[j].iloc[i-1]

    # Determine current regime
    current_regime = pd.Series(0, index=close.index)
    for i in range(len(close)):
        resp_values = [resp[j].iloc[i] for j in range(n_components)]
        current_regime.iloc[i] = np.argmax(resp_values) + 1

    # Calculate regime-specific momentums
    def rsi(close, period):
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(window=period, min_periods=1).mean()
        loss = (-delta).where(delta < 0, 0).rolling(window=period, min_periods=1).mean()
        rs = gain / loss.where(loss != 0, 1)
        return 100 - 100 / (1 + rs)

    regime1_momentum = rsi(close, momentum_length) - 50
    regime2_momentum = momentum * 5
    regime3_momentum = (close - close.rolling(window=momentum_length, min_periods=1).mean()) / close.rolling(window=momentum_length, min_periods=1).std() * 20

    # Weighted momentum
    weighted_momentum = regime1_momentum * resp[0] + regime2_momentum * resp[1] + regime3_momentum * resp[2]
    weighted_momentum = weighted_momentum.fillna(0)

    # Smooth with EMA
    kama = weighted_momentum.ewm(span=smoothing, adjust=False).mean()

    # Signal calculation
    signal = pd.Series(0, index=close.index)
    last_signal = 0
    for i in range(1, len(close)):
        if kama.iloc[i] > 0 and kama.iloc[i-1] <= 0:  # Crossover above 0
            last_signal = 1
        elif kama.iloc[i] < 0 and kama.iloc[i-1] >= 0:  # Crossunder below 0
            last_signal = -1
        signal.iloc[i] = last_signal
    signal = signal.fillna(method='ffill').fillna(0)

    return pd.DataFrame({
        'kama': kama,
        'signal': signal
    }, index=close.index)