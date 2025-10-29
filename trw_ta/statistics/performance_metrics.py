import numpy as np
import pandas as pd

def _to_array(data: pd.Series):
    """Convert pandas Series to flattened NumPy array."""
    if isinstance(data, pd.Series):
        data = np.asarray(data).flatten()
    return np.asarray(data, dtype=float)

def total_return(prices: pd.Series):
    """Total return from first to last price."""
    prices = _to_array(prices)
    return (prices[-1] - prices[0]) / prices[0]

def annualized_return(prices: pd.Series, periods_per_year=252):
    """Annualized return (CAGR approximation)."""
    prices = _to_array(prices)
    n_periods = len(prices)
    total_ret = prices[-1] / prices[0]
    return total_ret ** (periods_per_year / n_periods) - 1

def mean_return(returns: pd.Series):
    """Average of period returns."""
    returns = _to_array(returns)
    return np.mean(returns)

def volatility(returns: pd.Series, periods_per_year=252):
    """Annualized standard deviation of returns."""
    returns = _to_array(returns)
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)

def sharpe_ratio(returns: pd.Series, risk_free_rate=0.0, periods_per_year=252):
    """Sharpe ratio: excess return per unit of volatility."""
    returns = _to_array(returns)
    excess = returns - risk_free_rate / periods_per_year
    return np.mean(excess) / np.std(excess, ddof=1)

def sortino_ratio(returns: pd.Series, risk_free_rate=0.0, periods_per_year=252):
    """Sortino ratio: excess return per downside deviation."""
    returns = _to_array(returns)
    excess = returns - risk_free_rate / periods_per_year
    downside = np.std(excess[excess < 0], ddof=1)
    return np.mean(excess) / downside if downside != 0 else np.nan

def information_ratio(returns: pd.Series, benchmark_returns: pd.Series):
    """Information ratio: excess return per tracking error."""
    returns = _to_array(returns)
    benchmark = _to_array(benchmark_returns)
    diff = returns - benchmark
    return np.mean(diff) / np.std(diff, ddof=1)

def treynor_ratio(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate=0.0):
    """Treynor ratio: excess return per unit of beta."""
    returns = _to_array(returns)
    benchmark = _to_array(benchmark_returns)
    beta_val = beta(returns, benchmark)
    if beta_val == 0:
        return np.nan
    avg_return = np.mean(returns)
    return (avg_return - risk_free_rate) / beta_val

def calmar_ratio(prices: pd.Series, periods_per_year=252):
    """Calmar ratio: CAGR divided by maximum drawdown."""
    cagr = annualized_return(prices, periods_per_year)
    mdd = max_drawdown(prices)
    return cagr / abs(mdd) if mdd != 0 else np.nan

def max_drawdown(prices: pd.Series):
    """Maximum drawdown (largest peak-to-trough loss)."""
    prices = _to_array(prices)
    cummax = np.maximum.accumulate(prices)
    drawdown = (prices - cummax) / cummax
    return drawdown.min()

def average_drawdown(prices: pd.Series):
    """Average of all negative drawdowns."""
    prices = _to_array(prices)
    cummax = np.maximum.accumulate(prices)
    drawdown = (prices - cummax) / cummax
    return drawdown[drawdown < 0].mean()

def recovery_factor(prices: pd.Series):
    """Recovery factor: total return divided by max drawdown."""
    total_ret = total_return(prices)
    mdd = max_drawdown(prices)
    return total_ret / abs(mdd) if mdd != 0 else np.nan

def ulcer_index(prices: pd.Series):
    """Ulcer index: RMS of drawdown percentages."""
    prices = _to_array(prices)
    peak = np.maximum.accumulate(prices)
    dd = 100 * (prices - peak) / peak
    return np.sqrt(np.mean(dd ** 2))

def hit_ratio(returns: pd.Series):
    """Hit ratio: proportion of positive returns."""
    returns = _to_array(returns)
    return np.sum(returns > 0) / len(returns)

def profit_factor(returns: pd.Series):
    """Profit factor: total gains divided by total losses."""
    returns = _to_array(returns)
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return gains / losses if losses != 0 else np.inf

def payoff_ratio(returns: pd.Series):
    """Payoff ratio: average win size divided by average loss size."""
    returns = _to_array(returns)
    avg_win = returns[returns > 0].mean()
    avg_loss = -returns[returns < 0].mean()
    return avg_win / avg_loss if avg_loss != 0 else np.inf

def expected_value(returns: pd.Series):
    """Expected value: average expected gain per trade."""
    returns = _to_array(returns)
    p_win = np.sum(returns > 0) / len(returns)
    avg_win = returns[returns > 0].mean() if np.any(returns > 0) else 0
    avg_loss = -returns[returns < 0].mean() if np.any(returns < 0) else 0
    return (p_win * avg_win) - ((1 - p_win) * avg_loss)

def alpha(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate=0.0):
    """Alpha: excess return over CAPM expected return."""
    returns = _to_array(returns)
    benchmark = _to_array(benchmark_returns)
    beta_val = beta(returns, benchmark)
    return np.mean(returns) - (risk_free_rate + beta_val * (np.mean(benchmark) - risk_free_rate))

def beta(returns: pd.Series, benchmark_returns: pd.Series):
    """Beta: sensitivity to benchmark movements."""
    returns = _to_array(returns)
    benchmark = _to_array(benchmark_returns)
    cov = np.cov(returns, benchmark)[0, 1]
    var = np.var(benchmark)
    return cov / var if var != 0 else np.nan

def omega_ratio(returns: pd.Series, threshold=0.0):
    """Omega ratio: ratio of gains to losses above threshold."""
    returns = _to_array(returns)
    gains = np.maximum(returns - threshold, 0).sum()
    losses = np.maximum(threshold - returns, 0).sum()
    return gains / losses if losses != 0 else np.inf
