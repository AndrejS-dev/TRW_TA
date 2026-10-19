import numpy as np
import pandas as pd

def _to_array(data: pd.Series):
    """
    To array.
    
    Convert pandas Series to flattened NumPy array.
    
    Parameters
    ----------
    data : pd.Series
        Data.
    """
    if isinstance(data, pd.Series):
        data = np.asarray(data).flatten()
    return np.asarray(data, dtype=float)

def total_return(prices: pd.Series):
    """
    Total return.
    
    Total return from first to last price.
    
    Parameters
    ----------
    prices : pd.Series
        Price series.
    """
    prices = _to_array(prices)
    return (prices[-1] - prices[0]) / prices[0]

def annualized_return(prices: pd.Series, periods_per_year=252):
    """
    Annualized return.
    
    Parameters
    ----------
    prices : pd.Series
        Price series.
    periods_per_year : Any, default 252
        Bars per year for annualization.
    """
    prices = _to_array(prices)
    n_periods = len(prices)
    total_ret = prices[-1] / prices[0]
    return total_ret ** (periods_per_year / n_periods) - 1

def mean_return(returns: pd.Series):
    """
    Mean return.
    
    Average of period returns.
    
    Parameters
    ----------
    returns : pd.Series
        Period return series.
    """
    returns = _to_array(returns)
    return np.mean(returns)

def volatility(returns: pd.Series, periods_per_year=252):
    """
    Volatility.
    
    Annualized standard deviation of returns.
    
    Parameters
    ----------
    returns : pd.Series
        Period return series.
    periods_per_year : Any, default 252
        Bars per year for annualization.
    """
    returns = _to_array(returns)
    return np.std(returns, ddof=1) * np.sqrt(periods_per_year)

def sharpe_ratio(returns: pd.Series, risk_free_rate=0.0, periods_per_year=252):
    """
    Sharpe ratio.
    
    Sharpe ratio: excess return per unit of volatility.
    
    Parameters
    ----------
    returns : pd.Series
        Period return series.
    risk_free_rate : Any, default 0.0
        Risk-free rate (period-compatible).
    periods_per_year : Any, default 252
        Bars per year for annualization.
    """
    returns = _to_array(returns)
    excess = returns - risk_free_rate / periods_per_year
    return np.mean(excess) / np.std(excess, ddof=1)

def sortino_ratio(returns: pd.Series, risk_free_rate=0.0, periods_per_year=252):
    """
    Sortino ratio.
    
    Sortino ratio: excess return per downside deviation.
    
    Parameters
    ----------
    returns : pd.Series
        Period return series.
    risk_free_rate : Any, default 0.0
        Risk-free rate (period-compatible).
    periods_per_year : Any, default 252
        Bars per year for annualization.
    """
    returns = _to_array(returns)
    excess = returns - risk_free_rate / periods_per_year
    downside = np.std(excess[excess < 0], ddof=1)
    return np.mean(excess) / downside if downside != 0 else np.nan

def information_ratio(returns: pd.Series, benchmark_returns: pd.Series):
    """
    Information ratio.
    
    Information ratio: excess return per tracking error.
    
    Parameters
    ----------
    returns : pd.Series
        Period return series.
    benchmark_returns : pd.Series
        Benchmark returns.
    """
    returns = _to_array(returns)
    benchmark = _to_array(benchmark_returns)
    diff = returns - benchmark
    return np.mean(diff) / np.std(diff, ddof=1)

def treynor_ratio(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate=0.0):
    """
    Treynor ratio.
    
    Treynor ratio: excess return per unit of beta.
    
    Parameters
    ----------
    returns : pd.Series
        Period return series.
    benchmark_returns : pd.Series
        Benchmark returns.
    risk_free_rate : Any, default 0.0
        Risk-free rate (period-compatible).
    """
    returns = _to_array(returns)
    benchmark = _to_array(benchmark_returns)
    beta_val = beta(returns, benchmark)
    if beta_val == 0:
        return np.nan
    avg_return = np.mean(returns)
    return (avg_return - risk_free_rate) / beta_val

def calmar_ratio(prices: pd.Series, periods_per_year=252):
    """
    Calmar ratio.
    
    Calmar ratio: CAGR divided by maximum drawdown.
    
    Parameters
    ----------
    prices : pd.Series
        Price series.
    periods_per_year : Any, default 252
        Bars per year for annualization.
    """
    cagr = annualized_return(prices, periods_per_year)
    mdd = max_drawdown(prices)
    return cagr / abs(mdd) if mdd != 0 else np.nan

def max_drawdown(prices: pd.Series):
    """
    Max drawdown.
    
    Maximum drawdown (largest peak-to-trough loss).
    
    Parameters
    ----------
    prices : pd.Series
        Price series.
    """
    prices = _to_array(prices)
    cummax = np.maximum.accumulate(prices)
    drawdown = (prices - cummax) / cummax
    return drawdown.min()

def average_drawdown(prices: pd.Series):
    """
    Average drawdown.
    
    Average of all negative drawdowns.
    
    Parameters
    ----------
    prices : pd.Series
        Price series.
    """
    prices = _to_array(prices)
    cummax = np.maximum.accumulate(prices)
    drawdown = (prices - cummax) / cummax
    return drawdown[drawdown < 0].mean()

def recovery_factor(prices: pd.Series):
    """
    Recovery factor.
    
    Recovery factor: total return divided by max drawdown.
    
    Parameters
    ----------
    prices : pd.Series
        Price series.
    """
    total_ret = total_return(prices)
    mdd = max_drawdown(prices)
    return total_ret / abs(mdd) if mdd != 0 else np.nan

def ulcer_index(prices: pd.Series):
    """
    Ulcer index.
    
    Ulcer index: RMS of drawdown percentages.
    
    Parameters
    ----------
    prices : pd.Series
        Price series.
    """
    prices = _to_array(prices)
    peak = np.maximum.accumulate(prices)
    dd = 100 * (prices - peak) / peak
    return np.sqrt(np.mean(dd ** 2))

def hit_ratio(returns: pd.Series):
    """
    Hit ratio.
    
    Hit ratio: proportion of positive returns.
    
    Parameters
    ----------
    returns : pd.Series
        Period return series.
    """
    returns = _to_array(returns)
    return np.sum(returns > 0) / len(returns)

def profit_factor(returns: pd.Series):
    """
    Profit factor.
    
    Profit factor: total gains divided by total losses.
    
    Parameters
    ----------
    returns : pd.Series
        Period return series.
    """
    returns = _to_array(returns)
    gains = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    return gains / losses if losses != 0 else np.inf

def payoff_ratio(returns: pd.Series):
    """
    Payoff ratio.
    
    Payoff ratio: average win size divided by average loss size.
    
    Parameters
    ----------
    returns : pd.Series
        Period return series.
    """
    returns = _to_array(returns)
    avg_win = returns[returns > 0].mean()
    avg_loss = -returns[returns < 0].mean()
    return avg_win / avg_loss if avg_loss != 0 else np.inf

def expected_value(returns: pd.Series):
    """
    Expected value.
    
    Expected value: average expected gain per trade.
    
    Parameters
    ----------
    returns : pd.Series
        Period return series.
    """
    returns = _to_array(returns)
    p_win = np.sum(returns > 0) / len(returns)
    avg_win = returns[returns > 0].mean() if np.any(returns > 0) else 0
    avg_loss = -returns[returns < 0].mean() if np.any(returns < 0) else 0
    return (p_win * avg_win) - ((1 - p_win) * avg_loss)

def alpha(returns: pd.Series, benchmark_returns: pd.Series, risk_free_rate=0.0):
    """
    Alpha.
    
    Alpha: excess return over CAPM expected return.
    
    Parameters
    ----------
    returns : pd.Series
        Period return series.
    benchmark_returns : pd.Series
        Benchmark returns.
    risk_free_rate : Any, default 0.0
        Risk-free rate (period-compatible).
    """
    returns = _to_array(returns)
    benchmark = _to_array(benchmark_returns)
    beta_val = beta(returns, benchmark)
    return np.mean(returns) - (risk_free_rate + beta_val * (np.mean(benchmark) - risk_free_rate))

def beta(returns: pd.Series, benchmark_returns: pd.Series):
    """
    Beta.
    
    Beta: sensitivity to benchmark movements.
    
    Parameters
    ----------
    returns : pd.Series
        Period return series.
    benchmark_returns : pd.Series
        Benchmark returns.
    """
    returns = _to_array(returns)
    benchmark = _to_array(benchmark_returns)
    cov = np.cov(returns, benchmark)[0, 1]
    var = np.var(benchmark)
    return cov / var if var != 0 else np.nan

def omega_ratio(returns: pd.Series, threshold=0.0):
    """
    Omega ratio.
    
    Omega ratio: ratio of gains to losses above threshold.
    
    Parameters
    ----------
    returns : pd.Series
        Period return series.
    threshold : Any, default 0.0
        Threshold.
    """
    returns = _to_array(returns)
    gains = np.maximum(returns - threshold, 0).sum()
    losses = np.maximum(threshold - returns, 0).sum()
    return gains / losses if losses != 0 else np.inf
