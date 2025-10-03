import numpy as np
import pandas as pd
from trw_ta import register_outputs

@register_outputs('smooth_trend')
def market_sentiment_trend_gauge(
    close: pd.Series,
    benchmark_close: pd.Series,
    rsi_length: int = 14,
    fast_intra: int = 10,
    slow_intra: int = 21,
    fast_daily: int = 5,
    slow_daily: int = 20,
    fast_week: int = 3,
    slow_week: int = 10,
    bb_length: int = 20,
    bb_mult: float = 2.0
) -> pd.Series:
    """https://www.tradingview.com/script/OSdqOjNR-Market-Sentiment-Trend-Gauge-LevelUp/"""
    # RSI score
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain, index=close.index).rolling(window=rsi_length).mean()
    avg_loss = pd.Series(loss, index=close.index).rolling(window=rsi_length).mean()
    rs = np.where(avg_loss != 0, avg_gain / avg_loss, 0)
    rsi = 100 - (100 / (1 + rs))
    rsi_score = (rsi - 50) * 2

    # Intraday EMA score
    ema_f_intra = close.ewm(span=fast_intra, adjust=False).mean()
    ema_s_intra = close.ewm(span=slow_intra, adjust=False).mean()
    md_intra = (ema_f_intra - ema_s_intra) / ema_s_intra * 100
    ma_score_intra = np.clip(md_intra * 5, -100, 100)

    # Daily EMA score (assuming daily data is provided or same timeframe)
    ema_f_day = close.ewm(span=fast_daily, adjust=False).mean()
    ema_s_day = close.ewm(span=slow_daily, adjust=False).mean()
    md_daily = (ema_f_day - ema_s_day) / ema_s_day * 100
    ma_score_daily = np.clip(md_daily * 5, -100, 100)

    # Weekly EMA score (assuming weekly data is provided or same timeframe)
    ema_f_wk = close.ewm(span=fast_week, adjust=False).mean()
    ema_s_wk = close.ewm(span=slow_week, adjust=False).mean()
    md_week = (ema_f_wk - ema_s_wk) / ema_s_wk * 100
    ma_score_week = np.clip(md_week * 5, -100, 100)

    # Composite MA score
    ma_score = (ma_score_intra + ma_score_daily + ma_score_week) / 3

    # Bollinger Bands score
    bb_basis = close.rolling(window=bb_length).mean()
    bb_dev = bb_mult * close.rolling(window=bb_length).std()
    bb_upper = bb_basis + bb_dev
    bb_lower = bb_basis - bb_dev
    bb_position = (close - bb_lower) / (bb_upper - bb_lower) * 100
    bb_score = (bb_position - 50) * 2

    # Relative Strength vs Market
    market_change = (benchmark_close - benchmark_close.shift(1)) / benchmark_close.shift(1) * 100
    stock_change = (close - close.shift(1)) / close.shift(1) * 100
    rel_strength = stock_change - market_change
    rs_score = np.clip(rel_strength * 10, -100, 100)

    trend_score = (rsi_score * 0.25) + (ma_score * 0.35) + (bb_score * 0.20) + (rs_score * 0.20)
    smooth_trend = pd.Series(trend_score, index=close.index).ewm(span=5, adjust=False).mean()

    return smooth_trend