"""Trend-following indicators and filters."""
from .median_kijun_sen import median_kijun_sen
from .mad_trend import mad_trend
from .aad_trend import aad_trend
from .rmsd_trend import rmsd_trend
from .emd_trend import emd_trend
from .cauchy_trend import cauchy_trend
from .median_stdev import median_stdev
from .hma_swing_points import hma_swing_points
from .median_for_loop import median_for_loop
from .market_sentiment_trend_gauge import market_sentiment_trend_gauge
from .dema_vstop import dema_vstop
from .hull_for_loop import hull_for_loop
from .dema_sma_stdev import dema_sma_stdev
from .alma_vii import alma_vii
from .white_noise import white_noise
from .coral_trend import coral_trend
from .deb_supertrend import deb_supertrend
from .boosted_moving_average import boosted_moving_average
from .enhanced_keltner_trend import enhanced_keltner_trend
from .follow_line import follow_line
from .adaptive_price_zone import adaptive_price_zone
from .sine_wma_atr import swma_atr_signals
from .adaptive_trend_flow import adaptive_trend_flow
from .dynamic_score_psar import dynamic_score_psar
from .ema_volatility_channel import ema_volatility_channel
from .sd_zero_lag import sd_zero_lag
from .quartile_for_loop import quartile_for_loop
from .variable_moving_average import variable_moving_average
from .strength_index import strength_index
from .log_ma import logarithmic_moving_average
from .lowess import lowess_trend
from .sma_trend_spectrum import sma_trend_spectrum
from .bears_bulls import bears_bulls
from .rmo import rmo
from .projection_bands import projection_bands
from .trend_trigger_factor import trend_trigger_factor
from .moving_average_trend_sniper import moving_average_trend_sniper
from .pure_price_zone_flow import pure_price_zone_flow
from .continuation_index import continuation_index
from .kama_trend_flip import kama_trend_flip
from .swing_flow import swing_flow
from .kalman_hull_kijun import kalman_hull_kijun
from .polynomial_regression_ma import polynomial_regression_ma

__all__ = [
    "median_kijun_sen",
    "mad_trend",
    "aad_trend",
    "rmsd_trend",
    "emd_trend",
    "cauchy_trend",
    "median_stdev",
    "hma_swing_points",
    "median_for_loop",
    "market_sentiment_trend_gauge",
    "dema_vstop",
    "hull_for_loop",
    "dema_sma_stdev",
    "alma_vii",
    "white_noise",
    "coral_trend",
    "deb_supertrend",
    "boosted_moving_average",
    "enhanced_keltner_trend",
    "follow_line",
    "adaptive_price_zone",
    "swma_atr_signals",
    "adaptive_trend_flow",
    "dynamic_score_psar",
    "ema_volatility_channel",
    "sd_zero_lag",
    "quartile_for_loop",
    "variable_moving_average",
    "strength_index",
    "logarithmic_moving_average",
    "lowess_trend",
    "sma_trend_spectrum",
    "bears_bulls",
    "rmo",
    "projection_bands",
    "trend_trigger_factor",
    "moving_average_trend_sniper",
    "pure_price_zone_flow",
    "continuation_index",
    "kama_trend_flip",
    "swing_flow",
    "kalman_hull_kijun",
    "polynomial_regression_ma",
]
