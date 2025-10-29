# Library helper functions
from .list_of_functions import function_list, register_outputs

# Core functions
from .ta_core import sma, dema, ema, cum, nz, atr, atr1, atr2, wma, highest, lowest, average_day_range, rma, true_range, stdev, ma
from .ta_core import  bollinger_bands, dev, hma, tr, rate_of_change, lsma, median, momentum, smma, tema, mad, aad, rmsd, emd
from .ta_core import  price_volume_trend, alma, keltner, zlema, zlag_dema, zlag_ma, zlag_tema, gaussian_ma, rescale

# Default TradingView indicators
from .technicals_default.alma import alma
from .technicals_default.accumulation_distribution import accumulation_distribution
from .technicals_default.aroon import aroon
from .technicals_default.dmi_adx import dmi, adx
from .technicals_default.awesome_oscillator import awesome_oscillator
from .technicals_default.balance_of_power import balance_of_power
from .technicals_default.bb_trend import BBTrend
from .technicals_default.bb_percent import BB_percent
from .technicals_default.b_bandwidth import bollinger_bandwidth
from .technicals_default.bull_bear_power import bull_bear_power
from .technicals_default.chaikin_money_flow import chaikin_money_flow
from .technicals_default.chaikin_oscillator import chaikin_oscillator
from .technicals_default.chande_momentum_oscillator import chande_momentum_oscillator
from .technicals_default.choppiness_index import choppiness_index
from .technicals_default.commodity_channel_index import commodity_channel_index
from .technicals_default.coppock_curve import coppock_curve
from .technicals_default.correlation_coefficient import correlation_coefficient
from .technicals_default.donchian_channels import donchian_channels
from .technicals_default.detrended_price_oscillator import detrended_price_oscillator
from .technicals_default.elder_force_index import elder_force_index
from .technicals_default.ease_of_movement import ease_of_movement
from .technicals_default.fisher_transform import fisher_transform
from .technicals_default.keltner_channel import keltner_channels
from .technicals_default.historical_volatility import historical_volatility
from .technicals_default.klinger_oscillator import klinger_oscillator
from .technicals_default.know_sure_thing import know_sure_thing
from .technicals_default.money_flow_index import money_flow_index
from .technicals_default.macd import macd
from .technicals_default.parabolic_sar import sar
from .technicals_default.price_oscillator import price_oscillator
from .technicals_default.rank_correlation_index import rank_correlation_index
from .technicals_default.rsi import rsi
from .technicals_default.relative_vigor_index import relative_vigor_index
from .technicals_default.relative_volatility_index import relative_volatility_index
from .technicals_default.smi_ergodic import smi_ergodic
from .technicals_default.stochastic import stochastic
from .technicals_default.stochastic_momentum_index import stochastic_momentum_index
from .technicals_default.stochastic_rsi import stochastic_rsi
from .technicals_default.supertrend import supertrend
from .technicals_default.trend_strength_index import trend_strength_index
from .technicals_default.trix import trix
from .technicals_default.true_strength_index import true_strength_index
from .technicals_default.ultimate_oscillator import ultimate_oscillator
from .technicals_default.vstop import vstop
from .technicals_default.vortex import vortex
from .technicals_default.williams_percent_r import williams_percent_r

# Statistical tools
from .statistics.performance_metrics import total_return, annualized_return, mean_return, volatility, sharpe_ratio, sortino_ratio, information_ratio
from .statistics.performance_metrics import treynor_ratio, calmar_ratio, max_drawdown, average_drawdown, recovery_factor, ulcer_index, hit_ratio
from .statistics.performance_metrics import profit_factor, payoff_ratio, expected_value, alpha, beta, omega_ratio
from .statistics.adf import rolling_adf_test
from .statistics.linear_regression import linear_regression
from .statistics.garch_volatility import garch_volatility
from .statistics.linear_regression_slope import linear_regression_slope
from .statistics.autocorrelation import autocorrelation
from .statistics.kaufman_stress_indicator import kaufman_stress_indicator

# Volume tools
from .volume.fisher_volume_transform import fisher_volume_transform
from .volume.volume_summer import volume_summer
from .volume.volume_trend_swing_points import volume_trend_swing_points
from .volume.volume_z_score import volume_z_score

# Momentum tools
from .momentum.adaptive_sigmoid_zscore import adaptive_sigmoid_zscore
from .momentum.anchored_momentum import anchored_momentum
from .momentum.chandelier_exit_oscillator import chandelier_exit_oscillator
from .momentum.dpmo import dpmo
from .momentum.ehlers_center_of_gravity import ehlers_center_of_gravity
from .momentum.ehlers_instantaneous_trend import ehlers_instantaneous_trend
from .momentum.ehlers_simple_cycle_indicator import ehlers_simple_cycle_indicator
from .momentum.elliot_wave_oscillator import elliot_wave_oscillator
from .momentum.fdi_cumulative_price_momentum import fdi_cumulative_price_momentum
from .momentum.firefly_oscillator import firefly_oscillator
from .momentum.forward_backward_exponential_oscillator import forward_backward_exponential_oscillator
from .momentum.fsvzo import fsvzo
from .momentum.gaussian_mixture_model import gaussian_mixture_model
from .momentum.hurst_cycle_channel_oscillator import hurst_cycle_channel_oscillator
from .momentum.hurst_momentum_oscillator import hurst_momentum_oscillator
from .momentum.momentum_acceleration import momentum_acceleration
from .momentum.normalized_kama_oscillator import normalized_kama_osc
from .momentum.premier_stochastic_oscillator import premier_stochastic_oscillator
from .momentum.pulsema_oscillator import pulsema_oscillator
from .momentum.reflex_oscillator import reflex_oscillator
from .momentum.reversal_scalper import reversal_scalper
from .momentum.rsi_trail import rsi_trail
from .momentum.savitzky_golay_hampel_filter import savitzky_golay_hampel_filter
from .momentum.schaff_trend_cycle import schaff_trend_cycle
from .momentum.squeeze_momentum import squeeze_momentum
from .momentum.stochastic_zscore import stochastic_zscore
from .momentum.trades_in_favor import trades_in_favor
from .momentum.trendflex_oscillator import trendflex_oscillator
from .momentum.trendlines_oscillator import trendlines_oscillator
from .momentum.ttm_squeeze_oscillator import ttm_squeeze_oscillator
from .momentum.vams_oscillator import vams_oscillator
from .momentum.void_momentum_zscore import void_momentum_zscore
from .momentum.wave_trend import wavetrend
from .momentum.weis_wave_candle import weis_wave_candle

# Trend tools
from .trend.median_kijun_sen import median_kijun_sen
from .trend.mad_trend import mad_trend
from .trend.aad_trend import aad_trend
from .trend.rmsd_trend import rmsd_trend
from .trend.emd_trend import emd_trend
from .trend.cauchy_trend import cauchy_trend
from .trend.median_stdev import median_stdev
from .trend.hma_swing_points import hma_swing_points
from .trend.median_for_loop import median_for_loop
from .trend.market_sentiment_trend_gauge import market_sentiment_trend_gauge
from .trend.dema_vstop import dema_vstop
from .trend.hull_for_loop import hull_for_loop
from .trend.dema_sma_stdev import dema_sma_stdev
from .trend.alma_vii import alma_vii
from .trend.white_noise import white_noise
from .trend.coral_trend import coral_trend
from .trend.deb_supertrend import deb_supertrend
from .trend.boosted_moving_average import boosted_moving_average
from .trend.enhanced_keltner_trend import enhanced_keltner_trend
from .trend.follow_line import follow_line
from .trend.adaptive_price_zone import adaptive_price_zone
from .trend.sine_wma_atr import swma_atr_signals
from .trend.adaptive_trend_flow import adaptive_trend_flow
from .trend.dynamic_score_psar import dynamic_score_psar
from .trend.ema_volatility_channel import ema_volatility_channel
from .trend.sd_zero_lag import sd_zero_lag
from .trend.quartile_for_loop import quartile_for_loop
from .trend.variable_moving_average import variable_moving_average
from .trend.strength_index import strength_index
from .trend.log_ma import logarithmic_moving_average
from .trend.lowess import lowess_trend
from .trend.sma_trend_spectrum import sma_trend_spectrum
from .trend.bears_bulls import bears_bulls
from .trend.rmo import rmo
from .trend.projection_bands import projection_bands
from .trend.trend_trigger_factor import trend_trigger_factor
from .trend.moving_average_trend_sniper import moving_average_trend_sniper
from .trend.pure_price_zone_flow import pure_price_zone_flow
from .trend.continuation_index import continuation_index
from .trend.kama_trend_flip import kama_trend_flip

# Mean Reversion tools
from .mean_reversion.quantum_dip_hunter import quantum_dip_hunter

# Utility functions
from .utils.anchored_synthetic_data_generator import generate_anchored_synthetic_prices
from .utils.fetch_ohlcv_data import fetch_ohlcv_data
from .utils.telegram import send_telegram_message

# Machine Learning
from .ml.ml_core import euclidean_distance, manhattan_distance, minkowski_distance, cosine_distance, kl_divergence, js_divergence
from .ml.ml_core import mahalanobis_distance, lorentzian_distance, sigmoid, tanh, relu, leaky_relu, softmax, gelu, swish, mse, mae
from .ml.ml_core import huber_loss, binary_crossentropy, categorical_crossentropy, hinge_loss, minmax_scaling, l2_normalize
from .ml.ml_core import log_scale, linear_kernel, polynomial_kernel, rbf_kernel, laplacian_kernel, sigmoid_kernel, l1_regularization
from .ml.ml_core import l2_regularization, soft_threshold, dropout