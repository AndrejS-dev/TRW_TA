import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller

def rolling_adf_test(series: pd.Series, lookback=100, max_lag=0, confidence='90%'):
   """https://www.tradingview.com/script/KjD8ByIQ-Augmented-Dickey-Fuller-ADF-mean-reversion-test/"""
   
   conf_map = {'90%': '10%', '95%': '5%', '99%': '1%'}
   if confidence not in conf_map:
       raise ValueError("Confidence level must be one of '90%', '95%', or '99%'.")
   crit_key = conf_map[confidence]


   adf_stats = []
   crit_values = []
   mean_reverting = []


   for i in range(lookback, len(series)):
       window = series[i - lookback:i]
       result = adfuller(window, maxlag=max_lag, regression='c', autolag=None)
       adf_stat = result[0]
       crit_val = result[4][crit_key]
       is_reverting = adf_stat < crit_val


       adf_stats.append(adf_stat)
       crit_values.append(crit_val)
       mean_reverting.append(is_reverting)


   result_df = pd.DataFrame({
       'adf_stat': [np.nan]*lookback + adf_stats,
       'critical_value': [np.nan]*lookback + crit_values,
       'is_mean_reverting': [np.nan]*lookback + mean_reverting
   }, index=series.index)
   return result_df