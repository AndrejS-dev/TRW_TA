import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('ESCI', 'Trigger')
def ehlers_simple_cycle_indicator(high: pd.Series,
                                   low: pd.Series,
                                   alpha: float = 0.07) -> pd.DataFrame:
    """https://www.tradingview.com/script/xQ4mP4kc-Ehlers-Simple-Cycle-Indicator-LazyBear/"""
    src = (high + low) / 2.0
    smooth = (src + 2*src.shift(1) + 2*src.shift(2) + src.shift(3)) / 6.0
    cycle = np.zeros(len(src))

    for i in range(len(src)):
        if i < 7:
            if i >= 2:
                cycle[i] = (src.iloc[i] - 2*src.iloc[i-1] + src.iloc[i-2]) / 4.0
            else:
                cycle[i] = 0.0
        else:
            cycle[i] = ((1 - 0.5*alpha)**2 * (smooth.iloc[i] - 2*smooth.iloc[i-1] + smooth.iloc[i-2]) +
                        2*(1 - alpha)*cycle[i-1] -
                        (1 - alpha)**2 * cycle[i-2])

    cycle_series = pd.Series(cycle, index=src.index)

    trigger = cycle_series.shift(1)

    return pd.DataFrame({
        "ESCI": cycle_series,
        "Trigger": trigger
    })
