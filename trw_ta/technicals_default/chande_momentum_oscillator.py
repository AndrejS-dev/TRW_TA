import pandas as pd
import numpy as np
from trw_ta import register_outputs

@register_outputs('cmo')
def chande_momentum_oscillator(source: pd.Series, length: int = 9) -> pd.Series:
    mom = source.diff()

    m1 = np.where(mom >= 0, mom, 0.0)
    m2 = np.where(mom < 0, -mom, 0.0)

    sm1 = pd.Series(m1, index=source.index).rolling(window=length).sum()
    sm2 = pd.Series(m2, index=source.index).rolling(window=length).sum()

    cmo = 100 * (sm1 - sm2) / (sm1 + sm2)
    return cmo