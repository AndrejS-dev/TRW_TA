import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

@register_outputs('kst', 'signal')
def know_sure_thing(close: pd.Series,
                  roclen1: int = 10, smalen1: int = 10,
                  roclen2: int = 15, smalen2: int = 10,
                  roclen3: int = 20, smalen3: int = 10,
                  roclen4: int = 30, smalen4: int = 15,
                  siglen: int = 9) -> pd.Series:
    
    def smaroc(roclen, smalen):
        roc = ta.rate_of_change(close, roclen)
        return roc.rolling(window=smalen).mean()

    kst = (
        smaroc(roclen1, smalen1) +
        2 * smaroc(roclen2, smalen2) +
        3 * smaroc(roclen3, smalen3) +
        4 * smaroc(roclen4, smalen4)
    )

    signal = kst.rolling(window=siglen).mean()
    return pd.DataFrame({
        'kst': kst,
        'signal': signal
    })