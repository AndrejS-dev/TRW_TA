import pandas as pd
from trw_ta import register_outputs

@register_outputs('supertrend', 'direction')
def supertrend(high: pd.Series, low: pd.Series, close: pd.Series, atr_period: int = 10, factor: float = 3.0) -> pd.DataFrame:
    tr1 = high - low
    tr2 = (high - close.shift()).abs()
    tr3 = (low - close.shift()).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/atr_period, adjust=False).mean()

    hl2 = (high + low) / 2
    upperband = hl2 + factor * atr
    lowerband = hl2 - factor * atr

    direction = pd.Series(index=close.index, dtype='int64')
    supertrend = pd.Series(index=close.index, dtype='float64')

    for i in range(len(close)):
        if i == 0:
            direction.iloc[i] = 1
            supertrend.iloc[i] = lowerband.iloc[i]
            continue

        prev_close = close.iloc[i - 1]
        prev_supertrend = supertrend.iloc[i - 1]
        prev_direction = direction.iloc[i - 1]

        if close.iloc[i] > upperband.iloc[i - 1]:
            direction.iloc[i] = 1
        elif close.iloc[i] < lowerband.iloc[i - 1]:
            direction.iloc[i] = -1
        else:
            direction.iloc[i] = prev_direction
            if prev_direction == 1 and lowerband.iloc[i] > prev_supertrend:
                supertrend.iloc[i] = lowerband.iloc[i]
            elif prev_direction == -1 and upperband.iloc[i] < prev_supertrend:
                supertrend.iloc[i] = upperband.iloc[i]
            else:
                supertrend.iloc[i] = prev_supertrend
            continue

        supertrend.iloc[i] = lowerband.iloc[i] if direction.iloc[i] == 1 else upperband.iloc[i]

    return pd.DataFrame({
        'supertrend': supertrend,
        'direction': direction
    })
