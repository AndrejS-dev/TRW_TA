import numpy as np
import pandas as pd
import trw_ta as ta
from trw_ta import register_outputs

def DFT(x, y, Nx, direction):
    xArr = np.zeros(Nx)
    yArr = np.zeros(Nx)
    
    for i in range(Nx):
        xArr_i = 0.0
        yArr_i = 0.0
        kx = i / Nx
        arg = -direction * 2 * np.pi * kx
        for k in range(Nx):
            cos_val = np.cos(k * arg)
            sin_val = np.sin(k * arg)
            xArr_i += x[k] * cos_val - y[k] * sin_val
            yArr_i += x[k] * sin_val + y[k] * cos_val
        xArr[i] = xArr_i
        yArr[i] = yArr_i
    
    if direction == 1:
        xArr /= Nx
        yArr /= Nx
    
    return xArr, yArr

def calculate_dft(close, N):
    x = np.zeros(N)
    y = np.zeros(N)
    for i in range(N):
        if i < len(close):
            x[i] = close.iloc[-N+i] if i > 0 else close.iloc[-1]
        y[i] = 0.0
    
    x_dft, y_dft = DFT(x, y, N, 1)
    mag = np.sqrt(x_dft**2 + y_dft**2)
    return mag[0]

@register_outputs('vzo','prev_vzo')
def fsvzo(close: pd.Series, volume: pd.Series, N: int = 3, lengthV: int = 7, 
                  smoothvzo: bool = True, emaLength_init: int = 1, 
                  minEmaLength: int = 1, maxEmaLength: int = 8) -> pd.DataFrame:
    """https://www.tradingview.com/script/N1QqqFtc-FSVZO/"""
    df = pd.DataFrame({'close': close, 'volume': volume})
    df['dft'] = np.nan

    for i in range(N, len(df)):
        close_window = df['close'].iloc[max(0, i-N):i]
        df.loc[df.index[i], 'dft'] = calculate_dft(close_window, N)

    df['dft_diff'] = df['dft'].diff(2)
    df['sign'] = np.sign(df['dft_diff'])
    df['dvol'] = df['sign'] * df['volume']
    df['dvma'] = df['dvol'].rolling(window=lengthV).mean()
    df['vma'] = df['volume'].rolling(window=lengthV).mean()
    df['VZO'] = 100 * df['dvma'] / df['vma'].replace(0, np.nan)

    df['uniL'] = df['VZO'] > 0
    df['vzoL'] = df['VZO'] > df['VZO'].shift(1)

    df['emaLength'] = emaLength_init
    for i in range(1, len(df)):
        if (df['uniL'].iloc[i] and df['vzoL'].iloc[i]) or (not df['uniL'].iloc[i] and not df['vzoL'].iloc[i]):
            df.loc[df.index[i], 'emaLength'] = min(df['emaLength'].iloc[i-1] + 1, maxEmaLength)
        elif (df['uniL'].iloc[i] and not df['vzoL'].iloc[i]) or (not df['uniL'].iloc[i] and df['vzoL'].iloc[i]):
            df.loc[df.index[i], 'emaLength'] = max(df['emaLength'].iloc[i-1] - 1, minEmaLength)
        if df['emaLength'].iloc[i] > maxEmaLength or df['emaLength'].iloc[i] < minEmaLength:
            df.loc[df.index[i], 'emaLength'] = 7

    if smoothvzo:
        df['VZO_smoothed'] = np.nan
        for i in range(len(df)):
            period = int(df['emaLength'].iloc[i])
            df.loc[df.index[i], 'VZO_smoothed'] = ta.ema(df['VZO'][:i+1], period).iloc[-1]
        df['VZO'] = df['VZO_smoothed']

    df['VZO_prev'] = df['VZO'].shift(1)
    
    return df[['VZO', 'VZO_prev']]
