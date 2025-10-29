import ccxt
import pandas as pd
from datetime import datetime
from trw_ta import register_outputs

@register_outputs('DataFrame')
def fetch_ohlcv_data(
    symbol: str,
    start_date: str,
    end_date: str,
    timeframe: str = '1d',
    exchange_name: str = 'binance'
) -> pd.DataFrame:
    """
    Fetch historical OHLCV (Open, High, Low, Close, Volume) data for any symbol and exchange using CCXT.

    Parameters
    ----------
    symbol : str
        The trading pair symbol (e.g., 'BTC/USDT', 'ETH/USDT', 'AAPL/USD').
    start_date : str
        Start date in 'YYYY-MM-DD' format.
    end_date : str
        End date in 'YYYY-MM-DD' format.
    timeframe : str, default='1d'
        Candlestick timeframe (e.g., '1h', '4h', '1d', '1w').
    exchange_name : str, default='binance'
        Name of the exchange supported by CCXT (e.g., 'binance', 'kraken', 'coinbasepro').

    Returns
    -------
    pd.DataFrame
        A DataFrame containing columns:
        ['time', 'open', 'high', 'low', 'close', 'volume'] with datetime index.

    Notes
    -----
    - The function automatically handles pagination and fetches all available candles
      between `start_date` and `end_date`.
    - If the exchange’s historical depth is limited, it will stop once no more data is returned.
    - For high-frequency data, consider adding delay or rate-limit handling.

    Example
    -------
    >>> df = fetch_ohlcv_data('ETH/USDT', '2023-01-01', '2023-06-01', timeframe='1h', exchange_name='binance')
    >>> print(df.head())
    """

    # --- Initialize exchange ---
    try:
        exchange_class = getattr(ccxt, exchange_name)
        exchange = exchange_class()
    except AttributeError:
        raise ValueError(f"Exchange '{exchange_name}' is not supported by CCXT.")

    # --- Convert dates to timestamps ---
    since = int(datetime.strptime(start_date, '%Y-%m-%d').timestamp() * 1000)
    end_timestamp = int(datetime.strptime(end_date, '%Y-%m-%d').timestamp() * 1000)

    ohlcv = []

    # --- Paginated data fetching ---
    while since < end_timestamp:
        try:
            data = exchange.fetch_ohlcv(symbol, timeframe, since, limit=1000)
            if not data:
                break

            ohlcv.extend(data)
            since = data[-1][0] + 1  # Move to next batch

        except Exception as e:
            print(f"Error fetching data for {symbol} on {exchange_name}: {e}")
            break

    # --- Convert to DataFrame ---
    if not ohlcv:
        raise ValueError(f"No data returned for {symbol} on {exchange_name}.")

    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['time'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('time', inplace=True)

    # --- Return formatted DataFrame ---
    return df[['open', 'high', 'low', 'close', 'volume']]
