# live_data_fetcher.py
# Binance'ten gerçek zamanlı veri çekmek için kullanılacak

import requests
import pandas as pd
from datetime import datetime

BINANCE_BASE_URL = "https://api.binance.com"


def get_latest_ohlcv(symbol: str, interval: str = "5m", limit: int = 60) -> pd.DataFrame:
    """
    Binance'ten belirli bir sembol ve zaman aralığı için son OHLCV verilerini alır

    Args:
        symbol (str): Örnek: 'BTCUSDT'
        interval (str): Örnek: '1m', '5m', '15m', '1h', '1d'
        limit (int): Alınacak mum sayısı (maks: 1000)

    Returns:
        pd.DataFrame: OHLCV verisi
    """
    url = f"{BINANCE_BASE_URL}/api/v3/klines"
    params = {
        "symbol": symbol.upper(),
        "interval": interval,
        "limit": limit
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not isinstance(data, list) or len(data) == 0:
            raise ValueError("API boş veya geçersiz veri döndürdü.")

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base_volume", "taker_buy_quote_volume", "ignore"
        ])

        df = df[["open_time", "open", "high", "low", "close", "volume"]].copy()
        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)

        df = df.astype(float)
        return df

    except Exception as e:
        print(f"❌ Binance veri alma hatası: {e}")
        return pd.DataFrame()


# Test için örnek kullanım
if __name__ == "__main__":
    df = get_latest_ohlcv("BTCUSDT", interval="5m", limit=20)
    print(df.tail())
