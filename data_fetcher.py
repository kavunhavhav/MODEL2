#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Veri Çekme ve İşleme Modülü
"""

import os
import time
import logging
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from tqdm import tqdm

import config

logger = logging.getLogger(__name__)


class DataFetcher:
    """Veri çekme ve işleme sınıfı"""

    def __init__(self, api_manager):
        """
        DataFetcher sınıfını başlatır

        Args:
            api_manager: Binance API yöneticisi
        """
        self.api_manager = api_manager
        self.data_cache = {}  # Çekilen verileri önbelleğe almak için

    def fetch_historical_data(self, symbol, interval, lookback_days=None, start_time=None, end_time=None,
                              use_cache=True):
        """
        Tarihsel verileri çeker

        Args:
            symbol (str): İşlem çifti (örn. 'BTCUSDT')
            interval (str): Zaman aralığı (örn. '1h')
            lookback_days (int): Kaç günlük veri çekileceği
            start_time: Başlangıç zamanı (datetime veya timestamp)
            end_time: Bitiş zamanı (datetime veya timestamp)
            use_cache (bool): Önbellek kullanılsın mı

        Returns:
            pd.DataFrame: İşlenmiş veri çerçevesi
        """
        cache_key = f"{symbol}_{interval}_{lookback_days}_{start_time}_{end_time}"

        # Eğer önbellekte varsa ve kullan deniyorsa, önbellekten döndür
        if use_cache and cache_key in self.data_cache:
            logger.info(f"Önbellekten veri kullanılıyor: {cache_key}")
            return self.data_cache[cache_key].copy()

        # Zaman aralıklarını ayarla
        now = datetime.now()

        if not start_time:
            if lookback_days:
                start_time = int((now - timedelta(days=lookback_days)).timestamp() * 1000)
            else:
                start_time = int((now - timedelta(days=config.DEFAULT_DATA_DAYS)).timestamp() * 1000)
        elif isinstance(start_time, datetime):
            start_time = int(start_time.timestamp() * 1000)

        if not end_time:
            end_time = int(now.timestamp() * 1000)
        elif isinstance(end_time, datetime):
            end_time = int(end_time.timestamp() * 1000)

        # Veriyi çek
        logger.info(
            f"{symbol} için {interval} veri çekiliyor... (Başlangıç: {datetime.fromtimestamp(start_time / 1000)})")
        klines = self.api_manager.get_historical_klines(
            symbol=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time
        )

        if not klines or len(klines) == 0:
            logger.error(f"{symbol} için veri çekilemedi!")
            return None

        # DataFrame'e dönüştür
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Veri tiplerini düzelt
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['open'] = df['open'].astype(float)
        df['high'] = df['high'].astype(float)
        df['low'] = df['low'].astype(float)
        df['close'] = df['close'].astype(float)
        df['volume'] = df['volume'].astype(float)

        # Gereksiz sütunları kaldır
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
        df.set_index('timestamp', inplace=True)

        # Önbelleğe ekle
        self.data_cache[cache_key] = df.copy()

        logger.info(f"Toplam {len(df)} veri noktası çekildi: {symbol}_{interval}")
        return df

    def fetch_latest_data(self, symbol, interval, limit=100):
        """
        Geliştirilmiş veri çekme metodu - Sembol doğrulama ve hata yönetimi eklendi

        Args:
            symbol (str): İşlem çifti (TRUMPUSDT gibi)
            interval (str): Zaman aralığı (5m, 1h vb.)
            limit (int): Kayıt sayısı

        Returns:
            pd.DataFrame: OHLCV verileri veya None (hata durumunda)
        """
        try:
            # Sembol doğrulama
            if not self._validate_symbol(symbol):
                raise ValueError(f"Geçersiz sembol formatı: {symbol}")

            # API'den ham veriyi al
            raw_data = self.api_manager.get_historical_klines(
                symbol=self._format_symbol(symbol),
                interval=interval,
                limit=limit
            )

            if not raw_data:
                logger.error(f"{symbol} için veri alınamadı")
                return None

            # DataFrame'e dönüştür
            df = pd.DataFrame(raw_data, columns=[
                'timestamp', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_asset_volume', 'number_of_trades',
                'taker_buy_base', 'taker_buy_quote', 'ignore'
            ])

            # Sütun tiplerini dönüştür
            numeric_cols = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)

            # Teknik göstergeleri hesapla
            df = self._calculate_technical_indicators(df)

            return df

        except Exception as e:
            logger.error(f"Veri çekme hatası ({symbol} {interval}): {str(e)}")
            return None

    def _validate_symbol(self, symbol):
        """Sembol adını doğrular"""
        import re
        pattern = r'^[A-Z0-9-_.]{1,20}$'
        return re.match(pattern, symbol) is not None

    def _format_symbol(self, symbol):
        """Sembol adını API'nin beklediği formata dönüştürür"""
        return symbol.upper().replace('-', '').replace('_', '')

    def _calculate_technical_indicators(self, df):
        """Teknik göstergeleri hesaplar"""
        try:
            # SMA/EMA
            df['SMA_20'] = df['close'].rolling(20).mean()
            df['EMA_20'] = df['close'].ewm(span=20, adjust=False).mean()

            # MACD
            ema12 = df['close'].ewm(span=12, adjust=False).mean()
            ema26 = df['close'].ewm(span=26, adjust=False).mean()
            df['MACD'] = ema12 - ema26
            df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()

            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))

            # Bollinger Bands
            df['BB_upper'] = df['SMA_20'] + 2 * df['close'].rolling(20).std()
            df['BB_lower'] = df['SMA_20'] - 2 * df['close'].rolling(20).std()

            # Mum kalıpları
            df['body_size'] = (df['close'] - df['open']).abs() / df['open']
            df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
            df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']
            df['doji'] = ((df['body_size'] < 0.01) &
                          (df['upper_shadow'] > 0.1) &
                          (df['lower_shadow'] > 0.1)).astype(int)
            df['hammer'] = ((df['lower_shadow'] > 2 * df['body_size']) &
                            (df['upper_shadow'] < 0.1 * df['body_size'])).astype(int)

            return df

        except Exception as e:
            logger.error(f"Gösterge hesaplama hatası: {str(e)}")
            return df

    def save_data_to_csv(self, df, symbol, interval, base_dir=None):
        """
        Veri çerçevesini CSV olarak kaydeder

        Args:
            df (pd.DataFrame): Kaydedilecek veri çerçevesi
            symbol (str): İşlem çifti
            interval (str): Zaman aralığı
            base_dir (str): Kaydedilecek temel dizin

        Returns:
            str: Kaydedilen dosya yolu
        """
        if base_dir is None:
            base_dir = config.PATHS['data_dir']

        os.makedirs(base_dir, exist_ok=True)

        filename = f"{symbol.lower()}_{interval}_data.csv"
        filepath = os.path.join(base_dir, filename)

        # DataFrame'i resetle ve kaydet
        df_to_save = df.reset_index()
        df_to_save.to_csv(filepath, index=False)

        logger.info(f"Veri kaydedildi: {filepath}")
        return filepath

    def load_data_from_csv(self, symbol, interval, base_dir=None):
        """
        CSV'den veri yükler

        Args:
            symbol (str): İşlem çifti
            interval (str): Zaman aralığı
            base_dir (str): Yüklenecek temel dizin

        Returns:
            pd.DataFrame: Yüklenen veri çerçevesi
        """
        if base_dir is None:
            base_dir = config.PATHS['data_dir']

        filename = f"{symbol.lower()}_{interval}_data.csv"
        filepath = os.path.join(base_dir, filename)

        if not os.path.exists(filepath):
            logger.error(f"Dosya bulunamadı: {filepath}")
            return None

        try:
            df = pd.read_csv(filepath)

            # Timestamp sütununu datetime'a çevir
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)

            logger.info(f"Veri yüklendi: {filepath}")
            return df

        except Exception as e:
            logger.error(f"Veri yüklenirken hata: {str(e)}")
            return None

    def update_data(self, symbol, interval, df=None, base_dir=None):
        """
        Mevcut verileri güncelleyerek son verileri ekler

        Args:
            symbol (str): İşlem çifti
            interval (str): Zaman aralığı
            df (pd.DataFrame): Mevcut veri çerçevesi (None ise dosyadan yüklenir)
            base_dir (str): Veri dosyası dizini

        Returns:
            pd.DataFrame: Güncellenmiş veri çerçevesi
        """
        if df is None:
            df = self.load_data_from_csv(symbol, interval, base_dir)

        if df is None or len(df) == 0:
            logger.warning(f"{symbol}_{interval} için mevcut veri yok, sıfırdan çekiliyor...")
            return self.fetch_historical_data(symbol, interval)

        # Son verinin zamanını al
        last_time = df.index[-1]

        # Son veriden sonrasını çek (biraz örtüşme olmasını sağla)
        safety_margin = pd.Timedelta(hours=24)  # 24 saat güvenlik marjı
        start_time = (last_time - safety_margin).timestamp() * 1000

        logger.info(f"{symbol}_{interval} için güncelleme yapılıyor... (Son veri: {last_time})")

        new_df = self.fetch_historical_data(
            symbol=symbol,
            interval=interval,
            start_time=int(start_time),
            use_cache=False
        )

        if new_df is None or len(new_df) == 0:
            logger.warning(f"{symbol}_{interval} için yeni veri çekilemedi.")
            return df

        # Eski ve yeni verileri birleştir
        combined_df = pd.concat([df, new_df])

        # Yinelenen verileri temizle
        combined_df = combined_df[~combined_df.index.duplicated(keep='last')]

        # Zamansal sıralama yap
        combined_df.sort_index(inplace=True)

        # Güncellenmiş verileri kaydet
        self.save_data_to_csv(combined_df, symbol, interval, base_dir)

        logger.info(f"{symbol}_{interval} verileri güncellendi. Toplam veri: {len(combined_df)}")
        return combined_df

    def check_data_freshness(self, df, max_age_hours=1):
        """
        Verilerin güncelliğini kontrol eder

        Args:
            df (pd.DataFrame): Kontrol edilecek veri çerçevesi
            max_age_hours (int): Maksimum kabul edilebilir yaş (saat)

        Returns:
            bool: Veriler güncel mi
        """
        if df is None or len(df) == 0:
            return False

        last_time = df.index[-1]
        now = pd.Timestamp.now(tz=last_time.tz)
        age = now - last_time

        if age > pd.Timedelta(hours=max_age_hours):
            logger.warning(f"Veriler güncel değil! Son veri: {last_time}, Yaş: {age}")
            return False

        logger.info(f"Veriler güncel. Son veri: {last_time}, Yaş: {age}")
        return True

    def get_realtime_data(self, symbol):
        """
        Anlık fiyat verisini alır

        Args:
            symbol (str): İşlem çifti

        Returns:
            dict: Anlık fiyat verisi
        """
        try:
            ticker = self.api_manager.get_ticker_data(symbol)

            if ticker:
                return {
                    'symbol': symbol,
                    'price': float(ticker['lastPrice']),
                    'bid': float(ticker['bidPrice']),
                    'ask': float(ticker['askPrice']),
                    'volume': float(ticker['volume']),
                    'timestamp': datetime.now()
                }

            return None

        except Exception as e:
            logger.error(f"Anlık veri alınırken hata: {str(e)}")
            return None

    def prepare_multi_timeframe_data(self, symbol, intervals=None):
        """
        Çoklu zaman dilimli veri hazırlar

        Args:
            symbol (str): İşlem çifti
            intervals (list): Zaman aralıkları listesi

        Returns:
            dict: Zaman aralıklarına göre veri çerçeveleri
        """
        if intervals is None:
            intervals = ['1h', '4h', '1d']

        result = {}

        for interval in intervals:
            df = self.fetch_historical_data(symbol, interval)

            if df is not None and len(df) > 0:
                result[interval] = df

        return result