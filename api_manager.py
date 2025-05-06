#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binance API Yöneticisi - API bağlantıları ve işlemleri
"""

import time
import hmac
import hashlib
import logging
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException
from urllib.parse import urlencode

import config

logger = logging.getLogger(__name__)


class BinanceAPIManager:
    """Binance API Yöneticisi"""

    def __init__(self, api_key="", api_secret="", testnet=False):
        """
        Binance API Yöneticisi başlatılıyor

        Args:
            api_key (str): Binance API anahtarı
            api_secret (str): Binance API gizli anahtarı
            testnet (bool): TestNet mi kullanılacak
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.client = None
        self.connected = False
        self.market_info = {}

    def connect(self):
        """API bağlantısı kurulur"""
        try:
            # TestNet kontrolü ve URL ayarı
            if self.testnet:
                logger.info("TestNet modunda bağlantı kuruluyor...")
                base_url = config.API_KEYS["testnet"]["base_url"]

                # Binance TestNet için özel ayarlar
                self.client = Client(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    testnet=True,
                    requests_params={'timeout': 10},
                    tld='com',
                    base_url=base_url
                )
            else:
                logger.info("Canlı modda bağlantı kuruluyor...")
                self.client = Client(
                    api_key=self.api_key,
                    api_secret=self.api_secret,
                    requests_params={'timeout': 10}
                )

            # Bağlantı testi
            start_time = time.time()
            self.client.ping()
            latency = (time.time() - start_time) * 1000

            # Market bilgilerini güncelle
            self._update_market_info()

            self.connected = True
            logger.info(f"Binance API bağlantısı başarıyla kuruldu ({latency:.2f}ms). "
                        f"Mod: {'TestNet' if self.testnet else 'Canlı Sunucu'}")
            return True

        except BinanceAPIException as e:
            error_msg = f"Binance API hatası (Kod: {e.status_code}): {e.message}"
            logger.error(error_msg)
        except BinanceRequestException as e:
            logger.error(f"Binance istek hatası: {str(e)}")
        except Exception as e:
            logger.error(f"Beklenmeyen bağlantı hatası: {str(e)}")

        self.connected = False
        return False

    def _update_market_info(self):
        """Market bilgilerini günceller"""
        try:
            exchange_info = self.client.get_exchange_info()

            for symbol_info in exchange_info['symbols']:
                if symbol_info['status'] == 'TRADING':
                    symbol = symbol_info['symbol']
                    self.market_info[symbol] = {
                        'baseAsset': symbol_info['baseAsset'],
                        'quoteAsset': symbol_info['quoteAsset'],
                        'filters': {f['filterType']: f for f in symbol_info['filters']},
                        'precision': {
                            'price': symbol_info['quotePrecision'],
                            'quantity': symbol_info['baseAssetPrecision']
                        },
                        'min_notional': float(next(
                            (f['minNotional'] for f in symbol_info['filters']
                             if f['filterType'] == 'MIN_NOTIONAL'), 0))
                    }

            logger.info("Market bilgileri güncellendi. Toplam %d sembol", len(self.market_info))

        except Exception as e:
            logger.error("Market bilgileri güncellenirken hata: %s", str(e))

    def get_market_info(self, symbol):
        """Belirli bir sembol için market bilgilerini döndürür"""
        if symbol in self.market_info:
            return self.market_info[symbol]
        return None

    def get_historical_klines(self, symbol, interval, start_time=None, end_time=None, limit=1000):
        """
        Tarihsel kline verilerini çeker

        Args:
            symbol (str): İşlem çifti (örn. 'BTCUSDT')
            interval (str): Zaman aralığı (örn. '1h')
            start_time: Başlangıç zamanı (timestamp)
            end_time: Bitiş zamanı (timestamp)
            limit (int): Döndürülecek maksimum veri sayısı

        Returns:
            list: Kline veri listesi
        """
        if not self.connected or not self.client:
            logger.error("API bağlantısı kurulmadı")
            return None

        try:
            klines = self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_time,
                end_str=end_time,
                limit=limit
            )
            logger.info("Toplam %d kline verisi çekildi", len(klines))
            return klines

        except Exception as e:
            logger.error("Kline verileri alınırken hata: %s", str(e))
            return None

    def get_ticker_data(self, symbol=None):
        """
        Anlık fiyat verileri alır

        Args:
            symbol (str): İşlem çifti. None ise tüm semboller

        Returns:
            dict: Fiyat verileri
        """
        if not self.connected or not self.client:
            logger.error("API bağlantısı kurulmadı")
            return None

        try:
            if symbol:
                return self.client.get_ticker(symbol=symbol)
            else:
                return self.client.get_all_tickers()

        except Exception as e:
            logger.error("Ticker verileri alınırken hata: %s", str(e))
            return None

    def get_account_info(self):
        """
        Hesap bilgilerini çeker

        Returns:
            dict: Hesap bilgileri
        """
        if not self.connected or not self.client:
            logger.error("API bağlantısı kurulmadı")
            return None

        try:
            account_info = self.client.get_account()
            logger.info("Hesap bilgileri çekildi")
            return account_info

        except Exception as e:
            logger.error("Hesap bilgileri alınırken hata: %s", str(e))
            return None

    def place_order(self, symbol, side, order_type, quantity=None, price=None, test=True):
        """
        Emir yerleştirir

        Args:
            symbol (str): İşlem çifti (örn. 'BTCUSDT')
            side (str): İşlem yönü ('BUY' veya 'SELL')
            order_type (str): Emir tipi ('LIMIT', 'MARKET', vs.)
            quantity (float): İşlem miktarı
            price (float): Fiyat (LIMIT emirleri için)
            test (bool): Test emri mi

        Returns:
            dict: Emir yanıtı
        """
        if not self.connected or not self.client:
            logger.error("API bağlantısı kurulmadı")
            return None

        try:
            params = {
                'symbol': symbol,
                'side': side,
                'type': order_type
            }

            if quantity:
                params['quantity'] = self._format_number(quantity,
                                                         self.get_market_info(symbol)['precision']['quantity'])

            if price and order_type == 'LIMIT':
                params['price'] = self._format_number(price, self.get_market_info(symbol)['precision']['price'])
                params['timeInForce'] = 'GTC'

            if test:
                result = self.client.create_test_order(**params)
                logger.info("Test emri başarıyla oluşturuldu: %s", params)
            else:
                result = self.client.create_order(**params)
                logger.info("Gerçek emir başarıyla oluşturuldu: %s", params)

            return result

        except Exception as e:
            logger.error("Emir yerleştirirken hata: %s", str(e))
            return None

    def _format_number(self, number, precision):
        """Sayıyı belirli bir hassasiyette formatlar"""
        format_str = f"{{:.{precision}f}}"
        return format_str.format(number)

    def get_order_status(self, symbol, order_id):
        """
        Emir durumunu sorgular

        Args:
            symbol (str): İşlem çifti
            order_id: Emir ID

        Returns:
            dict: Emir durumu
        """
        if not self.connected or not self.client:
            logger.error("API bağlantısı kurulmadı")
            return None

        try:
            order = self.client.get_order(symbol=symbol, orderId=order_id)
            return order

        except Exception as e:
            logger.error("Emir durumu sorgulanırken hata: %s", str(e))
            return None

    def cancel_order(self, symbol, order_id):
        """
        Emri iptal eder

        Args:
            symbol (str): İşlem çifti
            order_id: Emir ID

        Returns:
            dict: İptal yanıtı
        """
        if not self.connected or not self.client:
            logger.error("API bağlantısı kurulmadı")
            return None

        try:
            result = self.client.cancel_order(symbol=symbol, orderId=order_id)
            logger.info("Emir başarıyla iptal edildi: %s, %s", symbol, order_id)
            return result

        except Exception as e:
            logger.error("Emir iptal edilirken hata: %s", str(e))
            return None

    def get_24h_stats(self, symbol=None):
        """
        24 saatlik istatistikleri alır

        Args:
            symbol (str): İşlem çifti. None ise tüm semboller

        Returns:
            dict: 24 saatlik istatistikler
        """
        if not self.connected or not self.client:
            logger.error("API bağlantısı kurulmadı")
            return None

        try:
            if symbol:
                return self.client.get_ticker(symbol=symbol)
            else:
                return self.client.get_ticker()

        except Exception as e:
            logger.error("24 saatlik istatistikler alınırken hata: %s", str(e))
            return None

    def get_balances(self):
        """
        Bakiye bilgilerini alır

        Returns:
            list: Bakiye bilgileri
        """
        if not self.connected or not self.client:
            logger.error("API bağlantısı kurulmadı")
            return None

        try:
            account = self.client.get_account()
            balances = []

            for asset in account['balances']:
                free = float(asset['free'])
                locked = float(asset['locked'])

                if free > 0 or locked > 0:
                    balances.append({
                        'asset': asset['asset'],
                        'free': free,
                        'locked': locked,
                        'total': free + locked
                    })

            return sorted(balances, key=lambda x: x['total'], reverse=True)

        except Exception as e:
            logger.error("Bakiye bilgileri alınırken hata: %s", str(e))
            return None

    def get_active_markets(self, quote_asset='USDT', min_volume=0):
        """
        Aktif pazarları listeler

        Args:
            quote_asset (str): Quote asset (örn. 'USDT')
            min_volume (float): Minimum 24s hacim

        Returns:
            list: Aktif pazar listesi
        """
        if not self.connected or not self.client:
            logger.error("API bağlantısı kurulmadı")
            return None

        try:
            tickers = self.client.get_ticker()
            markets = []

            for ticker in tickers:
                symbol = ticker['symbol']

                # Quote asset kontrolü
                if not symbol.endswith(quote_asset):
                    continue

                # Excluded coin kontrolü
                base_asset = symbol[:-len(quote_asset)]
                skip = False

                for excluded in config.EXCLUDED_COINS:
                    if excluded in base_asset:
                        skip = True
                        break

                if skip:
                    continue

                volume_24h = float(ticker['quoteVolume'])

                if volume_24h >= min_volume:
                    price_change = float(ticker['priceChangePercent'])

                    markets.append({
                        'symbol': symbol,
                        'base_asset': base_asset,
                        'quote_asset': quote_asset,
                        'price': float(ticker['lastPrice']),
                        'price_change_24h': price_change,
                        'volume_24h': volume_24h,
                        'high_24h': float(ticker['highPrice']),
                        'low_24h': float(ticker['lowPrice'])
                    })

            return sorted(markets, key=lambda x: x['volume_24h'], reverse=True)

        except Exception as e:
            logger.error("Aktif pazarlar listelenirken hata: %s", str(e))
            return None