#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Coin Tarama ve Filtreleme Modülü
"""

import logging
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
from queue import Queue

import config

logger = logging.getLogger(__name__)


class CryptoScanner:
    """Kripto para tarama ve filtreleme sınıfı"""

    def __init__(self, api_manager, data_fetcher):
        """
        Tarayıcıyı başlatır

        Args:
            api_manager: API yöneticisi
            data_fetcher: Veri çekme sınıfı
        """
        self.api_manager = api_manager
        self.data_fetcher = data_fetcher

        # Tarama kriterleri
        self.min_volume_usd = config.DEFAULT_SCANNER_PARAMS['min_volume_usd']
        self.min_price_change = config.DEFAULT_SCANNER_PARAMS['min_price_change']
        self.timeframe = config.DEFAULT_SCANNER_PARAMS['timeframe']
        self.max_results = config.DEFAULT_SCANNER_PARAMS['max_results']
        self.exclude_stablecoins = config.DEFAULT_SCANNER_PARAMS['exclude_stablecoins']

        # Tarama durumu
        self.scan_results = []
        self.is_scanning = False
        self.scan_thread = None
        self.result_queue = Queue()
        self.progress_callback = None
        self.result_callback = None
        self.log_callback = None

    def set_parameters(self, **kwargs):
        """
        Tarama parametrelerini ayarlar

        Args:
            **kwargs: Parametre anahtar-değer çiftleri
        """
        if 'min_volume_usd' in kwargs:
            self.min_volume_usd = float(kwargs['min_volume_usd'])

        if 'min_price_change' in kwargs:
            self.min_price_change = float(kwargs['min_price_change'])

        if 'timeframe' in kwargs:
            self.timeframe = kwargs['timeframe']

        if 'max_results' in kwargs:
            self.max_results = int(kwargs['max_results'])

        if 'exclude_stablecoins' in kwargs:
            self.exclude_stablecoins = bool(kwargs['exclude_stablecoins'])

        logger.info(f"Tarama parametreleri güncellendi: "
                    f"Min Hacim=${self.min_volume_usd}, "
                    f"Min Değişim={self.min_price_change}%, "
                    f"Zaman Dilimi={self.timeframe}, "
                    f"Maks Sonuç={self.max_results}")

    def log(self, message, error=False):
        """
        Log mesajı gönderir

        Args:
            message (str): Log mesajı
            error (bool): Hata mı
        """
        if self.log_callback:
            self.log_callback(message, error)
        else:
            if error:
                logger.error(message)
            else:
                logger.info(message)

    def scan_all_markets(self, quote_asset='USDT'):
        """
        Tüm kripto para çiftlerini tarar

        Args:
            quote_asset (str): Quote para birimi (örn. 'USDT')

        Returns:
            list: Filtrelenmiş sonuçlar
        """
        if self.is_scanning:
            self.log("Tarama zaten devam ediyor!", error=True)
            return []

        self.is_scanning = True
        self.scan_results = []
        self.result_queue = Queue()

        self.log(f"Tüm {quote_asset} pazarlarını tarama başlıyor...")

        # Arka plan iş parçacığı oluştur
        self.scan_thread = threading.Thread(
            target=self._scan_worker,
            args=(quote_asset,)
        )
        self.scan_thread.daemon = True
        self.scan_thread.start()

        return []

    def _scan_worker(self, quote_asset):
        """
        Tarama işçisi

        Args:
            quote_asset (str): Quote para birimi
        """
        try:
            # 1. Tüm aktif pazarları al
            self.log(f"Aktif pazarlar alınıyor...")

            all_markets = self.api_manager.get_active_markets(
                quote_asset=quote_asset,
                min_volume=0  # İlk aşamada hacim filtresini uygulamıyoruz
            )

            if not all_markets:
                self.is_scanning = False
                self.log("Pazar bilgileri alınamadı!", error=True)
                return

            self.log(f"Toplam {len(all_markets)} pazar bulundu.")

            # 2. İlk filtreleme: Hacim ve stabilcoin filtreleme
            filtered_markets = []

            for market in all_markets:
                # Stabilcoin kontrolü
                if self.exclude_stablecoins:
                    is_stablecoin = False

                    for excluded in config.EXCLUDED_COINS:
                        if excluded in market['base_asset']:
                            is_stablecoin = True
                            break

                    if is_stablecoin:
                        continue

                # Hacim kontrolü
                if market['volume_24h'] < self.min_volume_usd:
                    continue

                filtered_markets.append(market)

            self.log(f"İlk filtreleme sonrası: {len(filtered_markets)} pazar kaldı.")

            if not filtered_markets:
                self.is_scanning = False
                self.log("Kriterlere uyan pazar bulunamadı!", error=True)
                return

            # İlerleme takibi için
            total_markets = len(filtered_markets)
            processed = 0

            # 3. İlk N coini daha detaylı analiz et
            scan_limit = min(100, total_markets)  # En fazla ilk 100 coini detaylı analiz et
            detailed_results = []

            for i, market in enumerate(filtered_markets[:scan_limit]):
                symbol = market['symbol']

                try:
                    # İlerleme güncelle
                    processed += 1
                    progress = (processed / scan_limit) * 100

                    if self.progress_callback:
                        self.progress_callback(progress)

                    if i % 5 == 0:
                        self.log(f"İlerleme: {progress:.1f}% ({processed}/{scan_limit})")

                    # Son belirlenen zaman dilimindeki verileri al
                    df = self.data_fetcher.fetch_latest_data(symbol, interval=self.timeframe, limit=50)

                    if df is None or len(df) < 30:  # En az 30 veri noktası gerekli
                        continue

                    # Fiyat değişimi hesapla
                    first_price = df.iloc[0]['close']
                    last_price = df.iloc[-1]['close']
                    price_change_pct = ((last_price - first_price) / first_price) * 100

                    # Hacim değişimi
                    first_volume = df.iloc[:10]['volume'].mean()  # İlk 10 mum ortalama hacmi
                    last_volume = df.iloc[-10:]['volume'].mean()  # Son 10 mum ortalama hacmi
                    volume_change_pct = ((last_volume - first_volume) / first_volume) * 100 if first_volume > 0 else 0

                    # Volatilite
                    volatility = df['close'].pct_change().std() * 100

                    # Dikkat çeken örüntüler
                    rsi_signal = False
                    volume_spike = False
                    price_breakout = False

                    # RSI hesapla (basitleştirilmiş)
                    delta = df['close'].diff()
                    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
                    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
                    rs = gain / loss
                    rsi = 100 - (100 / (1 + rs))

                    # Son RSI değeri
                    last_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

                    # RSI sinyali
                    if last_rsi < 30 or last_rsi > 70:
                        rsi_signal = True

                    # Hacim patlaması
                    volume_mean = df['volume'].mean()
                    volume_std = df['volume'].std()

                    if df['volume'].iloc[-1] > volume_mean + 2 * volume_std:
                        volume_spike = True

                    # Fiyat kırılması
                    # Son 20 mumdaki en yüksek ve en düşük değerler
                    recent_high = df['high'].rolling(window=20).max().iloc[-2]
                    recent_low = df['low'].rolling(window=20).min().iloc[-2]

                    if last_price > recent_high or last_price < recent_low:
                        price_breakout = True

                    # Sonucu kaydet
                    result = {
                        'symbol': symbol,
                        'base_asset': market['base_asset'],
                        'quote_asset': market['quote_asset'],
                        'price': last_price,
                        'volume_24h': market['volume_24h'],
                        'price_change_pct': price_change_pct,
                        'volume_change_pct': volume_change_pct,
                        'volatility': volatility,
                        'rsi': last_rsi,
                        'has_rsi_signal': rsi_signal,
                        'has_volume_spike': volume_spike,
                        'has_price_breakout': price_breakout,
                        'interest_score': 0,  # İlgi skoru (daha sonra hesaplanacak)
                        'time': datetime.now()
                    }

                    # İlgi skorunu hesapla (basit bir örnek)
                    interest_score = 0

                    # Fiyat değişimi skoru (pozitif veya negatif büyük değişimler ilgi çekici)
                    interest_score += min(abs(price_change_pct) / 5, 20)

                    # Hacim değişimi skoru
                    interest_score += min(abs(volume_change_pct) / 10, 15)

                    # Volatilite skoru
                    interest_score += min(volatility, 25)

                    # Özel sinyaller
                    if rsi_signal:
                        interest_score += 15

                    if volume_spike:
                        interest_score += 10

                    if price_breakout:
                        interest_score += 20

                    result['interest_score'] = interest_score

                    # Minimum değişim kriterini kontrol et
                    if abs(price_change_pct) >= self.min_price_change:
                        detailed_results.append(result)

                        # Sonucu kuyruğa ekle (canlı güncelleme için)
                        self.result_queue.put(result)

                        # Callback kullan
                        if self.result_callback:
                            self.result_callback(result)

                except Exception as e:
                    self.log(f"{symbol} için veri analizi hatası: {str(e)}", error=True)

            # 4. İlgi skoruna göre sırala
            self.scan_results = sorted(detailed_results, key=lambda x: x['interest_score'], reverse=True)

            # 5. Maksimum sonuç sayısına sınırla
            self.scan_results = self.scan_results[:self.max_results]

            self.log(f"Tarama tamamlandı. {len(self.scan_results)} ilgi çekici coin bulundu.")

            # Tüm sonuçlar için callback çağır
            if self.result_callback:
                for result in self.scan_results:
                    self.result_callback(result)

        except Exception as e:
            self.log(f"Tarama sırasında hata: {str(e)}", error=True)

        finally:
            self.is_scanning = False

    def stop_scanning(self):
        """Taramayı durdurur"""
        if self.is_scanning:
            self.is_scanning = False
            self.log("Tarama durduruldu.")
            return True
        return False

    def get_scan_results(self):
        """
        Tarama sonuçlarını döndürür

        Returns:
            list: Tarama sonuçları
        """
        return self.scan_results

    def analyze_coin(self, symbol, interval):
        """
        Belirli bir coini analiz eder

        Args:
            symbol (str): Coin sembolü
            interval (str): Zaman aralığı

        Returns:
            dict: Analiz sonucu
        """
        self.log(f"{symbol} analiz ediliyor ({interval})...")

        try:
            # Son verileri al
            df = self.data_fetcher.fetch_latest_data(symbol, interval=interval, limit=100)

            if df is None or len(df) < 50:
                self.log(f"{symbol} için yeterli veri yok!", error=True)
                return None

            # Temel bilgiler
            first_price = df.iloc[0]['close']
            last_price = df.iloc[-1]['close']
            price_change_pct = ((last_price - first_price) / first_price) * 100

            # Son 24 saatteki hacim
            ticker = self.api_manager.get_ticker_data(symbol)
            volume_24h = float(ticker['quoteVolume']) if ticker and 'quoteVolume' in ticker else 0

            # Temel teknik göstergeler (basitleştirilmiş)
            # RSI
            delta = df['close'].diff()
            gain = delta.where(delta > 0, 0).rolling(window=14).mean()
            loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            last_rsi = rsi.iloc[-1] if not pd.isna(rsi.iloc[-1]) else 50

            # Hareketli ortalamalar
            ma_20 = df['close'].rolling(window=20).mean().iloc[-1]
            ma_50 = df['close'].rolling(window=50).mean().iloc[-1] if len(df) >= 50 else None

            # MACD (basitleştirilmiş)
            ema_12 = df['close'].ewm(span=12, adjust=False).mean()
            ema_26 = df['close'].ewm(span=26, adjust=False).mean()
            macd_line = ema_12 - ema_26
            signal_line = macd_line.ewm(span=9, adjust=False).mean()

            last_macd = macd_line.iloc[-1]
            last_signal = signal_line.iloc[-1]
            macd_histogram = last_macd - last_signal

            # Bollinger Bantları
            ma_20 = df['close'].rolling(window=20).mean()
            std_20 = df['close'].rolling(window=20).std()

            upper_band = ma_20 + (std_20 * 2)
            lower_band = ma_20 - (std_20 * 2)

            last_upper = upper_band.iloc[-1]
            last_lower = lower_band.iloc[-1]

            # Psikolojik seviyeler (örn. yuvarlak sayılar)
            price_digits = len(str(int(last_price)))
            round_level = 10 ** (price_digits - 1)
            nearest_round = round(last_price / round_level) * round_level
            distance_to_round = abs(last_price - nearest_round) / last_price * 100

            # Destek/direnç seviyeleri (basit yaklaşım)
            highs = df['high'].rolling(window=10).max()
            lows = df['low'].rolling(window=10).min()

            resistance_levels = []
            support_levels = []

            for i in range(10, len(df) - 5):
                # Direnç kontrolü
                if highs.iloc[i] > highs.iloc[i - 5:i].max() and highs.iloc[i] > highs.iloc[i + 1:i + 5].max():
                    resistance_levels.append(highs.iloc[i])

                # Destek kontrolü
                if lows.iloc[i] < lows.iloc[i - 5:i].min() and lows.iloc[i] < lows.iloc[i + 1:i + 5].min():
                    support_levels.append(lows.iloc[i])

            # En yakın destek/direnci bul
            closest_resistance = min(resistance_levels,
                                     key=lambda x: abs(x - last_price)) if resistance_levels else None
            closest_support = min(support_levels, key=lambda x: abs(x - last_price)) if support_levels else None

            # Sonucu oluştur
            analysis_result = {
                'symbol': symbol,
                'interval': interval,
                'price': last_price,
                'volume_24h': volume_24h,
                'price_change_pct': price_change_pct,
                'rsi': last_rsi,
                'ma_20': ma_20,
                'ma_50': ma_50,
                'macd': last_macd,
                'macd_signal': last_signal,
                'macd_histogram': macd_histogram,
                'bb_upper': last_upper,
                'bb_lower': last_lower,
                'closest_resistance': closest_resistance,
                'closest_support': closest_support,
                'nearest_round_level': nearest_round,
                'distance_to_round': distance_to_round,
                'analysis_time': datetime.now()
            }

            # Trend analizi
            if last_price > ma_20:
                trend = "YÜKSELİŞ"
            elif last_price < ma_20:
                trend = "DÜŞÜŞ"
            else:
                trend = "YATAY"

            analysis_result['trend'] = trend

            # Sinyal analizi
            signals = []

            # RSI sinyalleri
            if last_rsi < 30:
                signals.append("AŞIRI SATIM (RSI)")
            elif last_rsi > 70:
                signals.append("AŞIRI ALIM (RSI)")

            # MACD sinyalleri
            if last_macd > last_signal and last_macd > 0:
                signals.append("MACD POZİTİF KESME")
            elif last_macd < last_signal and last_macd < 0:
                signals.append("MACD NEGATİF KESME")

            # Bollinger Bantları sinyalleri
            if last_price > last_upper:
                signals.append("BB ÜST BANT GEÇİŞİ")
            elif last_price < last_lower:
                signals.append("BB ALT BANT GEÇİŞİ")

            # Hareketli Ortalama sinyalleri
            if ma_50 is not None:
                if last_price > ma_50 and df['close'].iloc[-2] <= ma_50:
                    signals.append("MA50 YUKARI KIRILIM")
                elif last_price < ma_50 and df['close'].iloc[-2] >= ma_50:
                    signals.append("MA50 AŞAĞI KIRILIM")

            if last_price > ma_20 and df['close'].iloc[-2] <= ma_20:
                signals.append("MA20 YUKARI KIRILIM")
            elif last_price < ma_20 and df['close'].iloc[-2] >= ma_20:
                signals.append("MA20 AŞAĞI KIRILIM")

            analysis_result['signals'] = signals

            # Anahtar seviyeler
            analysis_result['key_levels'] = {
                'resistance': sorted(resistance_levels[-5:]) if len(resistance_levels) > 5 else resistance_levels,
                'support': sorted(support_levels[-5:]) if len(support_levels) > 5 else support_levels
            }

            self.log(
                f"{symbol} analizi tamamlandı. Trend: {trend}, Sinyaller: {', '.join(signals) if signals else 'Yok'}")

            return analysis_result

        except Exception as e:
            self.log(f"{symbol} analiz edilirken hata: {str(e)}", error=True)
            return None

    def find_correlations(self, base_symbol, num_coins=20):
        """
        Bir coinle korelasyon gösteren diğer coinleri bulur

        Args:
            base_symbol (str): Baz alınacak coin
            num_coins (int): İncelenecek coin sayısı

        Returns:
            dict: Korelasyon sonuçları
        """
        self.log(f"{base_symbol} için korelasyon analizi başlıyor...")

        try:
            # Baz alınacak coinin verilerini al
            base_df = self.data_fetcher.fetch_latest_data(base_symbol, interval='1d', limit=30)

            if base_df is None or len(base_df) < 20:
                self.log(f"{base_symbol} için yeterli veri yok!", error=True)
                return None

            # Baz coinin günlük getirilerini hesapla
            base_returns = base_df['close'].pct_change().dropna()

            # En yüksek hacimli coinleri al
            top_markets = self.api_manager.get_active_markets(
                quote_asset=base_symbol.replace(base_symbol.rstrip('USDT'), ''),
                min_volume=self.min_volume_usd
            )

            if not top_markets:
                self.log("Pazar bilgileri alınamadı!", error=True)
                return None

            # Sınırlı sayıda coin al
            top_markets = top_markets[:num_coins]

            correlations = []

            for market in top_markets:
                symbol = market['symbol']

                if symbol == base_symbol:
                    continue

                try:
                    # Diğer coinin verilerini al
                    coin_df = self.data_fetcher.fetch_latest_data(symbol, interval='1d', limit=30)

                    if coin_df is None or len(coin_df) < 20:
                        continue

                    # Günlük getirileri hesapla
                    coin_returns = coin_df['close'].pct_change().dropna()

                    # Ortak indeksleri al
                    common_idx = base_returns.index.intersection(coin_returns.index)

                    if len(common_idx) < 10:
                        continue

                    # Korelasyon hesapla
                    corr = base_returns.loc[common_idx].corr(coin_returns.loc[common_idx])

                    correlations.append({
                        'symbol': symbol,
                        'base_asset': market['base_asset'],
                        'correlation': corr,
                        'volume_24h': market['volume_24h']
                    })

                except Exception as e:
                    self.log(f"{symbol} için korelasyon hesaplanamadı: {str(e)}", error=True)

            # Korelasyona göre sırala
            correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)

            # Pozitif ve negatif korelasyonları ayır
            positive_corr = [c for c in correlations if c['correlation'] > 0.7]
            negative_corr = [c for c in correlations if c['correlation'] < -0.7]

            self.log(
                f"Korelasyon analizi tamamlandı. {len(positive_corr)} pozitif, {len(negative_corr)} negatif korelasyon bulundu.")

            return {
                'base_symbol': base_symbol,
                'positive_correlations': positive_corr,
                'negative_correlations': negative_corr,
                'all_correlations': correlations
            }

        except Exception as e:
            self.log(f"Korelasyon analizi sırasında hata: {str(e)}", error=True)
            return None

    def get_multi_timeframe_signals(self, symbol, intervals=None):
        """
        Çoklu zaman dilimlerindeki sinyalleri getirir

        Args:
            symbol (str): Coin sembolü
            intervals (list): Zaman dilimleri listesi

        Returns:
            dict: Sinyal sonuçları
        """
        if intervals is None:
            intervals = ['15m', '1h', '4h', '1d']

        self.log(f"{symbol} için çoklu zaman dilimi analizi başlıyor...")

        results = {}
        for interval in intervals:
            results[interval] = self.analyze_coin(symbol, interval)

        # Sinyal özeti oluştur
        buy_signals = 0
        sell_signals = 0
        neutral_signals = 0
        strong_signals = []

        for interval, result in results.items():
            if result is None:
                continue

            signals = result.get('signals', [])

            # Alım sinyalleri
            buy_signal_count = sum(
                1 for s in signals if any(term in s for term in ['AŞIRI SATIM', 'YUKARI KIRILIM', 'POZİTİF']))

            # Satım sinyalleri
            sell_signal_count = sum(
                1 for s in signals if any(term in s for term in ['AŞIRI ALIM', 'AŞAĞI KIRILIM', 'NEGATİF']))

            buy_signals += buy_signal_count
            sell_signals += sell_signal_count

            if buy_signal_count > 0 and buy_signal_count > sell_signal_count:
                strong_signals.append(f"{interval}: ALIM ({buy_signal_count} sinyal)")
            elif sell_signal_count > 0 and sell_signal_count > buy_signal_count:
                strong_signals.append(f"{interval}: SATIM ({sell_signal_count} sinyal)")
            else:
                neutral_signals += 1
                strong_signals.append(f"{interval}: NÖTR")

        signal_summary = {
            'symbol': symbol,
            'buy_signals': buy_signals,
            'sell_signals': sell_signals,
            'neutral_signals': neutral_signals,
            'timeframes': len(intervals),
            'overall_bias': 'ALIM' if buy_signals > sell_signals else 'SATIM' if sell_signals > buy_signals else 'NÖTR',
            'signal_strength': (max(buy_signals, sell_signals) / (buy_signals + sell_signals + neutral_signals)) if (
                                                                                                                                buy_signals + sell_signals + neutral_signals) > 0 else 0,
            'time': datetime.now(),
            'detailed_signals': strong_signals,
            'timeframe_results': results
        }

        self.log(
            f"{symbol} sinyal özeti: {signal_summary['overall_bias']} (Güç: {signal_summary['signal_strength']:.2f})")

        return signal_summary