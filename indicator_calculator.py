#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Teknik Gösterge Hesaplama Modülü
"""

import logging
import sys
import numpy as np
import pandas as pd

# TA-Lib veya ta kütüphaneleri (hangisi yüklüyse onu kullan)
try:
    import talib as ta
    from talib import RSI, SMA, EMA, MACD, OBV, BBANDS, CCI, MFI, ROC, ATR, ADX, STDDEV, SAR, WILLR

    HAS_TALIB = True
    logging.info("TA-Lib başarıyla içe aktarıldı.")
except ImportError:
    import ta  # Alternatif kütüphane

    HAS_TALIB = False
    logging.info("Alternatif 'ta' kütüphanesi kullanılıyor.")

import config

logger = logging.getLogger(__name__)


class IndicatorCalculator:
    """Teknik göstergeleri hesaplayan sınıf"""

    def __init__(self):
        """Gösterge hesaplayıcıyı başlatır"""
        self.has_talib = HAS_TALIB
        logger.info(f"Gösterge hesaplayıcı başlatıldı. TA-Lib: {self.has_talib}")

    def calculate_indicators(self, df, indicators=None):
        """
        Verilen göstergeleri hesaplar

        Args:
            df (pd.DataFrame): Veri çerçevesi
            indicators (list): Hesaplanacak göstergeler listesi

        Returns:
            pd.DataFrame: Göstergelerle birlikte veri çerçevesi
        """
        if indicators is None:
            # Varsayılan olarak bazı popüler göstergeleri ekle
            indicators = ['SMA_20', 'EMA_20', 'RSI_14', 'MACD', 'Bollinger_Bands_20_2']

        logger.info(f"Seçilen göstergeler hesaplanıyor: {', '.join(indicators)}")

        # Veri çerçevesini kopyala
        result_df = df.copy()

        # TA-Lib veya ta paketine göre göstergeleri hesapla
        if self.has_talib:
            result_df = self._calculate_indicators_talib(result_df, indicators)
        else:
            result_df = self._calculate_indicators_ta(result_df, indicators)

        # NaN değerleri kaldır
        result_df.dropna(inplace=True)

        # Özel öznitelikleri ekle
        result_df = self.add_custom_features(result_df)

        logger.info(f"{len(indicators)} gösterge hesaplandı. Boyut: {result_df.shape}")
        return result_df

    def _calculate_indicators_talib(self, df, indicators):
        """
        TA-Lib kullanarak göstergeleri hesaplar

        Args:
            df (pd.DataFrame): Veri çerçevesi
            indicators (list): Hesaplanacak göstergeler listesi

        Returns:
            pd.DataFrame: Göstergelerle birlikte veri çerçevesi
        """
        for indicator in indicators:
            if indicator == "SMA_10":
                df['SMA_10'] = ta.SMA(df['close'], timeperiod=10)
            elif indicator == "SMA_20":
                df['SMA_20'] = ta.SMA(df['close'], timeperiod=20)
            elif indicator == "SMA_50":
                df['SMA_50'] = ta.SMA(df['close'], timeperiod=50)
            elif indicator == "SMA_100":
                df['SMA_100'] = ta.SMA(df['close'], timeperiod=100)
            elif indicator == "SMA_200":
                df['SMA_200'] = ta.SMA(df['close'], timeperiod=200)
            elif indicator == "EMA_10":
                df['EMA_10'] = ta.EMA(df['close'], timeperiod=10)
            elif indicator == "EMA_20":
                df['EMA_20'] = ta.EMA(df['close'], timeperiod=20)
            elif indicator == "EMA_50":
                df['EMA_50'] = ta.EMA(df['close'], timeperiod=50)
            elif indicator == "EMA_100":
                df['EMA_100'] = ta.EMA(df['close'], timeperiod=100)
            elif indicator == "EMA_200":
                df['EMA_200'] = ta.EMA(df['close'], timeperiod=200)
            elif indicator == "MACD":
                macd, macdsignal, macdhist = ta.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
                df['MACD'] = macd
                df['MACD_signal'] = macdsignal
                df['MACD_hist'] = macdhist
            elif indicator == "Parabolic_SAR":
                df['SAR'] = ta.SAR(df['high'], df['low'], acceleration=0.02, maximum=0.2)
            elif indicator == "Ichimoku_Cloud":
                high_9 = df['high'].rolling(window=9).max()
                low_9 = df['low'].rolling(window=9).min()
                df['tenkan_sen'] = (high_9 + low_9) / 2

                high_26 = df['high'].rolling(window=26).max()
                low_26 = df['low'].rolling(window=26).min()
                df['kijun_sen'] = (high_26 + low_26) / 2

                df['senkou_span_a'] = ((df['tenkan_sen'] + df['kijun_sen']) / 2).shift(26)
                high_52 = df['high'].rolling(window=52).max()
                low_52 = df['low'].rolling(window=52).min()
                df['senkou_span_b'] = ((high_52 + low_52) / 2).shift(26)
                df['chikou_span'] = df['close'].shift(-26)

            # Momentum göstergeleri
            elif indicator == "RSI_14":
                df['RSI_14'] = ta.RSI(df['close'], timeperiod=14)
            elif indicator == "RSI_7":
                df['RSI_7'] = ta.RSI(df['close'], timeperiod=7)
            elif indicator == "RSI_21":
                df['RSI_21'] = ta.RSI(df['close'], timeperiod=21)
            elif indicator == "Stochastic_14_3_3":
                slowk, slowd = ta.STOCH(df['high'], df['low'], df['close'],
                                        fastk_period=14, slowk_period=3, slowk_matype=0,
                                        slowd_period=3, slowd_matype=0)
                df['STOCH_K'] = slowk
                df['STOCH_D'] = slowd
            elif indicator == "CCI_20":
                df['CCI_20'] = ta.CCI(df['high'], df['low'], df['close'], timeperiod=20)
            elif indicator == "MFI_14":
                df['MFI_14'] = ta.MFI(df['high'], df['low'], df['close'], df['volume'], timeperiod=14)
            elif indicator == "Williams_%R":
                df['WILLR'] = ta.WILLR(df['high'], df['low'], df['close'], timeperiod=14)
            elif indicator == "ROC_10":
                df['ROC_10'] = ta.ROC(df['close'], timeperiod=10)

            # Volatilite göstergeleri
            elif indicator == "Bollinger_Bands_20_2":
                upper, middle, lower = ta.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0)
                df['BB_upper'] = upper
                df['BB_middle'] = middle
                df['BB_lower'] = lower
                df['BB_width'] = (upper - lower) / middle
            elif indicator == "ATR_14":
                df['ATR_14'] = ta.ATR(df['high'], df['low'], df['close'], timeperiod=14)
            elif indicator == "Keltner_Channels_20_2":
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                ema20 = ta.EMA(typical_price, timeperiod=20)
                atr20 = ta.ATR(df['high'], df['low'], df['close'], timeperiod=20)
                df['KC_upper'] = ema20 + 2 * atr20
                df['KC_middle'] = ema20
                df['KC_lower'] = ema20 - 2 * atr20
            elif indicator == "Standard_Deviation_20":
                df['STDDEV_20'] = ta.STDDEV(df['close'], timeperiod=20, nbdev=1)

            # Hacim göstergeleri
            elif indicator == "OBV":
                df['OBV'] = ta.OBV(df['close'], df['volume'])
            elif indicator == "VWAP":
                # VWAP hesaplama (her gün sıfırlanır)
                df['VWAP'] = self._calculate_vwap(df)

            elif indicator == "ADX_14":
                df['ADX_14'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
                df['DI_plus'] = ta.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
                df['DI_minus'] = ta.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)

        return df

    def _calculate_indicators_ta(self, df, indicators):
        """
        ta kütüphanesini kullanarak göstergeleri hesaplar

        Args:
            df (pd.DataFrame): Veri çerçevesi
            indicators (list): Hesaplanacak göstergeler listesi

        Returns:
            pd.DataFrame: Göstergelerle birlikte veri çerçevesi
        """
        for indicator in indicators:
            if indicator == "SMA_10":
                df['SMA_10'] = ta.trend.sma_indicator(df['close'], window=10)
            elif indicator == "SMA_20":
                df['SMA_20'] = ta.trend.sma_indicator(df['close'], window=20)
            elif indicator == "SMA_50":
                df['SMA_50'] = ta.trend.sma_indicator(df['close'], window=50)
            elif indicator == "SMA_100":
                df['SMA_100'] = ta.trend.sma_indicator(df['close'], window=100)
            elif indicator == "SMA_200":
                df['SMA_200'] = ta.trend.sma_indicator(df['close'], window=200)
            elif indicator == "EMA_10":
                df['EMA_10'] = ta.trend.ema_indicator(df['close'], window=10)
            elif indicator == "EMA_20":
                df['EMA_20'] = ta.trend.ema_indicator(df['close'], window=20)
            elif indicator == "EMA_50":
                df['EMA_50'] = ta.trend.ema_indicator(df['close'], window=50)
            elif indicator == "EMA_100":
                df['EMA_100'] = ta.trend.ema_indicator(df['close'], window=100)
            elif indicator == "EMA_200":
                df['EMA_200'] = ta.trend.ema_indicator(df['close'], window=200)
            elif indicator == "MACD":
                macd = ta.trend.MACD(df['close'])
                df['MACD'] = macd.macd()
                df['MACD_signal'] = macd.macd_signal()
                df['MACD_hist'] = macd.macd_diff()
            elif indicator == "RSI_14":
                df['RSI_14'] = ta.momentum.rsi(df['close'], window=14)
            elif indicator == "RSI_7":
                df['RSI_7'] = ta.momentum.rsi(df['close'], window=7)
            elif indicator == "RSI_21":
                df['RSI_21'] = ta.momentum.rsi(df['close'], window=21)
            elif indicator == "Bollinger_Bands_20_2":
                bb = ta.volatility.BollingerBands(df['close'], window=20, window_dev=2)
                df['BB_upper'] = bb.bollinger_hband()
                df['BB_middle'] = bb.bollinger_mavg()
                df['BB_lower'] = bb.bollinger_lband()
                df['BB_width'] = bb.bollinger_wband()
            elif indicator == "ATR_14":
                df['ATR_14'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'], window=14)
            elif indicator == "OBV":
                df['OBV'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            elif indicator == "Stochastic_14_3_3":
                stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'], window=14, smooth_window=3)
                df['STOCH_K'] = stoch.stoch()
                df['STOCH_D'] = stoch.stoch_signal()
            elif indicator == "ADX_14":
                adx_indicator = ta.trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
                df['ADX_14'] = adx_indicator.adx()
                df['DI_plus'] = adx_indicator.adx_pos()
                df['DI_minus'] = adx_indicator.adx_neg()
            elif indicator == "CCI_20":
                df['CCI_20'] = ta.trend.cci(df['high'], df['low'], df['close'], window=20)
            elif indicator == "VWAP":
                # VWAP hesaplama
                df['VWAP'] = self._calculate_vwap(df)

        return df

    def _calculate_vwap(self, df):
        """
        VWAP (Volume Weighted Average Price) hesaplar

        Args:
            df (pd.DataFrame): Veri çerçevesi

        Returns:
            pd.Series: VWAP değerleri
        """
        # Döngüsel olarak günlük hesaplama gerekiyor
        df = df.copy()

        # Tipik fiyat hesapla
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3

        # Her mum için volüm ağırlığını hesapla
        df['tp_volume'] = df['typical_price'] * df['volume']

        # Tarih bilgisini kullan
        df_with_date = df.reset_index()
        df_with_date['date'] = df_with_date['timestamp'].dt.date

        # Her gün için kümülatif değerler hesapla
        result = []
        for date, group in df_with_date.groupby('date'):
            cum_tp_vol = group['tp_volume'].cumsum()
            cum_vol = group['volume'].cumsum()
            vwap = cum_tp_vol / cum_vol
            result.append(pd.Series(vwap.values, index=group.index))

        vwap_series = pd.concat(result)
        vwap_series = vwap_series.sort_index()

        # İndeksi Tekrar ayarla
        vwap_series.index = df.index

        return vwap_series

    def add_custom_features(self, df):
        """
        Özel öznitelikler ekler (fiyat hareketleri, desen tanıma vb.)

        Args:
            df (pd.DataFrame): Veri çerçevesi

        Returns:
            pd.DataFrame: Özel özniteliklerle birlikte veri çerçevesi
        """
        result_df = df.copy()

        # Fiyat değişim yüzdeleri
        result_df['price_pct_change'] = result_df['close'].pct_change() * 100

        # Yüksek ve düşük fiyat farkı (volatilite göstergesi)
        result_df['high_low_diff'] = (result_df['high'] - result_df['low']) / result_df['close'] * 100

        # Fiyat trendleri (SMA20 üzerinde/altında)
        if 'SMA_20' in result_df.columns:
            result_df['above_sma20'] = (result_df['close'] > result_df['SMA_20']).astype(int)

        # RSI bölgeleri
        if 'RSI_14' in result_df.columns:
            result_df['rsi_oversold'] = (result_df['RSI_14'] < 30).astype(int)
            result_df['rsi_overbought'] = (result_df['RSI_14'] > 70).astype(int)

        # MACD sinyalleri
        if all(col in result_df.columns for col in ['MACD', 'MACD_signal']):
            result_df['macd_crossover'] = ((result_df['MACD'] > result_df['MACD_signal']) &
                                           (result_df['MACD'].shift() <= result_df['MACD_signal'].shift())).astype(int)
            result_df['macd_crossunder'] = ((result_df['MACD'] < result_df['MACD_signal']) &
                                            (result_df['MACD'].shift() >= result_df['MACD_signal'].shift())).astype(int)

        # Bollinger Band sinyalleri
        if all(col in result_df.columns for col in ['BB_upper', 'BB_lower']):
            result_df['bb_upper_cross'] = ((result_df['close'] > result_df['BB_upper']) &
                                           (result_df['close'].shift() <= result_df['BB_upper'].shift())).astype(int)
            result_df['bb_lower_cross'] = ((result_df['close'] < result_df['BB_lower']) &
                                           (result_df['close'].shift() >= result_df['BB_lower'].shift())).astype(int)

        # Stokastik sinyalleri
        if all(col in result_df.columns for col in ['STOCH_K', 'STOCH_D']):
            result_df['stoch_crossover'] = ((result_df['STOCH_K'] > result_df['STOCH_D']) &
                                            (result_df['STOCH_K'].shift() <= result_df['STOCH_D'].shift())).astype(int)
            result_df['stoch_crossunder'] = ((result_df['STOCH_K'] < result_df['STOCH_D']) &
                                             (result_df['STOCH_K'].shift() >= result_df['STOCH_D'].shift())).astype(int)

        # Fiyat mumları
        result_df['body_size'] = abs(result_df['close'] - result_df['open']) / result_df['open'] * 100
        result_df['upper_shadow'] = (result_df['high'] - result_df[['open', 'close']].max(axis=1)) / result_df[
            'open'] * 100
        result_df['lower_shadow'] = (result_df[['open', 'close']].min(axis=1) - result_df['low']) / result_df[
            'open'] * 100

        # Mum desenleri (basit örnekler)
        result_df['doji'] = (result_df['body_size'] < 0.1).astype(int)
        result_df['hammer'] = ((result_df['body_size'] > 0) &
                               (result_df['lower_shadow'] > 2 * result_df['body_size']) &
                               (result_df['upper_shadow'] < 0.5)).astype(int)

        return result_df

    def get_recommended_indicators(self, interval):
        """
        Zaman aralığına göre önerilen göstergeleri döndürür

        Args:
            interval (str): Zaman aralığı

        Returns:
            list: Önerilen göstergeler listesi
        """
        if interval in config.TIMEFRAME_RECOMMENDATIONS:
            recommendation = config.TIMEFRAME_RECOMMENDATIONS[interval]
            logger.info(f"Zaman aralığı tavsiyesi: {interval} - {recommendation}")

        if interval in ['1m', '3m', '5m', '15m']:
            return [
                'RSI_7', 'RSI_14', 'EMA_10', 'EMA_20', 'SMA_20',
                'VWAP', 'OBV', 'Bollinger_Bands_20_2'
            ]
        elif interval in ['30m', '1h', '2h']:
            return [
                'RSI_14', 'MACD', 'Bollinger_Bands_20_2', 'EMA_20',
                'EMA_50', 'MFI_14', 'OBV', 'VWAP'
            ]
        elif interval in ['4h', '6h', '8h', '12h']:
            return [
                'RSI_14', 'MACD', 'Bollinger_Bands_20_2', 'EMA_50',
                'SMA_50', 'SMA_100', 'ADX_14', 'ATR_14', 'Ichimoku_Cloud'
            ]
        else:  # '1d', '3d', '1w'
            return [
                'RSI_14', 'MACD', 'Bollinger_Bands_20_2', 'SMA_50',
                'SMA_200', 'ADX_14', 'MFI_14', 'Stochastic_14_3_3',
                'Parabolic_SAR', 'Ichimoku_Cloud'
            ]