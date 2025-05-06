#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
LSTM Model Yöneticisi
"""

import os
import time
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam

import config

logger = logging.getLogger(__name__)


class ModelManager:
    """LSTM Model Yönetim Sınıfı"""

    def __init__(self, symbol=None, interval=None):
        """
        Model yöneticisini başlatır

        Args:
            symbol (str): İşlem çifti (örn. 'BTCUSDT')
            interval (str): Zaman aralığı (örn. '1h')
        """
        self.symbol = symbol
        self.interval = interval
        self.model = None
        self.scaler = None
        self.lookback = config.DEFAULT_MODEL_PARAMS['lookback']
        self.feature_columns = []
        self.target_column = 'close'
        self.model_info = {}
        self.is_trained = False
        self.train_size = config.DEFAULT_MODEL_PARAMS['train_size']

        # İlgili dizinleri oluştur
        os.makedirs(config.PATHS['models_dir'], exist_ok=True)

    def prepare_data(self, df, feature_columns=None, target_column='close'):
        """
        Veriyi model için hazırlar

        Args:
            df (pd.DataFrame): Hazırlanacak veri
            feature_columns (list): Kullanılacak özellik sütunları
            target_column (str): Hedef sütun

        Returns:
            tuple: X_train, y_train, X_test, y_test, scaler
        """
        logger.info("Veri ön işleme gerçekleştiriliyor...")

        if feature_columns is None:
            feature_columns = ['close', 'high', 'low', 'open', 'volume']

            # Eğer varsa teknik göstergeleri ekle
            for col in df.columns:
                if col not in ['timestamp', 'close', 'high', 'low', 'open', 'volume']:
                    feature_columns.append(col)

        # Feature sütunlarını kaydet
        self.feature_columns = feature_columns
        self.target_column = target_column

        # Hedef sütunu shift ederek gelecek değeri tahmin etme
        df_shifted = df.copy()
        df_shifted['target'] = df_shifted[target_column].shift(-1)
        df_shifted.dropna(inplace=True)

        # Train/test ayrımı
        train_size = int(len(df_shifted) * self.train_size)
        train = df_shifted.iloc[:train_size]
        test = df_shifted.iloc[train_size:]

        logger.info(f"Eğitim seti: {len(train)} kayıt, Test seti: {len(test)} kayıt")

        # Verileri normalize et
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        train_scaled = self.scaler.fit_transform(train[feature_columns + ['target']])
        test_scaled = self.scaler.transform(test[feature_columns + ['target']])

        X_train = train_scaled[:, :-1]  # Feature'lar
        y_train = train_scaled[:, -1]  # Hedef değer
        X_test = test_scaled[:, :-1]
        y_test = test_scaled[:, -1]

        # Verileri LSTM için yeniden şekillendir
        X_train_reshaped = []
        y_train_reshaped = []
        for i in range(self.lookback, len(X_train)):
            X_train_reshaped.append(X_train[i - self.lookback:i, :])
            y_train_reshaped.append(y_train[i])

        X_test_reshaped = []
        y_test_reshaped = []
        for i in range(self.lookback, len(X_test)):
            X_test_reshaped.append(X_test[i - self.lookback:i, :])
            y_test_reshaped.append(y_test[i])

        # Numpy array'lerine çevir
        X_train_np = np.array(X_train_reshaped)
        y_train_np = np.array(y_train_reshaped)
        X_test_np = np.array(X_test_reshaped)
        y_test_np = np.array(y_test_reshaped)

        logger.info(f"Eğitim seti boyutu: {X_train_np.shape}, Test seti boyutu: {X_test_np.shape}")

        return X_train_np, y_train_np, X_test_np, y_test_np, self.scaler

    def build_model(self, input_shape, lstm_units_1=None, lstm_units_2=None, dropout_rate=None, learning_rate=None):
        """
        LSTM modelini oluşturur

        Args:
            input_shape (tuple): Giriş verisi şekli (lookback, features)
            lstm_units_1 (int): İlk LSTM katmanındaki birim sayısı
            lstm_units_2 (int): İkinci LSTM katmanındaki birim sayısı
            dropout_rate (float): Dropout oranı
            learning_rate (float): Öğrenme oranı

        Returns:
            model: Oluşturulan Keras modeli
        """
        # Parametreler belirtilmemişse varsayılanları kullan
        if lstm_units_1 is None:
            lstm_units_1 = config.DEFAULT_MODEL_PARAMS['lstm_units_1']
        if lstm_units_2 is None:
            lstm_units_2 = config.DEFAULT_MODEL_PARAMS['lstm_units_2']
        if dropout_rate is None:
            dropout_rate = config.DEFAULT_MODEL_PARAMS['dropout_rate']
        if learning_rate is None:
            learning_rate = config.DEFAULT_MODEL_PARAMS['learning_rate']

        logger.info(f"LSTM modeli oluşturuluyor... (units={lstm_units_1},{lstm_units_2}, dropout={dropout_rate})")

        model = Sequential()

        # İlk LSTM katmanı (return_sequences=True ile sonraki LSTM katmanına bağlanır)
        model.add(LSTM(units=lstm_units_1, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(dropout_rate))

        # İkinci LSTM katmanı
        model.add(LSTM(units=lstm_units_2))
        model.add(Dropout(dropout_rate))

        # Çıkış katmanı (tek nöron - regresyon problemi)
        model.add(Dense(units=1))

        # Model derleme
        model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mse')

        # Model özetini logla
        model.summary(print_fn=lambda x: logger.info(x))

        self.model = model
        return model

    def train_model(self, X_train, y_train, epochs=None, batch_size=None, validation_split=0.1, early_stopping=True,
                    progress_callback=None):
        """
        Modeli eğitir

        Args:
            X_train (np.array): Eğitim verileri
            y_train (np.array): Eğitim hedefleri
            epochs (int): Epoch sayısı
            batch_size (int): Batch size
            validation_split (float): Doğrulama seti oranı
            early_stopping (bool): Erken durma kullanılsın mı
            progress_callback (function): İlerleme geri çağırma fonksiyonu

        Returns:
            history: Eğitim geçmişi
        """
        if self.model is None:
            logger.error("Eğitim öncesi model oluşturulmalı!")
            return None

        # Parametreler belirtilmemişse varsayılanları kullan
        if epochs is None:
            epochs = config.DEFAULT_MODEL_PARAMS['epochs']
        if batch_size is None:
            batch_size = config.DEFAULT_MODEL_PARAMS['batch_size']

        logger.info(f"Model eğitiliyor (epoch={epochs}, batch_size={batch_size})...")

        # Geçici checkpoint dosyası
        temp_checkpoint_path = os.path.join(config.PATHS['models_dir'],
                                            f'temp_{self.symbol}_{self.interval}_checkpoint.h5')

        # Callbacks
        callbacks = []

        if early_stopping:
            callbacks.append(EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True))

        callbacks.append(ModelCheckpoint(temp_checkpoint_path, save_best_only=True))

        # İlerleme callback'i
        if progress_callback:
            class ProgressCallback(tf.keras.callbacks.Callback):
                def __init__(self, progress_callback):
                    self.progress_callback = progress_callback

                def on_epoch_end(self, epoch, logs=None):
                    if logs is not None and self.progress_callback is not None:
                        self.progress_callback(
                            epoch + 1,
                            self.params['epochs'],
                            logs.get('loss', 0),
                            logs.get('val_loss', 0)
                        )

            callbacks.append(ProgressCallback(progress_callback))

        # Eğitim
        start_time = time.time()

        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callbacks,
            verbose=1 if progress_callback is None else 0
        )

        # Eğitim süresi
        training_time = time.time() - start_time

        # Model bilgilerini kaydet
        self.model_info = {
            'symbol': self.symbol,
            'interval': self.interval,
            'features': self.feature_columns,
            'lookback': self.lookback,
            'train_size': self.train_size,
            'lstm_units': [config.DEFAULT_MODEL_PARAMS['lstm_units_1'], config.DEFAULT_MODEL_PARAMS['lstm_units_2']],
            'dropout_rate': config.DEFAULT_MODEL_PARAMS['dropout_rate'],
            'learning_rate': config.DEFAULT_MODEL_PARAMS['learning_rate'],
            'epochs': epochs,
            'batch_size': batch_size,
            'training_time': training_time,
            'train_samples': len(X_train),
            'trained_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'loss': float(history.history['loss'][-1]),
            'val_loss': float(history.history['val_loss'][-1])
        }

        self.is_trained = True
        logger.info(f"Model eğitimi tamamlandı. Süre: {training_time:.2f} saniye")

        return history

    def evaluate_model(self, X_test, y_test, denormalize=True):
        """
        Modeli değerlendirir

        Args:
            X_test (np.array): Test verileri
            y_test (np.array): Test hedefleri
            denormalize (bool): Tahminler denormalize edilsin mi

        Returns:
            dict: Değerlendirme sonuçları
        """
        if not self.is_trained or self.model is None:
            logger.error("Değerlendirme için eğitilmiş bir model gerekli!")
            return None

        logger.info("Model değerlendiriliyor...")

        # Tahminleri al
        y_pred = self.model.predict(X_test)

        # Denormalize et
        if denormalize and self.scaler is not None:
            # Boş bir array oluştur ve sadece hedef sütunu doldur
            n_features = len(self.feature_columns)
            dummy_array = np.zeros((len(y_test), n_features + 1))

            # Tahminleri son sütuna yerleştir
            dummy_array[:, -1] = y_pred.flatten()
            y_pred_denorm = self.scaler.inverse_transform(dummy_array)[:, -1]

            # Gerçek değerleri de denormalize et
            dummy_array[:, -1] = y_test
            y_test_denorm = self.scaler.inverse_transform(dummy_array)[:, -1]
        else:
            y_pred_denorm = y_pred.flatten()
            y_test_denorm = y_test

        # Metrikler
        mse = mean_squared_error(y_test_denorm, y_pred_denorm)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test_denorm, y_pred_denorm)
        r2 = r2_score(y_test_denorm, y_pred_denorm)

        # Yön tahmini başarısı
        actual_direction = np.sign(np.diff(y_test_denorm, prepend=y_test_denorm[0]))
        predicted_direction = np.sign(np.diff(y_pred_denorm, prepend=y_pred_denorm[0]))

        direction_accuracy = (actual_direction == predicted_direction).mean() * 100

        # Sonuçları kaydet
        results = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'y_test': y_test_denorm,
            'y_pred': y_pred_denorm
        }

        # Sonuçları logla
        logger.info(f"MSE: {mse:.6f}")
        logger.info(f"RMSE: {rmse:.6f}")
        logger.info(f"MAE: {mae:.6f}")
        logger.info(f"R²: {r2:.6f}")
        logger.info(f"Yön Tahmini Doğruluğu: {direction_accuracy:.2f}%")

        # Model bilgilerini güncelle
        self.model_info.update({
            'evaluation': {
                'mse': float(mse),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2),
                'direction_accuracy': float(direction_accuracy)
            }
        })

        return results

    def save_model(self, custom_name=None):
        """
        Modeli ve meta verileri kaydeder

        Args:
            custom_name (str): Özel dosya adı

        Returns:
            str: Model dosya yolu
        """
        if not self.is_trained or self.model is None:
            logger.error("Kaydedilecek eğitilmiş bir model yok!")
            return None

        # Dosya adını oluştur
        if custom_name:
            filename = f"{custom_name}.h5"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.symbol}_{self.interval}_{timestamp}.h5"

        # Tam dosya yolu
        model_path = os.path.join(config.PATHS['models_dir'], filename)

        # Modeli kaydet
        self.model.save(model_path)

        # Meta verileri JSON olarak kaydet
        meta_filename = os.path.splitext(filename)[0] + "_meta.json"
        meta_path = os.path.join(config.PATHS['models_dir'], meta_filename)

        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(self.model_info, f, indent=4, ensure_ascii=False)

        logger.info(f"Model kaydedildi: {model_path}")
        logger.info(f"Model meta verileri kaydedildi: {meta_path}")

        return model_path

    def load_model(self, model_path):
        """
        Kaydedilmiş modeli yükler

        Args:
            model_path (str): Model dosya yolu

        Returns:
            bool: Başarılı mı
        """
        if not os.path.exists(model_path):
            logger.error(f"Model dosyası bulunamadı: {model_path}")
            return False

        try:
            # Modeli yükle
            self.model = load_model(model_path)

            # Meta verileri yükle
            meta_path = os.path.splitext(model_path)[0] + "_meta.json"

            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    self.model_info = json.load(f)

                # Meta verilerden gerekli bilgileri çıkar
                self.symbol = self.model_info.get('symbol')
                self.interval = self.model_info.get('interval')
                self.feature_columns = self.model_info.get('features', [])
                self.lookback = self.model_info.get('lookback', self.lookback)

            self.is_trained = True
            logger.info(f"Model yüklendi: {model_path}")

            if self.model_info:
                logger.info(f"Model bilgileri: {self.symbol} {self.interval}, "
                            f"Eğitim tarihi: {self.model_info.get('trained_date')}")

            return True

        except Exception as e:
            logger.error(f"Model yüklenirken hata: {str(e)}")
            return False

    def predict_next(self, sequence, denormalize=True):
        """
        Sonraki değeri tahmin eder

        Args:
            sequence (np.array): Girdi dizisi
            denormalize (bool): Tahmin denormalize edilsin mi

        Returns:
            float: Tahmin edilen değer
        """
        if not self.is_trained or self.model is None:
            logger.error("Tahmin için eğitilmiş bir model gerekli!")
            return None

        # Girdi dizisini kontrol et ve yeniden şekillendir
        if len(sequence.shape) == 2:
            # (lookback, features) -> (1, lookback, features)
            sequence = np.array([sequence])

        # Tahmini al
        predicted = self.model.predict(sequence, verbose=0)[0][0]

        # Denormalize et
        if denormalize and self.scaler is not None:
            n_features = len(self.feature_columns)
            dummy_array = np.zeros((1, n_features + 1))
            dummy_array[0, -1] = predicted
            denormalized = self.scaler.inverse_transform(dummy_array)[0, -1]
            return denormalized

        return predicted

    def predict_future(self, last_sequence, steps=5, denormalize=True):
        """
        Gelecekteki birden fazla değeri tahmin eder

        Args:
            last_sequence (np.array): Son veri dizisi
            steps (int): Kaç adım ileriye tahmin edileceği
            denormalize (bool): Tahminler denormalize edilsin mi

        Returns:
            list: Tahmin edilen değerler
        """
        if not self.is_trained or self.model is None:
            logger.error("Tahmin için eğitilmiş bir model gerekli!")
            return None

        # Girdi dizisini kontrol et
        original_shape = last_sequence.shape

        if len(original_shape) == 3 and original_shape[0] == 1:
            # (1, lookback, features) -> (lookback, features)
            current_sequence = last_sequence[0].copy()
        elif len(original_shape) == 2:
            # (lookback, features)
            current_sequence = last_sequence.copy()
        else:
            logger.error(f"Geçersiz girdi dizisi şekli: {original_shape}")
            return None

        logger.info(f"{steps} adım ileri tahmin yapılıyor...")

        future_predictions = []

        for _ in range(steps):
            # Mevcut diziden bir tahmin yap
            next_pred = self.predict_next(np.array([current_sequence]), denormalize=False)
            future_predictions.append(next_pred)

            # Diziyi güncelle: ilk satırı sil, sona yeni bir satır ekle
            # Bu yeni satır, girdi özelliklerinin son satırını ve yeni tahmini içerir
            new_row = np.zeros_like(current_sequence[-1])

            # Son satırdaki öznitelikleri kopyala
            # Eğer hedef "close" değeri sütun 0-3 arasındaysa (OHLC), o sütunu güncelle
            target_idx = self.feature_columns.index(
                self.target_column) if self.target_column in self.feature_columns else -1

            if target_idx >= 0:
                new_row = current_sequence[-1].copy()
                new_row[target_idx] = next_pred
            else:
                # Bu durum genellikle olmamalı, ama bir hata korumadır
                new_row = current_sequence[-1].copy()

            # Diziden ilk satırı at ve sona yeni satırı ekle
            current_sequence = np.vstack([current_sequence[1:], new_row])

        # Tahminleri denormalize et
        if denormalize and self.scaler is not None:
            n_features = len(self.feature_columns)
            dummy_array = np.zeros((len(future_predictions), n_features + 1))
            dummy_array[:, -1] = np.array(future_predictions).flatten()
            denormalized = self.scaler.inverse_transform(dummy_array)[:, -1]
            future_predictions = denormalized.tolist()

        return future_predictions