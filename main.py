#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binance LSTM Kripto Tahmin Botu - Ana Dosya
"""
import gui.simulation_panel as sp
import inspect
print("📎 Yüklenen simulation_panel.py dosya yolu:", inspect.getfile(sp))

import os
import sys
import logging

import argparse
from datetime import datetime
import tkinter as tk

# Uygulama modülleri
import config
import utils
from api_manager import BinanceAPIManager
from data_fetcher import DataFetcher
from indicator_calculator import IndicatorCalculator
from model_manager import ModelManager
from simulator import TradingSimulator
from scanner import CryptoScanner

# GUI modülleri (uygun klasör yapısı için)
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gui.main_window import MainWindow


def main():
    # Argümanları ayarla
    parser = argparse.ArgumentParser(description="Binance LSTM Kripto Tahmin Botu")
    parser.add_argument('--gui', action='store_true', help='GUI modunda başlat')
    parser.add_argument('--console', action='store_true', help='Konsol modunda başlat')
    parser.add_argument('--debug', action='store_true', help='Debug modunu etkinleştir')
    parser.add_argument('--symbol', type=str, default='BTCUSDT', help='İşlem çifti (varsayılan: BTCUSDT)')
    parser.add_argument('--interval', type=str, default='1h', help='Zaman aralığı (varsayılan: 1h)')
    parser.add_argument('--scan', action='store_true', help='Pazar taraması yap')
    parser.add_argument('--model', type=str, help='Kullanılacak model dosyası')

    args = parser.parse_args()

    # Loglama ayarları
    log_level = logging.DEBUG if args.debug else logging.INFO
    utils.setup_logging(level=log_level)
    logger = logging.getLogger('main')

    # Klasörleri oluştur
    for path_name, path in config.PATHS.items():
        os.makedirs(path, exist_ok=True)

    # Başlangıç mesajı
    logger.info("=" * 50)
    logger.info("Binance LSTM Kripto Tahmin Botu Başlıyor")
    logger.info(f"Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 50)

    # Ana bileşenleri oluştur
    api_manager = BinanceAPIManager()
    data_fetcher = DataFetcher(api_manager)
    indicator_calculator = IndicatorCalculator()
    model_manager = ModelManager(args.symbol, args.interval)
    simulator = TradingSimulator(api_manager, model_manager, data_fetcher)
    scanner = CryptoScanner(api_manager, data_fetcher)

    # GUI veya konsol modunda başlat
    if args.gui or (not args.console and not args.scan):
        logger.info("GUI modu başlatılıyor...")
        root = tk.Tk()
        app = MainWindow(
            root,
            api_manager=api_manager,
            data_fetcher=data_fetcher,
            indicator_calculator=indicator_calculator,
            model_manager=model_manager,
            simulator=simulator,
            scanner=scanner
        )
        root.mainloop()
    elif args.scan:
        logger.info("Pazar taraması başlatılıyor...")
        # API'ye bağlan
        if not api_manager.connect():
            logger.error("API bağlantısı kurulamadı!")
            return

        # Tarama parametrelerini ayarla
        scanner.set_parameters(
            min_volume_usd=20000000,  # 20M USD
            min_price_change=3.0,  # %3 değişim
            timeframe='4h',  # 4 saatlik
            max_results=10  # En iyi 10 sonuç
        )

        # Taramayı başlat
        scanner.scan_all_markets()

        # Tarama tamamlanana kadar bekle
        scanner.scan_thread.join()

        # Sonuçları göster
        results = scanner.get_scan_results()
        print("\n=== Tarama Sonuçları ===")
        for i, result in enumerate(results, 1):
            symbol = result['symbol']
            price = result['price']
            price_change = result['price_change_pct']
            volume = utils.format_large_number(result['volume_24h'])
            score = result['interest_score']

            print(f"{i}. {symbol}: ${price:.6f} ({price_change:+.2f}%) | Hacim: ${volume} | Skor: {score:.1f}")
    else:
        logger.info("Konsol modu başlatılıyor...")
        # API'ye bağlan
        if not api_manager.connect():
            logger.error("API bağlantısı kurulamadı!")
            return

        # Veri çek
        df = data_fetcher.fetch_historical_data(args.symbol, args.interval, lookback_days=60)

        if df is None:
            logger.error(f"{args.symbol} için veri çekilemedi!")
            return

        logger.info(f"Veri yüklendi: {len(df)} satır.")

        # Gösterge hesapla
        recommended_indicators = indicator_calculator.get_recommended_indicators(args.interval)
        df_with_indicators = indicator_calculator.calculate_indicators(df, recommended_indicators)

        if args.model:
            # Modeli yükle
            model_path = args.model
            if not os.path.exists(model_path):
                model_path = os.path.join(config.PATHS['models_dir'], args.model)

            if os.path.exists(model_path):
                logger.info(f"Model yükleniyor: {model_path}")
                model_manager.load_model(model_path)

                # Model yüklendiyse tahmin yap
                if model_manager.is_trained:
                    # Son veriyi hazırla
                    X_train, y_train, X_test, y_test, scaler = model_manager.prepare_data(df_with_indicators)

                    # Tahmin yap
                    eval_results = model_manager.evaluate_model(X_test, y_test)

                    if eval_results:
                        logger.info(
                            f"Model değerlendirmesi: RMSE={eval_results['rmse']:.6f}, Yön Doğruluğu={eval_results['direction_accuracy']:.2f}%")

                        # Geri-test simülasyonu
                        simulator.run_backtest(df_with_indicators)

                        # Gelecek tahmini
                        future_steps = 5
                        logger.info(f"Gelecek {future_steps} zaman dilimi için tahminler:")

                        last_sequence = X_test[-1:]
                        future_preds = model_manager.predict_future(last_sequence, steps=future_steps)

                        for i, pred in enumerate(future_preds, 1):
                            logger.info(f"  {i}. adım: {pred:.6f}")
                    else:
                        logger.error("Model değerlendirme hatası!")
            else:
                logger.error(f"Model dosyası bulunamadı: {model_path}")
        else:
            logger.info("Model eğitme işlemi başlatılıyor...")

            # Veriyi hazırla
            X_train, y_train, X_test, y_test, scaler = model_manager.prepare_data(df_with_indicators)

            # Modeli oluştur ve eğit
            model_manager.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
            model_manager.train_model(X_train, y_train)

            # Modeli değerlendir
            model_manager.evaluate_model(X_test, y_test)

            # Modeli kaydet
            model_manager.save_model()

    logger.info("Uygulama tamamlandı.")


if __name__ == "__main__":
    main()