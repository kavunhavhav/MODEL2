#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binance LSTM Kripto Tahmin Botu - Yapılandırma Dosyası
"""

# API bilgileri
API_KEYS = {
    "live": {
        "api_key": "",
        "api_secret": ""
    },
    "testnet": {
        "api_key": "",
        "api_secret": "",
        "base_url": "https://testnet.binance.vision"
    }
}

# Desteklenen zaman aralıkları
SUPPORTED_INTERVALS = ["1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w"]

# Varsayılan model parametreleri
DEFAULT_MODEL_PARAMS = {
    "lookback": 60,
    "epochs": 50,
    "batch_size": 32,
    "train_size": 0.8,
    "lstm_units_1": 128,
    "lstm_units_2": 64,
    "dropout_rate": 0.3,
    "learning_rate": 0.001
}

# Varsayılan simülasyon parametreleri
DEFAULT_SIMULATION_PARAMS = {
    "initial_balance": 100.0,
    "fee_percent": 0.1,
    "stop_loss_percent": 2.0,
    "take_profit_percent": 5.0,
    "max_open_positions": 3,
    "position_size_percent": 30.0
}

# Tarama kriterleri varsayılan değerleri
DEFAULT_SCANNER_PARAMS = {
    "min_volume_usd": 20000000,  # Minimum 24s hacim (USD)
    "min_price_change": 5.0,     # Minimum fiyat değişimi (%)
    "timeframe": "2h",           # Son 2 saat
    "max_results": 10,           # Maksimum sonuç sayısı
    "min_market_cap": 1000000,   # Minimum piyasa değeri
    "exclude_stablecoins": True  # Stablecoin'leri hariç tut
}

# Gösterge grupları
INDICATOR_GROUPS = {
    "Trend": ["SMA_10", "SMA_20", "SMA_50", "EMA_10", "EMA_20", "EMA_50", "MACD", "Parabolic_SAR", "Ichimoku_Cloud"],
    "Momentum": ["RSI_14", "Stochastic_14_3_3", "CCI_20", "MFI_14", "Williams_%R", "ROC_10"],
    "Volatility": ["Bollinger_Bands_20_2", "ATR_14", "Keltner_Channels_20_2", "Standard_Deviation_20"],
    "Volume": ["OBV", "VWAP", "Chaikin_Money_Flow_20", "Accumulation_Distribution"],
    "Support/Resistance": ["Pivot_Points", "Fibonacci_Retracement_38.2", "Fibonacci_Retracement_61.8"],
    "Advanced": ["ADX_14", "Hurst_Exponent_30", "GARCH_Volatility", "Fractal_Dimension"]
}

# Zaman aralığı tavsiyeleri
TIMEFRAME_RECOMMENDATIONS = {
    "5m": "Yüksek frekanslı veri için RSI_9, EMA_12, VWAP ve OBV önerilir. Ayrıca noise'ı azaltmak için SMA_20 ekleyin.",
    "15m": "Yüksek frekanslı veri için RSI_9, EMA_12, VWAP ve OBV önerilir. Ayrıca noise'ı azaltmak için SMA_20 ekleyin.",
    "30m": "Orta vadeli analizler için RSI_14, Bollinger Bands, MFI_14 ve MACD kullanabilirsiniz.",
    "1h": "Saatlik analizlerde EMA_20, EMA_50, RSI_14 ve Volume göstergeleri önemlidir.",
    "4h": "Uzun vadeli trendler için MACD, Ichimoku Cloud ve ADX kullanın. Destek/direnç seviyeleri için Fibonacci ekleyin.",
    "1d": "Günlük analizlerde SMA_50, SMA_200, RSI_14 ve Pivot noktaları daha verimlidir."
}

# GUI Ayarları
GUI_SETTINGS = {
    "window_title": "Binance LSTM Kripto Tahmin Botu",
    "window_size": "1000x700",
    "min_window_size": "800x600",
    "theme": {
        "bg_color": "#2c3e50",
        "fg_color": "#ecf0f1",
        "accent_color": "#3498db",
        "button_text_color": "#ffffff",
        "success_color": "#2ecc71",
        "warning_color": "#f39c12",
        "danger_color": "#e74c3c"
    }
}

# Dosya yolları
PATHS = {
    "models_dir": "models",
    "data_dir": "data",
    "logs_dir": "logs",
    "results_dir": "results"
}

# Hariç tutulacak coinler
EXCLUDED_COINS = [
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "PAX", "USDP",  # Stablecoinler
    "BULL", "BEAR", "UP", "DOWN", "HEDGE"  # Kaldıraçlı tokenler
]

# Varsayılan sütunlar
DEFAULT_DISPLAY_COLUMNS = [
    "timestamp", "open", "high", "low", "close", "volume",
    "predicted_close", "actual_direction", "predicted_direction",
    "is_correct", "pnl", "balance", "trade_note"
]

# Toplam veri süresi (gün)
DEFAULT_DATA_DAYS = 180  # 6 ay veri

# Excel export ayarları
EXCEL_SETTINGS = {
    "default_filename": "kripto_tahmin_sonuclari.xlsx",
    "sheet_names": {
        "sonuclar": "Tahmin Sonuçları",
        "ozet": "Özet İstatistikler",
        "model": "Model Bilgileri",
        "ayarlar": "Simülasyon Ayarları"
    }
}