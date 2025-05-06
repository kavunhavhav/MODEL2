#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binance LSTM Kripto Tahmin Botu - Tarama Paneli
"""

import os
import time
import logging
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import config
import utils

logger = logging.getLogger(__name__)


class ScannerPanel(ttk.Frame):
    """Tarama Paneli Sınıfı"""

    def __init__(self, parent, scanner, api_manager, data_fetcher, log_callback, status_callback):
        """
        Tarama panelini başlatır

        Args:
            parent: Üst widget
            scanner: Tarama modülü
            api_manager: API yöneticisi
            data_fetcher: Veri çekme modülü
            log_callback: Log mesajı callback fonksiyonu
            status_callback: Durum güncelleme callback fonksiyonu
        """
        super().__init__(parent, padding=10)
        self.scanner = scanner
        self.api_manager = api_manager
        self.data_fetcher = data_fetcher
        self.log_callback = log_callback
        self.status_callback = status_callback

        # Tarayıcı callback'lerini ayarla
        self.scanner.log_callback = self.log_callback
        self.scanner.progress_callback = self.update_progress
        self.scanner.result_callback = self.add_scan_result

        # Değişkenler
        self.min_volume_var = tk.StringVar(value=str(int(config.DEFAULT_SCANNER_PARAMS['min_volume_usd'] / 1000000)))
        self.min_change_var = tk.StringVar(value=str(config.DEFAULT_SCANNER_PARAMS['min_price_change']))
        self.timeframe_var = tk.StringVar(value=config.DEFAULT_SCANNER_PARAMS['timeframe'])
        self.max_results_var = tk.StringVar(value=str(config.DEFAULT_SCANNER_PARAMS['max_results']))
        self.quote_asset_var = tk.StringVar(value="USDT")
        self.exclude_stablecoins_var = tk.BooleanVar(value=config.DEFAULT_SCANNER_PARAMS['exclude_stablecoins'])

        self.current_symbol = None
        self.scan_results = []

        # GUI oluştur
        self.create_criteria_frame()
        self.create_results_frame()
        self.create_analysis_frame()

    def create_criteria_frame(self):
        """Tarama kriterleri panelini oluşturur"""
        criteria_frame = ttk.LabelFrame(self, text="Tarama Kriterleri", padding=10)
        criteria_frame.pack(fill=tk.X, padx=5, pady=5)

        # Min Hacim
        ttk.Label(criteria_frame, text="Minimum Hacim (Milyon $):").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(criteria_frame, textvariable=self.min_volume_var, width=10).grid(row=0, column=1, pady=2, padx=5,
                                                                                   sticky=tk.W)

        # Min Değişim
        ttk.Label(criteria_frame, text="Minimum Değişim (%):").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(criteria_frame, textvariable=self.min_change_var, width=10).grid(row=1, column=1, pady=2, padx=5,
                                                                                   sticky=tk.W)

        # Zaman Dilimi
        ttk.Label(criteria_frame, text="Zaman Dilimi:").grid(row=0, column=2, sticky=tk.W, pady=2, padx=(20, 0))
        ttk.Combobox(criteria_frame, textvariable=self.timeframe_var, width=8,
                     values=["15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]).grid(row=0, column=3, pady=2, padx=5,
                                                                                      sticky=tk.W)

        # Quote Asset
        ttk.Label(criteria_frame, text="Quote Asset:").grid(row=1, column=2, sticky=tk.W, pady=2, padx=(20, 0))
        ttk.Combobox(criteria_frame, textvariable=self.quote_asset_var, width=8,
                     values=["USDT", "BUSD", "USDC", "BTC", "ETH"]).grid(row=1, column=3, pady=2, padx=5, sticky=tk.W)

        # Maksimum Sonuç
        ttk.Label(criteria_frame, text="Maksimum Sonuç:").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(criteria_frame, textvariable=self.max_results_var, width=10).grid(row=2, column=1, pady=2, padx=5,
                                                                                    sticky=tk.W)

        # Stablecoin Hariç Tut
        ttk.Checkbutton(criteria_frame, text="Stablecoin'leri Hariç Tut", variable=self.exclude_stablecoins_var).grid(
            row=2, column=2, columnspan=2, sticky=tk.W, pady=2, padx=(20, 0))

        # Butonlar
        button_frame = ttk.Frame(criteria_frame)
        button_frame.grid(row=3, column=0, columnspan=4, pady=10)

        # Tarama Butonu
        self.scan_button = ttk.Button(button_frame, text="Taramayı Başlat", command=self.start_scan)
        self.scan_button.pack(side=tk.LEFT, padx=5)

        # Durdurma Butonu
        self.stop_button = ttk.Button(button_frame, text="Taramayı Durdur", command=self.stop_scan, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)

        # Dışa Aktarma Butonu
        self.export_button = ttk.Button(button_frame, text="Sonuçları Dışa Aktar", command=self.export_results,
                                        state=tk.DISABLED)
        self.export_button.pack(side=tk.LEFT, padx=5)

        # İlerleme çubuğu
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(criteria_frame, variable=self.progress_var, length=100, mode='determinate')
        self.progress_bar.grid(row=4, column=0, columnspan=4, sticky=tk.EW, pady=5)

    def create_results_frame(self):
        """Sonuçlar panelini oluşturur"""
        results_frame = ttk.LabelFrame(self, text="Tarama Sonuçları", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Sonuç listesi
        columns = ("sembol", "fiyat", "hacim", "degisim", "skor", "volatilite", "rsi", "sinyaller")
        self.results_tree = ttk.Treeview(results_frame, columns=columns, show="headings", height=8)

        # Sütun başlıkları
        self.results_tree.heading("sembol", text="Sembol")
        self.results_tree.heading("fiyat", text="Fiyat")
        self.results_tree.heading("hacim", text="24s Hacim ($)")
        self.results_tree.heading("degisim", text="Değişim (%)")
        self.results_tree.heading("skor", text="İlgi Skoru")
        self.results_tree.heading("volatilite", text="Volatilite (%)")
        self.results_tree.heading("rsi", text="RSI")
        self.results_tree.heading("sinyaller", text="Sinyaller")

        # Sütun genişlikleri
        self.results_tree.column("sembol", width=80)
        self.results_tree.column("fiyat", width=100)
        self.results_tree.column("hacim", width=120)
        self.results_tree.column("degisim", width=100)
        self.results_tree.column("skor", width=80)
        self.results_tree.column("volatilite", width=80)
        self.results_tree.column("rsi", width=60)
        self.results_tree.column("sinyaller", width=200)

        # Scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)

        # Widget yerleşimi
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Olay bağlama
        self.results_tree.bind("<Double-1>", self.on_result_double_click)

    def create_analysis_frame(self):
        """Analiz panelini oluşturur"""
        analysis_frame = ttk.LabelFrame(self, text="Coin Analizi", padding=10)
        analysis_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Sol Panel - Grafik
        chart_frame = ttk.Frame(analysis_frame)
        chart_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # Grafik için yer tutucu
        self.chart_placeholder = ttk.Label(chart_frame, text="Analiz için bir coin seçin")
        self.chart_placeholder.pack(fill=tk.BOTH, expand=True)

        # Sağ Panel - Analiz Sonuçları
        info_frame = ttk.Frame(analysis_frame)
        info_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(10, 0))

        # Coin bilgileri
        self.coin_info_frame = ttk.LabelFrame(info_frame, text="Coin Bilgileri", padding=10)
        self.coin_info_frame.pack(fill=tk.X, pady=5)

        # Varsayılan bilgi etiketi
        self.info_placeholder = ttk.Label(self.coin_info_frame, text="Coin seçilmedi")
        self.info_placeholder.pack(fill=tk.X)

        # Teknik göstergeler çerçevesi
        self.indicators_frame = ttk.LabelFrame(info_frame, text="Teknik Göstergeler", padding=10)
        self.indicators_frame.pack(fill=tk.X, pady=5)

        # Varsayılan gösterge etiketi
        self.indicators_placeholder = ttk.Label(self.indicators_frame, text="Gösterge bilgisi yok")
        self.indicators_placeholder.pack(fill=tk.X)

        # Sinyaller çerçevesi
        self.signals_frame = ttk.LabelFrame(info_frame, text="Sinyaller", padding=10)
        self.signals_frame.pack(fill=tk.X, pady=5)

        # Varsayılan sinyal etiketi
        self.signals_placeholder = ttk.Label(self.signals_frame, text="Sinyal bilgisi yok")
        self.signals_placeholder.pack(fill=tk.X)

        # Analiz butonları çerçevesi
        buttons_frame = ttk.Frame(info_frame)
        buttons_frame.pack(fill=tk.X, pady=10)

        # Analiz butonları
        ttk.Button(buttons_frame, text="Detaylı Analiz", command=self.detailed_analysis).pack(side=tk.LEFT, padx=5)
        ttk.Button(buttons_frame, text="Çoklu Zaman Dilimi", command=self.multi_timeframe_analysis).pack(side=tk.LEFT,
                                                                                                         padx=5)

    def start_scan(self):
        """Taramayı başlatır"""
        if not self.api_manager.connected:
            self.log_callback("API bağlantısı kurulmamış!", error=True)
            messagebox.showerror("Hata", "Önce API bağlantısını kurmanız gerekiyor!")
            return

        # Parametreleri ayarla
        try:
            min_volume = float(self.min_volume_var.get()) * 1000000  # Milyon dolar
            min_change = float(self.min_change_var.get())
            timeframe = self.timeframe_var.get()
            max_results = int(self.max_results_var.get())
            quote_asset = self.quote_asset_var.get()
            exclude_stablecoins = self.exclude_stablecoins_var.get()
        except ValueError as e:
            self.log_callback(f"Geçersiz parametre: {str(e)}", error=True)
            messagebox.showerror("Hata", "Geçersiz parametre değeri. Lütfen sayısal değerler girin.")
            return

        # Scanner parametrelerini ayarla
        self.scanner.set_parameters(
            min_volume_usd=min_volume,
            min_price_change=min_change,
            timeframe=timeframe,
            max_results=max_results,
            exclude_stablecoins=exclude_stablecoins
        )

        # Ağaçtaki sonuçları temizle
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)

        self.scan_results = []

        # Buton durumlarını güncelle
        self.scan_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.export_button.config(state=tk.DISABLED)

        # İlerleme çubuğunu sıfırla
        self.progress_var.set(0)

        # Taramayı başlat
        self.log_callback(
            f"Tarama başlatılıyor... (Min Hacim: ${min_volume / 1000000:.1f}M, Min Değişim: %{min_change}, Zaman Dilimi: {timeframe})")
        self.status_callback("Taranıyor...", "Aktif")

        self.scanner.scan_all_markets(quote_asset=quote_asset)

    def stop_scan(self):
        """Taramayı durdurur"""
        if self.scanner.is_scanning:
            self.scanner.stop_scanning()
            self.log_callback("Tarama kullanıcı tarafından durduruldu.")
            self.status_callback("Tarama durduruldu", "Aktif")

        # Buton durumlarını güncelle
        self.scan_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

        if self.scan_results:
            self.export_button.config(state=tk.NORMAL)

    def update_progress(self, progress):
        """
        İlerleme çubuğunu günceller

        Args:
            progress (float): İlerleme yüzdesi (0-100)
        """
        self.progress_var.set(progress)

        if progress >= 100:
            # Tarama tamamlandı
            self.scan_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)

            if self.scan_results:
                self.export_button.config(state=tk.NORMAL)

            self.status_callback("Tarama tamamlandı", "Aktif")

    def add_scan_result(self, result):
        """
        Tarama sonucu ekler

        Args:
            result (dict): Tarama sonucu
        """
        self.scan_results.append(result)

        # Sinyalleri formatla
        signals = []
        if result.get('has_rsi_signal', False):
            signals.append("RSI")

        if result.get('has_volume_spike', False):
            signals.append("Hacim Artışı")

        if result.get('has_price_breakout', False):
            signals.append("Fiyat Kırılımı")

        # Ağaca ekle
        self.results_tree.insert("", tk.END,
                                 values=(
                                     result['symbol'],
                                     f"{result['price']:.6f}",
                                     utils.format_large_number(result['volume_24h']),
                                     f"{result['price_change_pct']:+.2f}%",
                                     f"{result['interest_score']:.1f}",
                                     f"{result['volatility']:.2f}%",
                                     f"{result['rsi']:.1f}",
                                     ", ".join(signals)
                                 )
                                 )

    def on_result_double_click(self, event):
        """
        Sonuç listesinde çift tıklama olayı

        Args:
            event: Tıklama olayı
        """
        item = self.results_tree.selection()[0]
        values = self.results_tree.item(item, "values")

        if values:
            symbol = values[0]
            self.current_symbol = symbol

            # Coin analiz et
            self.analyze_coin(symbol)

    def analyze_coin(self, symbol):
        """
        Coin analizi yapar

        Args:
            symbol (str): Coin sembolü
        """
        self.log_callback(f"{symbol} analiz ediliyor...")

        try:
            # Son verileri çek
            interval = self.timeframe_var.get()
            df = self.data_fetcher.fetch_latest_data(symbol, interval=interval, limit=100)

            if df is None or len(df) < 20:
                self.log_callback(f"{symbol} için yeterli veri bulunamadı!", error=True)
                return

            # Grafik çiz
            self.draw_chart(symbol, df)

            # Coin bilgilerini göster
            self.show_coin_info(symbol, df)

        except Exception as e:
            self.log_callback(f"{symbol} analiz edilirken hata: {str(e)}", error=True)

    def draw_chart(self, symbol, df):
        """
        Grafik çizer

        Args:
            symbol (str): Coin sembolü
            df (pd.DataFrame): Veri
        """
        # Mevcut grafiği temizle
        if hasattr(self, 'canvas'):
            self.canvas.get_tk_widget().destroy()

        if hasattr(self, 'chart_placeholder'):
            self.chart_placeholder.pack_forget()

        # Figür oluştur
        fig, ax = plt.subplots(figsize=(8, 4))

        # OHLC çiz
        ax.plot(df.index, df['close'], label='Fiyat', color='blue')

        # Son 20 mumun basit hareketli ortalaması
        sma20 = df['close'].rolling(window=20).mean()
        ax.plot(df.index, sma20, label='SMA(20)', color='red', linestyle='--', alpha=0.7)

        # Zaman formatlama
        if len(df) > 100:
            date_format = '%Y-%m-%d'
        else:
            date_format = '%Y-%m-%d %H:%M'

        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(date_format))
        plt.xticks(rotation=45)

        # Grafik ayarları
        ax.set_title(f'{symbol} - {self.timeframe_var.get()}')
        ax.set_ylabel('Fiyat')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left')

        # Figürü Tkinter'a yerleştir
        chart_frame = self.chart_placeholder.master
        self.canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def show_coin_info(self, symbol, df):
        """
        Coin bilgilerini gösterir

        Args:
            symbol (str): Coin sembolü
            df (pd.DataFrame): Veri
        """
        # Placeholder etiketlerini temizle
        for widget in self.coin_info_frame.winfo_children():
            widget.destroy()

        for widget in self.indicators_frame.winfo_children():
            widget.destroy()

        for widget in self.signals_frame.winfo_children():
            widget.destroy()

        # Temel bilgiler
        current_price = df['close'].iloc[-1]
        prev_price = df['close'].iloc[-2]
        first_price = df['close'].iloc[0]

        price_change = (current_price - prev_price) / prev_price * 100
        total_change = (current_price - first_price) / first_price * 100

        high_price = df['high'].max()
        low_price = df['low'].min()

        ttk.Label(self.coin_info_frame, text=f"Sembol: {symbol}").pack(anchor=tk.W)
        ttk.Label(self.coin_info_frame, text=f"Fiyat: {current_price:.8f}").pack(anchor=tk.W)
        ttk.Label(self.coin_info_frame, text=f"Değişim (Son): {price_change:+.2f}%").pack(anchor=tk.W)
        ttk.Label(self.coin_info_frame, text=f"Değişim (Toplam): {total_change:+.2f}%").pack(anchor=tk.W)
        ttk.Label(self.coin_info_frame, text=f"En Yüksek: {high_price:.8f}").pack(anchor=tk.W)
        ttk.Label(self.coin_info_frame, text=f"En Düşük: {low_price:.8f}").pack(anchor=tk.W)

        # Teknik göstergeler
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        last_rsi = rsi.iloc[-1]

        # Bollinger Bantları
        ma20 = df['close'].rolling(window=20).mean()
        std20 = df['close'].rolling(window=20).std()
        upper_band = ma20 + (std20 * 2)
        lower_band = ma20 - (std20 * 2)

        # Volatilite
        volatility = df['close'].pct_change().std() * 100

        # MACD
        ema12 = df['close'].ewm(span=12, adjust=False).mean()
        ema26 = df['close'].ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()

        ttk.Label(self.indicators_frame, text=f"RSI(14): {last_rsi:.2f}").pack(anchor=tk.W)
        ttk.Label(self.indicators_frame, text=f"MACD: {macd.iloc[-1]:.6f}").pack(anchor=tk.W)
        ttk.Label(self.indicators_frame, text=f"MACD Signal: {signal.iloc[-1]:.6f}").pack(anchor=tk.W)
        ttk.Label(self.indicators_frame, text=f"Bollinger Üst: {upper_band.iloc[-1]:.6f}").pack(anchor=tk.W)
        ttk.Label(self.indicators_frame, text=f"Bollinger Alt: {lower_band.iloc[-1]:.6f}").pack(anchor=tk.W)
        ttk.Label(self.indicators_frame, text=f"Volatilite: {volatility:.2f}%").pack(anchor=tk.W)

        # Sinyaller
        signals = []

        # RSI sinyalleri
        if last_rsi < 30:
            signals.append(("RSI Aşırı Satım", "green"))
        elif last_rsi > 70:
            signals.append(("RSI Aşırı Alım", "red"))

        # MACD sinyalleri
        if macd.iloc[-1] > signal.iloc[-1] and macd.iloc[-2] <= signal.iloc[-2]:
            signals.append(("MACD Yukarı Kesim", "green"))
        elif macd.iloc[-1] < signal.iloc[-1] and macd.iloc[-2] >= signal.iloc[-2]:
            signals.append(("MACD Aşağı Kesim", "red"))

        # Bollinger sinyalleri
        if current_price > upper_band.iloc[-1]:
            signals.append(("Bollinger Üst Kırılım", "red"))
        elif current_price < lower_band.iloc[-1]:
            signals.append(("Bollinger Alt Kırılım", "green"))

        # SMA sinyalleri
        if current_price > ma20.iloc[-1] and prev_price <= ma20.iloc[-2]:
            signals.append(("SMA20 Yukarı Kırılım", "green"))
        elif current_price < ma20.iloc[-1] and prev_price >= ma20.iloc[-2]:
            signals.append(("SMA20 Aşağı Kırılım", "red"))

        # Hacim sinyalleri
        volume_avg = df['volume'].rolling(window=20).mean()
        if df['volume'].iloc[-1] > volume_avg.iloc[-1] * 2:
            signals.append(("Hacim Patlaması", "orange"))

        if signals:
            for signal_text, color in signals:
                label = ttk.Label(self.signals_frame, text=signal_text)
                label.pack(anchor=tk.W)
                # Tkinter ttk etiketi için doğrudan renk ayarı
                label.configure(foreground=color)
        else:
            ttk.Label(self.signals_frame, text="Önemli bir sinyal yok").pack(anchor=tk.W)

    def detailed_analysis(self):
        """Detaylı analiz yapar"""
        if not self.current_symbol:
            messagebox.showinfo("Bilgi", "Lütfen önce bir coin seçin.")
            return

        # Scanner'ın analiz fonksiyonunu kullan
        self.log_callback(f"{self.current_symbol} için detaylı analiz yapılıyor...")

        analysis = self.scanner.analyze_coin(self.current_symbol, self.timeframe_var.get())

        if analysis:
            # Yeni pencere aç
            analysis_window = tk.Toplevel(self)
            analysis_window.title(f"{self.current_symbol} Detaylı Analiz")
            analysis_window.geometry("800x600")
            analysis_window.minsize(800, 600)

            # Detaylı analiz gösterim paneli
            content_frame = ttk.Frame(analysis_window, padding=10)
            content_frame.pack(fill=tk.BOTH, expand=True)

            # Üst bilgi paneli
            info_frame = ttk.LabelFrame(content_frame, text="Coin Bilgileri", padding=10)
            info_frame.pack(fill=tk.X, padx=5, pady=5)

            # İki sütunlu yerleşim için çerçeve
            info_grid = ttk.Frame(info_frame)
            info_grid.pack(fill=tk.X)

            # Sütun 1
            col1 = ttk.Frame(info_grid)
            col1.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            ttk.Label(col1, text=f"Sembol: {analysis['symbol']}").pack(anchor=tk.W, pady=2)
            ttk.Label(col1, text=f"Fiyat: {analysis['price']:.8f}").pack(anchor=tk.W, pady=2)
            ttk.Label(col1, text=f"Değişim: {analysis['price_change_pct']:+.2f}%").pack(anchor=tk.W, pady=2)
            ttk.Label(col1, text=f"Hacim: ${utils.format_large_number(analysis['volume_24h'])}").pack(anchor=tk.W,
                                                                                                      pady=2)

            # Sütun 2
            col2 = ttk.Frame(info_grid)
            col2.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

            ttk.Label(col2, text=f"Trend: {analysis['trend']}").pack(anchor=tk.W, pady=2)
            ttk.Label(col2, text=f"RSI: {analysis['rsi']:.2f}").pack(anchor=tk.W, pady=2)
            ttk.Label(col2, text=f"MACD: {analysis['macd']:.6f}").pack(anchor=tk.W, pady=2)
            ttk.Label(col2, text=f"Volatilite: {analysis.get('volatility', 0):.2f}%").pack(anchor=tk.W, pady=2)

            # Sinyaller paneli
            signals_frame = ttk.LabelFrame(content_frame, text="Sinyaller", padding=10)
            signals_frame.pack(fill=tk.X, padx=5, pady=5)

            signals = analysis.get('signals', [])
            if signals:
                for signal in signals:
                    ttk.Label(signals_frame, text=f"• {signal}").pack(anchor=tk.W, pady=2)
            else:
                ttk.Label(signals_frame, text="Önemli bir sinyal yok").pack(anchor=tk.W, pady=2)

            # Destek/Direnç seviyeleri
            levels_frame = ttk.LabelFrame(content_frame, text="Anahtar Seviyeler", padding=10)
            levels_frame.pack(fill=tk.X, padx=5, pady=5)

            key_levels = analysis.get('key_levels', {})

            if key_levels:
                resistance_levels = key_levels.get('resistance', [])
                support_levels = key_levels.get('support', [])

                if resistance_levels:
                    ttk.Label(levels_frame, text="Direnç Seviyeleri:").pack(anchor=tk.W, pady=2)
                    for i, level in enumerate(sorted(resistance_levels), 1):
                        ttk.Label(levels_frame, text=f"  {i}. {level:.8f}").pack(anchor=tk.W, padx=10)

                if support_levels:
                    ttk.Label(levels_frame, text="Destek Seviyeleri:").pack(anchor=tk.W, pady=2)
                    for i, level in enumerate(sorted(support_levels, reverse=True), 1):
                        ttk.Label(levels_frame, text=f"  {i}. {level:.8f}").pack(anchor=tk.W, padx=10)
            else:
                ttk.Label(levels_frame, text="Belirgin destek/direnç seviyesi bulunamadı").pack(anchor=tk.W, pady=2)

            # Özet ve öneriler
            summary_frame = ttk.LabelFrame(content_frame, text="Özet ve Öneriler", padding=10)
            summary_frame.pack(fill=tk.X, padx=5, pady=5)

            # Basit bir öneri algoritması
            buy_signals = sum(1 for s in signals if "YUKARI" in s or "AŞIRI SATIM" in s)
            sell_signals = sum(1 for s in signals if "AŞAĞI" in s or "AŞIRI ALIM" in s)

            if analysis['trend'] == "YÜKSELİŞ" and buy_signals > sell_signals:
                recommendation = "ALINMA Potansiyeli: Yükselen trend ve olumlu sinyaller mevcut."
                recommendation_color = "green"
            elif analysis['trend'] == "DÜŞÜŞ" and sell_signals > buy_signals:
                recommendation = "SATILMA Potansiyeli: Düşen trend ve olumsuz sinyaller mevcut."
                recommendation_color = "red"
            else:
                recommendation = "BEKLE: Belirgin bir yön gözlenmiyor veya sinyaller karışık."
                recommendation_color = "blue"

            recommendation_label = ttk.Label(summary_frame, text=recommendation)
            recommendation_label.pack(anchor=tk.W, pady=5)
            recommendation_label.configure(foreground=recommendation_color)

            ttk.Label(summary_frame,
                      text="NOT: Bu analiz bir öneri değildir. Kendi araştırmanızı yapın ve risklerinizi yönetin.").pack(
                anchor=tk.W, pady=5)
        else:
            messagebox.showerror("Hata", "Analiz yapılamadı!")

    def multi_timeframe_analysis(self):
        """Çoklu zaman dilimi analizi yapar"""
        if not self.current_symbol:
            messagebox.showinfo("Bilgi", "Lütfen önce bir coin seçin.")
            return

        # Tarayıcının çoklu zaman dilimi fonksiyonunu kullan
        self.log_callback(f"{self.current_symbol} için çoklu zaman dilimi analizi yapılıyor...")

        intervals = ["15m", "1h", "4h", "1d"]
        result = self.scanner.get_multi_timeframe_signals(self.current_symbol, intervals)

        if result:
            # Yeni pencere aç
            mtf_window = tk.Toplevel(self)
            mtf_window.title(f"{self.current_symbol} Çoklu Zaman Dilimi Analizi")
            mtf_window.geometry("900x700")
            mtf_window.minsize(900, 700)

            # Ana çerçeve
            main_frame = ttk.Frame(mtf_window, padding=10)
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Üst bilgi paneli
            info_frame = ttk.LabelFrame(main_frame, text="Özet", padding=10)
            info_frame.pack(fill=tk.X, padx=5, pady=5)

            # Özet bilgiler
            ttk.Label(info_frame, text=f"Sembol: {result['symbol']}").pack(anchor=tk.W, pady=2)
            ttk.Label(info_frame, text=f"Genel Eğilim: {result['overall_bias']}").pack(anchor=tk.W, pady=2)
            ttk.Label(info_frame, text=f"Sinyal Gücü: {result['signal_strength']:.2f}").pack(anchor=tk.W, pady=2)
            ttk.Label(info_frame, text=f"Alım Sinyalleri: {result['buy_signals']}").pack(anchor=tk.W, pady=2)
            ttk.Label(info_frame, text=f"Satım Sinyalleri: {result['sell_signals']}").pack(anchor=tk.W, pady=2)

            # Detaylı sinyal paneli
            signals_frame = ttk.LabelFrame(main_frame, text="Zaman Dilimi Sinyalleri", padding=10)
            signals_frame.pack(fill=tk.X, padx=5, pady=5)

            for signal in result['detailed_signals']:
                signal_text = signal
                signal_color = "black"

                if "ALIM" in signal:
                    signal_color = "green"
                elif "SATIM" in signal:
                    signal_color = "red"

                signal_label = ttk.Label(signals_frame, text=signal_text)
                signal_label.pack(anchor=tk.W, pady=2)
                signal_label.configure(foreground=signal_color)

            # Grafikler için notebook
            notebook = ttk.Notebook(main_frame)
            notebook.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

            # Her zaman dilimi için bir sekme oluştur
            for interval, analysis in result['timeframe_results'].items():
                if analysis is None:
                    continue

                # Sekme içeriği
                tab_frame = ttk.Frame(notebook, padding=10)
                notebook.add(tab_frame, text=interval)

                # Grafik çerçevesi
                chart_frame = ttk.Frame(tab_frame)
                chart_frame.pack(fill=tk.BOTH, expand=True)

                # Grafiği çiz
                fig, ax = plt.subplots(figsize=(8, 4))

                # OHLC çiz
                ax.plot(analysis['timestamp'].index, analysis['close'], label='Fiyat', color='blue')

                # SMA20
                if 'ma_20' in analysis:
                    ax.plot(analysis['timestamp'].index, analysis['ma_20'], label='SMA(20)', color='red',
                            linestyle='--', alpha=0.7)

                # Bollinger Bantları
                if all(k in analysis for k in ['bb_upper', 'bb_lower']):
                    ax.plot(analysis['timestamp'].index, analysis['bb_upper'], label='BB Üst', color='green',
                            linestyle='--', alpha=0.5)
                    ax.plot(analysis['timestamp'].index, analysis['bb_lower'], label='BB Alt', color='green',
                            linestyle='--', alpha=0.5)

                # Grafik ayarları
                ax.set_title(f'{self.current_symbol} - {interval}')
                ax.set_ylabel('Fiyat')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper left')

                # Figürü Tkinter'a yerleştir
                canvas = FigureCanvasTkAgg(fig, master=chart_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

                # Sinyaller paneli
                tab_signals_frame = ttk.LabelFrame(tab_frame, text=f"{interval} Sinyalleri", padding=10)
                tab_signals_frame.pack(fill=tk.X, padx=5, pady=5)

                # Sinyalleri göster
                signals = analysis.get('signals', [])

                if signals:
                    for signal in signals:
                        signal_color = "black"
                        if "YUKARI" in signal or "AŞIRI SATIM" in signal:
                            signal_color = "green"
                        elif "AŞAĞI" in signal or "AŞIRI ALIM" in signal:
                            signal_color = "red"

                        signal_label = ttk.Label(tab_signals_frame, text=f"• {signal}")
                        signal_label.pack(anchor=tk.W, pady=2)
                        signal_label.configure(foreground=signal_color)
                else:
                    ttk.Label(tab_signals_frame, text="Sinyal yok").pack(anchor=tk.W, pady=2)
        else:
            messagebox.showerror("Hata", "Çoklu zaman dilimi analizi yapılamadı!")

    def export_results(self):
        """Tarama sonuçlarını dışarı aktarır"""
        if not self.scan_results:
            messagebox.showinfo("Bilgi", "Dışa aktarılacak sonuç yok.")
            return

        try:
            # DataFrame oluştur
            df = pd.DataFrame(self.scan_results)

            # Dosya kaydet iletişim kutusu
            filename = f"tarama_sonuclari_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            filepath = filedialog.asksaveasfilename(
                title="Sonuçları Kaydet",
                initialfile=filename,
                defaultextension=".xlsx",
                filetypes=[("Excel Dosyası", "*.xlsx"), ("CSV Dosyası", "*.csv"), ("Tüm Dosyalar", "*.*")]
            )

            if not filepath:
                return

            # Dosya uzantısına göre kaydet
            if filepath.endswith('.xlsx'):
                df.to_excel(filepath, index=False)
            elif filepath.endswith('.csv'):
                df.to_csv(filepath, index=False)
            else:
                df.to_excel(filepath, index=False)

            self.log_callback(f"Tarama sonuçları kaydedildi: {filepath}")
            messagebox.showinfo("Başarılı", f"Tarama sonuçları başarıyla kaydedildi:\n{filepath}")

        except Exception as e:
            self.log_callback(f"Sonuçlar dışa aktarılırken hata: {str(e)}", error=True)
            messagebox.showerror("Hata", f"Sonuçlar dışa aktarılırken hata oluştu:\n{str(e)}")

    # !/usr/bin/env python
    # -*- coding: utf-8 -*-

    """
    Binance LSTM Kripto Tahmin Botu - Tarama Paneli
    """

    import os
    import time
    import logging
    import threading
    from datetime import datetime
    import tkinter as tk
    from tkinter import ttk, messagebox
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

    import config
    import utils

    logger = logging.getLogger(__name__)

    class ScannerPanel(ttk.Frame):
        """Tarama Paneli Sınıfı"""

        def __init__(self, parent, scanner, api_manager, data_fetcher, log_callback, status_callback):
            """
            Tarama panelini başlatır

            Args:
                parent: Üst widget
                scanner: Tarama modülü
                api_manager: API yöneticisi
                data_fetcher: Veri çekme modülü
                log_callback: Log mesajı callback fonksiyonu
                status_callback: Durum güncelleme callback fonksiyonu
            """
            super().__init__(parent, padding=10)
            self.scanner = scanner
            self.api_manager = api_manager
            self.data_fetcher = data_fetcher
            self.log_callback = log_callback
            self.status_callback = status_callback

            # Tarayıcı callback'lerini ayarla
            self.scanner.log_callback = self.log_callback
            self.scanner.progress_callback = self.update_progress
            self.scanner.result_callback = self.add_scan_result

            # Değişkenler
            self.min_volume_var = tk.StringVar(
                value=str(int(config.DEFAULT_SCANNER_PARAMS['min_volume_usd'] / 1000000)))
            self.min_change_var = tk.StringVar(value=str(config.DEFAULT_SCANNER_PARAMS['min_price_change']))
            self.timeframe_var = tk.StringVar(value=config.DEFAULT_SCANNER_PARAMS['timeframe'])
            self.max_results_var = tk.StringVar(value=str(config.DEFAULT_SCANNER_PARAMS['max_results']))
            self.quote_asset_var = tk.StringVar(value="USDT")
            self.exclude_stablecoins_var = tk.BooleanVar(value=config.DEFAULT_SCANNER_PARAMS['exclude_stablecoins'])

            self.current_symbol = None
            self.scan_results = []

            # GUI oluştur
            self.create_criteria_frame()
            self.create_results_frame()
            self.create_analysis_frame()

        def create_criteria_frame(self):
            """Tarama kriterleri panelini oluşturur"""
            criteria_frame = ttk.LabelFrame(self, text="Tarama Kriterleri", padding=10)
            criteria_frame.pack(fill=tk.X, padx=5, pady=5)

            # Min Hacim
            ttk.Label(criteria_frame, text="Minimum Hacim (Milyon $):").grid(row=0, column=0, sticky=tk.W, pady=2)
            ttk.Entry(criteria_frame, textvariable=self.min_volume_var, width=10).grid(row=0, column=1, pady=2, padx=5,
                                                                                       sticky=tk.W)

            # Min Değişim
            ttk.Label(criteria_frame, text="Minimum Değişim (%):").grid(row=1, column=0, sticky=tk.W, pady=2)
            ttk.Entry(criteria_frame, textvariable=self.min_change_var, width=10).grid(row=1, column=1, pady=2, padx=5,
                                                                                       sticky=tk.W)

            # Zaman Dilimi
            ttk.Label(criteria_frame, text="Zaman Dilimi:").grid(row=0, column=2, sticky=tk.W, pady=2, padx=(20, 0))
            ttk.Combobox(criteria_frame, textvariable=self.timeframe_var, width=8,
                         values=["15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]).grid(row=0, column=3, pady=2,
                                                                                          padx=5, sticky=tk.W)