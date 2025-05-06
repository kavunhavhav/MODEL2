#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binance LSTM Kripto Tahmin Botu - Simülasyon Paneli
"""

import os
import time
import logging
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import config
import utils

logger = logging.getLogger(__name__)


class SimulationPanel(ttk.Frame):
    """Simülasyon Paneli Sınıfı"""

    def __init__(self, parent, simulator, api_manager, data_fetcher, model_manager, indicator_calculator, log_callback,
                 status_callback):
        """
        Simülasyon panelini başlatır

        Args:
            parent: Üst widget
            simulator: Simülasyon modülü
            api_manager: API yöneticisi
            data_fetcher: Veri çekme modülü
            model_manager: Model yönetim modülü
            indicator_calculator: Gösterge hesaplama modülü
            log_callback: Log mesajı callback fonksiyonu
            status_callback: Durum güncelleme callback fonksiyonu
        """
        super().__init__(parent, padding=10)
        self.simulator = simulator
        self.api_manager = api_manager
        self.data_fetcher = data_fetcher
        self.model_manager = model_manager
        self.indicator_calculator = indicator_calculator
        self.log_callback = log_callback
        self.status_callback = status_callback
        self.root = parent  # Ana pencere referansı eklendi

        # Simülatör callback'lerini ayarla
        self.simulator.log_callback = self.log_callback
        self.simulator.progress_callback = lambda current, total, loss, val_loss: self.update_progress(
            "Model Eğitimi", current, total, f"Loss: {loss:.4f}, Val: {val_loss:.4f}"
        )
        self.simulator.result_callback = None

        # Değişkenler
        self.symbol_var = tk.StringVar(value="BTCUSDT")
        self.interval_var = tk.StringVar(value="1h")
        self.lookback_days_var = tk.StringVar(value="60")
        self.initial_balance_var = tk.StringVar(value=str(config.DEFAULT_SIMULATION_PARAMS['initial_balance']))
        self.fee_percent_var = tk.StringVar(value=str(config.DEFAULT_SIMULATION_PARAMS['fee_percent']))
        self.stop_loss_percent_var = tk.StringVar(value=str(config.DEFAULT_SIMULATION_PARAMS['stop_loss_percent']))
        self.take_profit_percent_var = tk.StringVar(value=str(config.DEFAULT_SIMULATION_PARAMS['take_profit_percent']))
        self.position_size_percent_var = tk.StringVar(
            value=str(config.DEFAULT_SIMULATION_PARAMS['position_size_percent']))
        self.selected_indicators = {}
        self.real_time_var = tk.BooleanVar(value=False)

        # İndikatör değişkenlerini doldur
        for group, indicators in config.INDICATOR_GROUPS.items():
            for indicator in indicators:
                self.selected_indicators[indicator] = tk.BooleanVar(value=False)

        # Varsayılan göstergeler
        default_indicators = ['RSI_14', 'MACD', 'Bollinger_Bands_20_2', 'SMA_20', 'EMA_20']
        for indicator in default_indicators:
            if indicator in self.selected_indicators:
                self.selected_indicators[indicator].set(True)

        # Geçici değişkenler
        self.current_df = None
        self.current_model_path = None
        self.is_simulation_running = False

        # GUI oluştur
        self.create_left_panel()
        self.create_right_panel()

    def update_progress(self, task_name, current, total, status=""):
        import inspect
        print("🟢 update_progress çağrıldı")
        print("📁 Kaynak dosya:", inspect.getfile(self.update_progress))

        try:
            percent = (current / total) * 100
            message = f"{task_name}: {current}/{total} - {status}"
            self._update_progress_gui(percent, message)
        except Exception as e:
            print(f"update_progress hatası: {e}")

    # Esnek argümanlarla güncellenmiş
    def _update_progress_gui(self, progress, message):
        """GUI bileşenlerini günceller (ana thread'de çalıştırılmalı)"""
        try:
            # Progress bar'ı güncelle (önemli değişiklerde)
            if abs(progress - self.progress_var.get()) > 1 or progress in (0, 100):
                self.progress_var.set(progress)
                self.progress_label.config(text=f"{progress:.1f}%")

            # Durum mesajını güncelle
            if message:
                self.status_callback(message, "Aktif")
        except Exception as e:
            logger.error(f"Progress güncelleme hatası: {str(e)}")

    def create_left_panel(self):
        """Sol panel oluşturur (ayarlar)"""
        left_panel = ttk.Frame(self)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))

        # Veri Seçimi
        data_frame = ttk.LabelFrame(left_panel, text="Veri Seçimi", padding=10)
        data_frame.pack(fill=tk.X, pady=5)

        # Sembol
        ttk.Label(data_frame, text="Sembol:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(data_frame, textvariable=self.symbol_var, width=15).grid(row=0, column=1, pady=2, padx=5, sticky=tk.W)

        # Zaman Dilimi
        ttk.Label(data_frame, text="Zaman Dilimi:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(data_frame, textvariable=self.interval_var, width=13,
                     values=["5m", "15m", "30m", "1h", "2h", "4h", "6h", "12h", "1d"]).grid(row=1, column=1, pady=2,
                                                                                            padx=5, sticky=tk.W)

        # Geriye Dönük Günler
        ttk.Label(data_frame, text="Geriye Dönük (gün):").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(data_frame, textvariable=self.lookback_days_var, width=15).grid(row=2, column=1, pady=2, padx=5,
                                                                                  sticky=tk.W)

        # Veri Butonları
        button_frame = ttk.Frame(data_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=5)

        ttk.Button(button_frame, text="Veri Çek", command=self.fetch_data).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Göstergeleri Hesapla", command=self.calculate_indicators).pack(side=tk.LEFT,
                                                                                                      padx=5)

        # Göstergeler
        indicators_frame = ttk.LabelFrame(left_panel, text="Göstergeler", padding=10)
        indicators_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Notebook ile gösterge grupları
        indicators_notebook = ttk.Notebook(indicators_frame)
        indicators_notebook.pack(fill=tk.BOTH, expand=True)

        # Her gösterge grubu için bir sekme oluştur
        for group, indicators in config.INDICATOR_GROUPS.items():
            group_frame = ttk.Frame(indicators_notebook, padding=5)
            indicators_notebook.add(group_frame, text=group)

            # Göstergeleri iki sütunda yerleştir
            for i, indicator in enumerate(indicators):
                row = i // 2
                col = i % 2

                if indicator in self.selected_indicators:
                    ttk.Checkbutton(
                        group_frame,
                        text=indicator,
                        variable=self.selected_indicators[indicator]
                    ).grid(row=row, column=col, sticky=tk.W, padx=5, pady=2)

        # Simülasyon Ayarları
        sim_frame = ttk.LabelFrame(left_panel, text="Simülasyon Ayarları", padding=10)
        sim_frame.pack(fill=tk.X, pady=5)

        # Başlangıç Bakiyesi
        ttk.Label(sim_frame, text="Başlangıç Bakiyesi ($):").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Entry(sim_frame, textvariable=self.initial_balance_var, width=10).grid(row=0, column=1, pady=2, padx=5,
                                                                                   sticky=tk.W)

        # İşlem Ücreti
        ttk.Label(sim_frame, text="İşlem Ücreti (%):").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(sim_frame, textvariable=self.fee_percent_var, width=10).grid(row=1, column=1, pady=2, padx=5,
                                                                               sticky=tk.W)

        # Stop Loss
        ttk.Label(sim_frame, text="Stop Loss (%):").grid(row=2, column=0, sticky=tk.W, pady=2)
        ttk.Entry(sim_frame, textvariable=self.stop_loss_percent_var, width=10).grid(row=2, column=1, pady=2, padx=5,
                                                                                     sticky=tk.W)

        # Take Profit
        ttk.Label(sim_frame, text="Take Profit (%):").grid(row=3, column=0, sticky=tk.W, pady=2)
        ttk.Entry(sim_frame, textvariable=self.take_profit_percent_var, width=10).grid(row=3, column=1, pady=2, padx=5,
                                                                                       sticky=tk.W)

        # Pozisyon Boyutu
        ttk.Label(sim_frame, text="Pozisyon Boyutu (%):").grid(row=4, column=0, sticky=tk.W, pady=2)
        ttk.Entry(sim_frame, textvariable=self.position_size_percent_var, width=10).grid(row=4, column=1, pady=2,
                                                                                         padx=5, sticky=tk.W)

        # Gerçek Zamanlı Simülasyon
        ttk.Checkbutton(sim_frame, text="Gerçek Zamanlı Simülasyon", variable=self.real_time_var).grid(row=5, column=0,
                                                                                                       columnspan=2,
                                                                                                       sticky=tk.W,
                                                                                                       pady=5)

        # Simülasyon Butonları
        sim_button_frame = ttk.Frame(left_panel)
        sim_button_frame.pack(fill=tk.X, pady=10)

        # Model işlemleri için butonlar
        self.train_button = ttk.Button(sim_button_frame, text="Model Eğit", command=self.train_model, state=tk.DISABLED)
        self.train_button.pack(side=tk.LEFT, padx=5)

        self.load_model_button = ttk.Button(sim_button_frame, text="Model Yükle", command=self.load_model)
        self.load_model_button.pack(side=tk.LEFT, padx=5)

        # İkinci buton satırı
        sim_button_frame2 = ttk.Frame(left_panel)
        sim_button_frame2.pack(fill=tk.X, pady=5)

        # Simülasyon butonları
        self.start_sim_button = ttk.Button(sim_button_frame2, text="Simülasyonu Başlat", command=self.start_simulation,
                                           state=tk.DISABLED)
        self.start_sim_button.pack(side=tk.LEFT, padx=5)

        self.stop_sim_button = ttk.Button(sim_button_frame2, text="Simülasyonu Durdur", command=self.stop_simulation,
                                          state=tk.DISABLED)
        self.stop_sim_button.pack(side=tk.LEFT, padx=5)

    def create_right_panel(self):
        """Sağ panel oluşturur (sonuçlar)"""
        right_panel = ttk.Frame(self)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # Grafik Alanı
        chart_frame = ttk.LabelFrame(right_panel, text="Fiyat ve Tahmin Grafiği", padding=10)
        chart_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Grafik için yer tutucu
        self.chart_placeholder = ttk.Label(chart_frame,
                                           text="Veri yükleyip model eğitince grafik burada görüntülenecek")
        self.chart_placeholder.pack(fill=tk.BOTH, expand=True)

        # İlerleme çerçevesi
        progress_frame = ttk.Frame(right_panel)
        progress_frame.pack(fill=tk.X, pady=5)

        # İlerleme çubuğu
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(progress_frame, variable=self.progress_var, length=100, mode='determinate')
        self.progress_bar.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        self.progress_label = ttk.Label(progress_frame, text="0%")
        self.progress_label.pack(side=tk.RIGHT, padx=5)

        # Sonuç özeti
        self.results_frame = ttk.LabelFrame(right_panel, text="Simülasyon Sonuçları", padding=10)
        self.results_frame.pack(fill=tk.X, pady=5)

        # Placeholder
        self.results_placeholder = ttk.Label(self.results_frame, text="Simülasyon sonuçları burada görüntülenecek")
        self.results_placeholder.pack(fill=tk.X)

    def fetch_data(self):
        """Veri çeker"""
        if not self.api_manager.connected:
            self.log_callback("API bağlantısı kurulmamış!", error=True)
            messagebox.showerror("Hata", "Önce API bağlantısını kurmanız gerekiyor!")
            return

        try:
            symbol = self.symbol_var.get().strip().upper()
            interval = self.interval_var.get()
            lookback_days = int(self.lookback_days_var.get())

            if not symbol:
                messagebox.showerror("Hata", "Lütfen bir sembol girin!")
                return

            self.log_callback(f"{symbol} için {interval} veri çekiliyor ({lookback_days} gün)...")
            self.status_callback("Veri çekiliyor...", "Aktif")

            # İlerleme çubuğunu sıfırla
            self.progress_var.set(0)
            self.progress_label.config(text="0%")

            # Veri çekme işlemi
            def fetch_data_thread():
                try:
                    df = self.data_fetcher.fetch_historical_data(
                        symbol=symbol,
                        interval=interval,
                        lookback_days=lookback_days
                    )

                    if df is not None and len(df) > 0:
                        self.current_df = df
                        self.root.after(0, lambda: self.on_data_fetched(df))
                    else:
                        self.root.after(0, lambda: self.log_callback(f"{symbol} için veri çekilemedi!", error=True))
                        self.root.after(0, lambda: messagebox.showerror("Hata", f"{symbol} için veri çekilemedi!"))
                except Exception as e:
                    self.root.after(0, lambda: self.log_callback(f"Veri çekilirken hata: {str(e)}", error=True))
                    self.root.after(0, lambda: messagebox.showerror("Hata", f"Veri çekilirken hata: {str(e)}"))
                finally:
                    self.root.after(0, lambda: self.status_callback("Hazır", "Aktif"))

            # Thread başlat
            thread = threading.Thread(target=fetch_data_thread)
            thread.daemon = True
            thread.start()

        except Exception as e:
            self.log_callback(f"Veri çekme parametrelerinde hata: {str(e)}", error=True)
            messagebox.showerror("Hata", f"Veri çekme parametrelerinde hata: {str(e)}")

    def on_data_fetched(self, df):
        """
        Veri çekildikten sonra çağrılır

        Args:
            df (pd.DataFrame): Çekilen veri
        """
        self.log_callback(f"Veri başarıyla çekildi: {len(df)} satır.")

        # İlerleme çubuğunu güncelle
        self.progress_var.set(100)
        self.progress_label.config(text="100%")

        # Basit grafik göster
        self.draw_price_chart(df)

        # Calculate Indicators butonunu etkinleştir
        self.train_button.config(state=tk.DISABLED)
        self.start_sim_button.config(state=tk.DISABLED)

    def draw_price_chart(self, df):
        """Fiyat grafiği çizer"""
        # Eski grafiği temizle
        for widget in self.chart_placeholder.winfo_children():
            widget.destroy()

        # Yeni grafik oluştur
        fig, ax = plt.subplots(figsize=(10, 5))
        df['close'].plot(ax=ax, title=f"{self.symbol_var.get()} Fiyat Grafiği", grid=True)
        plt.tight_layout()

        # Tkinter'a yerleştir
        canvas = FigureCanvasTkAgg(fig, master=self.chart_placeholder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def calculate_indicators(self):
        """Göstergeleri hesaplar"""
        if self.current_df is None:
            messagebox.showinfo("Bilgi", "Önce veri çekmelisiniz.")
            return

        try:
            # Seçilen göstergeleri al
            selected = [indicator for indicator, var in self.selected_indicators.items() if var.get()]

            if not selected:
                # Kullanıcıya bilgi ver
                if messagebox.askyesno("Gösterge Seçimi",
                                       "Hiç gösterge seçilmedi. Varsayılan göstergeleri kullanmak ister misiniz?"):
                    # Varsayılan göstergeler
                    selected = ['RSI_14', 'MACD', 'Bollinger_Bands_20_2', 'SMA_20', 'EMA_20']

                    # Gösterge değişkenlerini güncelle
                    for indicator in selected:
                        if indicator in self.selected_indicators:
                            self.selected_indicators[indicator].set(True)
                else:
                    return

            self.log_callback(f"Seçilen göstergeler hesaplanıyor: {', '.join(selected)}")
            self.status_callback("Göstergeler hesaplanıyor...", "Aktif")

            # İlerleme çubuğunu sıfırla
            self.progress_var.set(0)
            self.progress_label.config(text="0%")

            # Gösterge hesaplama işlemi
            def calculate_indicators_thread():
                try:
                    df_with_indicators = self.indicator_calculator.calculate_indicators(
                        self.current_df, selected
                    )

                    # Özel öznitelikler ekle
                    df_with_indicators = self.indicator_calculator.add_custom_features(df_with_indicators)

                    if df_with_indicators is not None:
                        self.current_df = df_with_indicators
                        self.root.after(0, lambda: self.on_indicators_calculated(df_with_indicators))
                    else:
                        self.root.after(0, lambda: self.log_callback("Göstergeler hesaplanamadı!", error=True))
                        self.root.after(0, lambda: messagebox.showerror("Hata", "Göstergeler hesaplanamadı!"))
                except Exception as e:
                    self.root.after(0,
                                    lambda: self.log_callback(f"Göstergeler hesaplanırken hata: {str(e)}", error=True))
                    self.root.after(0,
                                    lambda: messagebox.showerror("Hata", f"Göstergeler hesaplanırken hata: {str(e)}"))
                finally:
                    self.root.after(0, lambda: self.status_callback("Hazır", "Aktif"))
                    self.root.after(0, lambda: self.progress_var.set(100))
                    self.root.after(0, lambda: self.progress_label.config(text="100%"))

            # Thread başlat
            thread = threading.Thread(target=calculate_indicators_thread)
            thread.daemon = True
            thread.start()

        except Exception as e:
            self.log_callback(f"Gösterge hesaplama hatası: {str(e)}", error=True)
            messagebox.showerror("Hata", f"Gösterge hesaplama hatası: {str(e)}")

    def on_indicators_calculated(self, df):
        """
        Göstergeler hesaplandıktan sonra çağrılır

        Args:
            df (pd.DataFrame): Göstergelerle birlikte veri
        """
        self.log_callback(f"Göstergeler hesaplandı. Veri boyutu: {df.shape}")

        # İndikatörlerle birlikte grafik göster
        self.draw_indicators_chart(df)

        # Train Model butonunu etkinleştir
        self.train_button.config(state=tk.NORMAL)
        self.start_sim_button.config(state=tk.DISABLED)

    def draw_indicators_chart(self, df):
        """İndikatörlerle birlikte grafik çizer"""
        # Eski grafiği temizle
        for widget in self.chart_placeholder.winfo_children():
            widget.destroy()

        # Yeni grafik oluştur
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

        # Fiyat grafiği
        df['close'].plot(ax=ax1, title=f"{self.symbol_var.get()} Fiyat ve Göstergeler", grid=True)

        # İndikatör grafiği (örnek olarak RSI)
        if 'RSI_14' in df.columns:
            df['RSI_14'].plot(ax=ax2, title="RSI", grid=True)
            ax2.axhline(70, color='r', linestyle='--')
            ax2.axhline(30, color='g', linestyle='--')

        plt.tight_layout()

        # Tkinter'a yerleştir
        canvas = FigureCanvasTkAgg(fig, master=self.chart_placeholder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def train_model(self):
        """Modeli eğitir"""
        if self.current_df is None:
            messagebox.showinfo("Bilgi", "Önce veri çekip göstergeleri hesaplamalısınız.")
            return

        try:
            self.train_button.config(state=tk.DISABLED)
            self.log_callback("Model eğitimi başlatılıyor...")
            self.status_callback("Model eğitiliyor...", "Aktif")

            # İlerleme çubuğunu sıfırla
            self.progress_var.set(0)
            self.progress_label.config(text="0%")

            def train_model_thread():
                try:
                    # Önce model_manager'ın symbol ve interval değerlerini ayarla
                    self.model_manager.symbol = self.symbol_var.get()
                    self.model_manager.interval = self.interval_var.get()

                    # Veriyi hazırla
                    X_train, y_train, X_test, y_test, scaler = self.model_manager.prepare_data(self.current_df)

                    # Modeli oluştur
                    input_shape = (X_train.shape[1], X_train.shape[2])
                    self.model_manager.build_model(input_shape)

                    # Modeli eğit (doğru parametrelerle)
                    history = self.model_manager.train_model(
                        X_train,
                        y_train,
                        epochs=config.DEFAULT_MODEL_PARAMS['epochs'],
                        batch_size=config.DEFAULT_MODEL_PARAMS['batch_size'],
                        progress_callback=lambda current, total, loss, val_loss: self.update_progress("Model Eğitimi",
                                                                                                      current, total,
                                                                                                      f"Loss: {loss:.4f}, Val: {val_loss:.4f}"), )

                    # Modeli değerlendir
                    eval_results = self.model_manager.evaluate_model(X_test, y_test)

                    # Modeli kaydet
                    trained_model_path = self.model_manager.save_model()

                    if trained_model_path:
                        self.current_model_path = trained_model_path
                        self.root.after(0, lambda: self.on_model_trained(trained_model_path))
                    else:
                        self.root.after(0, lambda: self.log_callback("Model eğitilemedi!", error=True))
                        self.root.after(0, lambda: messagebox.showerror("Hata", "Model eğitilemedi!"))
                except Exception as e:
                    self.root.after(0, lambda: self.log_callback(f"Model eğitilirken hata: {str(e)}", error=True))
                    self.root.after(0, lambda: messagebox.showerror("Hata", f"Model eğitilirken hata: {str(e)}"))
                finally:
                    self.root.after(0, lambda: self.status_callback("Hazır", "Aktif"))
                    self.root.after(0, lambda: self.train_button.config(state=tk.NORMAL))

            # Thread başlat
            thread = threading.Thread(target=train_model_thread)
            thread.daemon = True
            thread.start()

        except Exception as e:
            self.log_callback(f"Model eğitimi başlatılırken hata: {str(e)}", error=True)
            messagebox.showerror("Hata", f"Model eğitimi başlatılırken hata: {str(e)}")
            self.train_button.config(state=tk.NORMAL)

    def on_model_trained(self, model_path):
        """
        Model eğitildikten sonra çağrılır

        Args:
            model_path (str): Eğitilen modelin kaydedildiği yol
        """
        self.log_callback(f"Model başarıyla eğitildi ve kaydedildi: {model_path}")
        messagebox.showinfo("Başarılı", "Model başarıyla eğitildi!")

        # Simülasyon butonunu etkinleştir
        self.start_sim_button.config(state=tk.NORMAL)

    def load_model(self, model_path):
        try:
            if not os.path.exists(model_path):
                logger.error(f"Model dosyası bulunamadı: {model_path}")
                return False

            self.model = keras.models.load_model(model_path, compile=False)
            self.model.compile(optimizer='adam', loss='mse')  # ✅ Bu satır tam burada olmalı
            self.is_trained = True

            logger.info(f"Model yüklendi: {model_path}")
            return True

        except Exception as e:
            logger.error(f"Model yüklenirken hata: {str(e)}")
            return False



        except Exception as e:
            self.log_callback(f"Model yüklenirken hata: {str(e)}", error=True)
            messagebox.showerror("Hata", f"Model yüklenirken hata: {str(e)}")

    def start_simulation(self):
        """Simülasyonu başlatır"""
        if self.current_model_path is None:
            messagebox.showinfo("Bilgi", "Önce veri çekip model yüklemelisiniz.")
            return

        try:
            # Parametreleri al
            params = {
                'initial_balance': float(self.initial_balance_var.get()),
                'fee_percent': float(self.fee_percent_var.get()) / 100,
                'stop_loss_percent': float(self.stop_loss_percent_var.get()) / 100,
                'take_profit_percent': float(self.take_profit_percent_var.get()) / 100,
                'position_size_percent': float(self.position_size_percent_var.get()) / 100,
                'real_time': self.real_time_var.get()
            }

            self.log_callback("Simülasyon başlatılıyor...")
            self.status_callback("Simülasyon çalışıyor...", "Aktif")
            self.is_simulation_running = True

            self.start_sim_button.config(state=tk.DISABLED)
            self.stop_sim_button.config(state=tk.NORMAL)

            def simulation_thread():
                try:
                    real_time_mode = params.pop("real_time", False)

                    if real_time_mode:
                        self.simulator.run_simulation(
                            data=self.symbol_var.get(),
                            model_path=self.current_model_path,
                            real_time=True,
                            progress_callback=self.update_progress,
                            log_callback=self.log_callback,
                            status_callback=self.status_callback,
                            **params
                        )
                    else:
                        if self.current_df is None:
                            raise ValueError("Simülasyon için veri yüklenmedi!")

                        results = self.simulator.run_simulation(
                            self.current_df,
                            self.current_model_path,
                            progress_callback=lambda current, total, loss: self.update_progress(
                                "Simülasyon", current, total, f"Loss: {loss:.4f}"
                            ),
                            **params
                        )

                        if results and self.is_simulation_running:
                            self.root.after(0, lambda: self.on_simulation_complete(results))
                        else:
                            self.root.after(0, lambda: self.log_callback("Simülasyon durduruldu!", error=True))
                except Exception as e:
                    error_msg = str(e)
                    self.root.after(0, lambda: self.log_callback(f"Simülasyon sırasında hata: {error_msg}", error=True))
                    self.root.after(0, lambda: messagebox.showerror("Hata", f"Simülasyon sırasında hata: {error_msg}"))
                finally:
                    self.root.after(0, lambda: self.status_callback("Hazır", "Aktif"))
                    self.root.after(0, lambda: self.start_sim_button.config(state=tk.NORMAL))
                    self.root.after(0, lambda: self.stop_sim_button.config(state=tk.DISABLED))
                    self.is_simulation_running = False

            thread = threading.Thread(target=simulation_thread)
            thread.daemon = True
            thread.start()

        except Exception as e:
            self.log_callback(f"Simülasyon başlatılırken hata: {str(e)}", error=True)
            messagebox.showerror("Hata", f"Simülasyon başlatılırken hata: {str(e)}")
            self.start_sim_button.config(state=tk.NORMAL)
            self.stop_sim_button.config(state=tk.DISABLED)
            self.is_simulation_running = False

    def stop_simulation(self):
        """Simülasyonu durdurur"""
        self.is_simulation_running = False
        self.simulator.stop_simulation()
        self.log_callback("Simülasyon durduruluyor...")
        self.status_callback("Simülasyon durduruluyor...", "Aktif")

    def on_simulation_complete(self, results):
        """
        Simülasyon tamamlandığında çağrılır

        Args:
            results (dict): Simülasyon sonuçları
        """
        self.log_callback("Simülasyon başarıyla tamamlandı!")
        messagebox.showinfo("Başarılı", "Simülasyon başarıyla tamamlandı!")

        # Sonuçları göster
        self.show_simulation_results(results)

    def show_simulation_results(self, results):
        """Simülasyon sonuçlarını gösterir"""
        # Placeholder'ı temizle
        self.results_placeholder.pack_forget()

        # Sonuçları göster
        if 'summary' in results:
            summary = results['summary']

            # Özet bilgileri göster
            result_text = f"""
            Toplam İşlem: {summary.get('total_trades', 0)}
            Başarılı İşlem: {summary.get('winning_trades', 0)}
            Başarısız İşlem: {summary.get('losing_trades', 0)}
            Başarı Oranı: {summary.get('win_rate', 0) * 100:.2f}%
            Toplam Kar/Zarar: ${summary.get('total_pnl', 0):.2f}
            Son Bakiye: ${summary.get('final_balance', 0):.2f}
            """

            result_label = ttk.Label(self.results_frame, text=result_text)
            result_label.pack(fill=tk.X, pady=5)

        # Grafik göster
        if 'results_df' in results:
            self.draw_simulation_results(results['results_df'])

    def draw_simulation_results(self, results_df):
        """Simülasyon sonuç grafiğini çizer"""
        # Eski grafiği temizle
        for widget in self.chart_placeholder.winfo_children():
            widget.destroy()

        # Yeni grafik oluştur
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={'height_ratios': [2, 1]})

        # Fiyat ve pozisyonlar
        if 'close' in results_df.columns:
            results_df['close'].plot(ax=ax1, title="Fiyat ve Pozisyonlar", grid=True)

        # Bakiye grafiği
        if 'balance' in results_df.columns:
            results_df['balance'].plot(ax=ax2, title="Bakiye Değişimi", grid=True)

        plt.tight_layout()

        # Tkinter'a yerleştir
        canvas = FigureCanvasTkAgg(fig, master=self.chart_placeholder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)


def update_progress(self, task_name, current, total, status=""):
    import inspect
    print("🧪 update_progress çağrıldı")
    print("📁 Bu fonksiyonun dosya yolu:", inspect.getfile(self.update_progress))
    print("🟢 TEST: update_progress fonksiyonu ÇALIŞTI")


"""
        Eğitim/Süreç ilerlemesini GUI’ye aktarır.
        """
try:
    percent = (current / total) * 100
    message = f"{task_name}: {current}/{total} - {status}"
    self._update_progress_gui(percent, message)
except Exception as e:
    print(f"HATA (update_progress): {e}")
