#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binance LSTM Kripto Tahmin Botu - Ana Pencere
"""

import os
import time
import logging
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext

import config
from gui.settings_panel import SettingsPanel
from gui.scanner_panel import ScannerPanel
from gui.simulation_panel import SimulationPanel
from gui.results_panel import ResultsPanel

logger = logging.getLogger(__name__)


class MainWindow:
    """Ana Pencere Sınıfı"""

    def __init__(self, root, api_manager, data_fetcher, indicator_calculator, model_manager, simulator, scanner):
        """
        Ana pencereyi başlatır

        Args:
            root: Tkinter ana penceresi
            api_manager: API yöneticisi
            data_fetcher: Veri çekme modülü
            indicator_calculator: Gösterge hesaplama modülü
            model_manager: Model yönetim modülü
            simulator: Simülasyon modülü
            scanner: Tarama modülü
        """
        self.root = root
        self.api_manager = api_manager
        self.data_fetcher = data_fetcher
        self.indicator_calculator = indicator_calculator
        self.model_manager = model_manager
        self.simulator = simulator
        self.scanner = scanner

        # Simülatöre callback fonksiyonları ata
        self.simulator.log_callback = self.log_message
        self.simulator.progress_callback = None

        # Tarayıcıya callback fonksiyonları ata
        self.scanner.log_callback = self.log_message
        self.scanner.progress_callback = None

        # Pencere ayarları
        self.setup_window()

        # Log ve durum çubuğu
        self.create_log_area()
        self.create_status_bar()

        # Tab panelleri
        self.create_notebook()

        # Ana menü
        self.create_main_menu()

        # İlk log mesajı
        self.log_message("Uygulama başlatıldı.")

    def setup_window(self):
        """Pencere özelliklerini ayarlar"""
        title = config.GUI_SETTINGS['window_title']
        size = config.GUI_SETTINGS['window_size']
        min_size = config.GUI_SETTINGS['min_window_size']

        self.root.title(title)
        self.root.geometry(size)
        self.root.minsize(*map(int, min_size.split('x')))

        # Tema renklerini ayarla
        self.style = ttk.Style()

        # Renk paleti
        bg_color = '#f0f0f0'  # Açık gri arkaplan
        fg_color = '#333333'  # Koyu gri yazı
        accent_color = '#2c7be5'  # Mavi aksan rengi

        self.root.configure(bg=bg_color)

        # Temel stiller
        self.style.configure(".",
                             font=('Segoe UI', 10),
                             background=bg_color,
                             foreground=fg_color)

        # Pencere kapatıldığında onay iste
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_log_area(self):
        """Log alanını oluşturur"""
        log_frame = ttk.LabelFrame(self.root, text="Log", padding=10)
        log_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=80, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.log_text.config(state=tk.DISABLED)

    def create_status_bar(self):
        """Durum çubuğunu oluşturur"""
        self.status_bar = ttk.Frame(self.root)
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_label = ttk.Label(self.status_bar, text="Hazır", padding=5)
        self.status_label.pack(side=tk.LEFT)

        self.connection_label = ttk.Label(self.status_bar, text="Bağlantı: Yok", padding=5)
        self.connection_label.pack(side=tk.RIGHT)

    def create_notebook(self):
        """Tab panellerini oluşturur"""
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Ayarlar sekmesi
        self.settings_panel = SettingsPanel(
            self.notebook,
            self.api_manager,
            self.log_message,
            self.update_status
        )
        self.notebook.add(self.settings_panel, text="Ayarlar")

        # Tarama sekmesi
        self.scanner_panel = ScannerPanel(
            self.notebook,
            self.scanner,
            self.api_manager,
            self.data_fetcher,
            self.log_message,
            self.update_status
        )
        self.notebook.add(self.scanner_panel, text="Coin Tarama")

        # Simülasyon sekmesi
        self.simulation_panel = SimulationPanel(
            self.notebook,
            self.simulator,
            self.api_manager,
            self.data_fetcher,
            self.model_manager,
            self.indicator_calculator,
            self.log_message,
            self.update_status
        )
        self.notebook.add(self.simulation_panel, text="Simülasyon")

        # Sonuçlar sekmesi
        self.results_panel = ResultsPanel(
            self.notebook,
            self.simulator,
            self.log_message,
            self.update_status
        )
        self.notebook.add(self.results_panel, text="Sonuçlar")

    def create_main_menu(self):
        """Ana menüyü oluşturur"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # Dosya menüsü
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Dosya", menu=file_menu)

        file_menu.add_command(label="Rapor Oluştur", command=self.create_report)
        file_menu.add_command(label="Eski Dosyaları Temizle", command=self.clean_old_files)
        file_menu.add_separator()
        file_menu.add_command(label="Çıkış", command=self.on_closing)

        # Bağlantı menüsü
        connection_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Bağlantı", menu=connection_menu)

        connection_menu.add_command(label="API Bağlantısı Kur", command=self.connect_api)
        connection_menu.add_command(label="Test API Sorgula", command=self.test_api_query)
        connection_menu.add_command(label="Bakiye Sorgula", command=self.query_balance)

        # Araçlar menüsü
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Araçlar", menu=tools_menu)

        tools_menu.add_command(label="Versiyon", command=self.show_version)
        tools_menu.add_command(label="Log Temizle", command=self.clear_log)
        tools_menu.add_separator()
        tools_menu.add_command(label="Test Modu Aç", command=self.enable_test_mode)

    def log_message(self, message, error=False):
        """
        Log mesajı ekler

        Args:
            message (str): Log mesajı
            error (bool): Hata mı
        """

        def _add_log():
            self.log_text.config(state=tk.NORMAL)
            timestamp = datetime.now().strftime("%H:%M:%S")

            if error:
                self.log_text.insert(tk.END, f"[{timestamp}] HATA: {message}\n", "error")
                self.log_text.tag_configure("error", foreground="red")
            else:
                self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")

            self.log_text.see(tk.END)
            self.log_text.config(state=tk.DISABLED)

        self.root.after(0, _add_log)

    def update_status(self, message, connection_status=None):
        """
        Durum çubuğunu günceller

        Args:
            message (str): Durum mesajı
            connection_status (str): Bağlantı durumu
        """

        def _update():
            self.status_label.config(text=message)
            if connection_status is not None:
                self.connection_label.config(text=f"Bağlantı: {connection_status}")

        self.root.after(0, _update)

    def on_closing(self):
        """Uygulama kapatılırken"""
        if messagebox.askokcancel("Çıkış", "Uygulamadan çıkmak istiyor musunuz?"):
            if hasattr(self.simulator, 'is_running') and self.simulator.is_running:
                self.simulator.stop_simulation()
            if hasattr(self.scanner, 'is_scanning') and self.scanner.is_scanning:
                self.scanner.stop_scanning()
            self.root.destroy()

    def create_report(self):
        """Rapor oluşturur"""
        self.log_message("Rapor oluşturuluyor...")
        if hasattr(self, 'results_panel'):
            self.results_panel.create_report()

    def clean_old_files(self):
        """Eski dosyaları temizler"""
        if messagebox.askokcancel("Temizlik", "30 günden eski tüm dosyalar silinecek. Devam etmek istiyor musunuz?"):
            self.log_message("Eski dosyalar temizleniyor...")

            def run_cleanup():
                import utils
                utils.clear_old_logs_and_data()
                self.log_message("Eski dosyalar temizlendi.")

            thread = threading.Thread(target=run_cleanup)
            thread.daemon = True
            thread.start()

    def connect_api(self):
        """API bağlantısını kurar"""
        self.log_message("API bağlantısı kuruluyor...")
        if hasattr(self, 'settings_panel'):
            self.notebook.select(0)  # Settings sekmesi
            self.settings_panel.connect_api()

    def test_api_query(self):
        """API sorgusu test eder"""
        self.log_message("API sorgusu test ediliyor...")
        if not self.api_manager.connected:
            self.log_message("API bağlantısı kurulmamış!", error=True)
            messagebox.showerror("Hata", "İlk önce API bağlantısı kurmanız gerekiyor!")
            return
        try:
            ticker = self.api_manager.get_ticker_data('BTCUSDT')
            if ticker:
                price = float(ticker['lastPrice'])
                volume = float(ticker['quoteVolume'])
                change = float(ticker['priceChangePercent'])
                message = f"BTCUSDT: ${price:.2f} ({change:+.2f}%) - 24s Hacim: ${volume:.2f}"
                self.log_message(f"API sorgusu başarılı! {message}")
                messagebox.showinfo("Başarılı", f"API sorgusu başarılı!\n{message}")
            else:
                self.log_message("API sorgusu başarısız!", error=True)
                messagebox.showerror("Hata", "API sorgusu başarısız!")
        except Exception as e:
            self.log_message(f"API sorgusu sırasında hata: {str(e)}", error=True)
            messagebox.showerror("Hata", f"API sorgusu sırasında hata: {str(e)}")

    def query_balance(self):
        """Bakiye sorgular"""
        self.log_message("Bakiye sorgulanıyor...")
        if not self.api_manager.connected:
            self.log_message("API bağlantısı kurulmamış!", error=True)
            messagebox.showerror("Hata", "İlk önce API bağlantısı kurmanız gerekiyor!")
            return
        try:
            balances = self.api_manager.get_balances()
            if balances:
                balance_text = "Bakiyeler:\n\n"
                for i, balance in enumerate(balances[:10], 1):
                    asset = balance['asset']
                    free = balance['free']
                    locked = balance['locked']
                    total = balance['total']
                    balance_text += f"{i}. {asset}: {total:.8f} (Serbest: {free:.8f}, Kilitli: {locked:.8f})\n"
                if len(balances) > 10:
                    balance_text += f"\n... ve {len(balances) - 10} daha fazla varlık."
                self.log_message(f"Bakiye sorgusu başarılı! {len(balances)} varlık bulundu.")
                messagebox.showinfo("Bakiyeler", balance_text)
            else:
                self.log_message("Bakiye sorgusu başarısız!", error=True)
                messagebox.showerror("Hata", "Bakiye sorgusu başarısız!")
        except Exception as e:
            self.log_message(f"Bakiye sorgusu sırasında hata: {str(e)}", error=True)
            messagebox.showerror("Hata", f"Bakiye sorgusu sırasında hata: {str(e)}")

    def show_version(self):
        """Versiyon bilgisini gösterir"""
        version = "1.0.0"
        build_date = "2025-04-25"
        info_text = f"""
        Binance LSTM Kripto Tahmin Botu

        Versiyon: {version}
        Build Tarihi: {build_date}

        Geliştirici: AI Tüccar
        """
        messagebox.showinfo("Versiyon Bilgisi", info_text)

    def clear_log(self):
        """Log alanını temizler"""
        self.log_text.config(state=tk.NORMAL)
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state=tk.DISABLED)
        self.log_message("Log temizlendi.")

    def enable_test_mode(self):
        """Test modunu etkinleştirir"""
        if messagebox.askokcancel("Test Modu",
                                  "Test modu etkinleştirilecek. Bu mod gerçek emirler göndermez. Devam etmek istiyor musunuz?"):
            import utils
            api_key, api_secret = utils.generate_random_key()
            if hasattr(self, 'settings_panel'):
                self.settings_panel.api_key_var.set(api_key)
                self.settings_panel.api_secret_var.set(api_secret)
                self.settings_panel.testnet_var.set(True)
            self.log_message("Test modu etkinleştirildi. Rastgele API anahtarları oluşturuldu.")
            self.update_status("Test Modu Aktif", "TestNet")