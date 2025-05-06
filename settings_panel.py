#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binance LSTM Kripto Tahmin Botu - Ayarlar Paneli
"""

import os
import json
import logging
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

import config

logger = logging.getLogger(__name__)


class SettingsPanel(ttk.Frame):
    """Ayarlar Paneli Sınıfı"""

    def __init__(self, parent, api_manager, log_callback, status_callback):
        """
        Ayarlar panelini başlatır

        Args:
            parent: Üst widget
            api_manager: API yöneticisi
            log_callback: Log mesajı callback fonksiyonu
            status_callback: Durum güncelleme callback fonksiyonu
        """
        super().__init__(parent, padding=10)
        self.api_manager = api_manager
        self.log_callback = log_callback
        self.status_callback = status_callback

        # Değişkenler
        self.api_key_var = tk.StringVar()
        self.api_secret_var = tk.StringVar()
        self.testnet_var = tk.BooleanVar(value=False)
        self.settings_file_path = os.path.join(config.PATHS['data_dir'], 'settings.json')

        # GUI oluştur
        self.create_api_settings()
        self.create_general_settings()

        # Ayarları yükle
        self.load_settings()

    def create_api_settings(self):
        """API ayarları panelini oluşturur"""
        api_frame = ttk.LabelFrame(self, text="Binance API Ayarları", padding=10)
        api_frame.pack(fill=tk.X, padx=5, pady=5)

        # API Key
        ttk.Label(api_frame, text="API Key:").grid(row=0, column=0, sticky=tk.W, pady=2)
        api_key_entry = ttk.Entry(api_frame, textvariable=self.api_key_var, width=50)
        api_key_entry.grid(row=0, column=1, pady=2, padx=5, sticky=tk.W)

        # API Secret
        ttk.Label(api_frame, text="API Secret:").grid(row=1, column=0, sticky=tk.W, pady=2)
        api_secret_entry = ttk.Entry(api_frame, textvariable=self.api_secret_var, width=50, show="*")
        api_secret_entry.grid(row=1, column=1, pady=2, padx=5, sticky=tk.W)

        # TestNet
        ttk.Checkbutton(api_frame, text="TestNet Kullan", variable=self.testnet_var).grid(row=2, column=0, columnspan=2,
                                                                                          sticky=tk.W, pady=2)

        # Bağlantı Butonu
        ttk.Button(api_frame, text="Bağlantı Kur", command=self.connect_api).grid(row=3, column=0, pady=10, padx=5,
                                                                                  sticky=tk.W)

        # Kaydetme Butonu
        ttk.Button(api_frame, text="Ayarları Kaydet", command=self.save_settings).grid(row=3, column=1, pady=10, padx=5,
                                                                                       sticky=tk.W)

        # API Bilgileri
        info_text = "NOT: API anahtarlarınızı güvenli bir şekilde saklayın ve sadece gerekli izinleri verin."
        ttk.Label(api_frame, text=info_text, wraplength=400, foreground="gray").grid(row=4, column=0, columnspan=2,
                                                                                     sticky=tk.W, pady=5)

    def create_general_settings(self):
        """Genel ayarlar panelini oluşturur"""
        general_frame = ttk.LabelFrame(self, text="Genel Ayarlar", padding=10)
        general_frame.pack(fill=tk.X, padx=5, pady=5)

        # Dosya Yolları
        ttk.Label(general_frame, text="Model Dizini:").grid(row=0, column=0, sticky=tk.W, pady=2)
        models_path_var = tk.StringVar(value=config.PATHS['models_dir'])
        models_path_entry = ttk.Entry(general_frame, textvariable=models_path_var, width=40)
        models_path_entry.grid(row=0, column=1, pady=2, padx=5, sticky=tk.W)
        ttk.Button(general_frame, text="Gözat", command=lambda: self.browse_directory(models_path_var)).grid(row=0,
                                                                                                             column=2,
                                                                                                             pady=2,
                                                                                                             padx=5)

        ttk.Label(general_frame, text="Veri Dizini:").grid(row=1, column=0, sticky=tk.W, pady=2)
        data_path_var = tk.StringVar(value=config.PATHS['data_dir'])
        data_path_entry = ttk.Entry(general_frame, textvariable=data_path_var, width=40)
        data_path_entry.grid(row=1, column=1, pady=2, padx=5, sticky=tk.W)
        ttk.Button(general_frame, text="Gözat", command=lambda: self.browse_directory(data_path_var)).grid(row=1,
                                                                                                           column=2,
                                                                                                           pady=2,
                                                                                                           padx=5)

        ttk.Label(general_frame, text="Sonuç Dizini:").grid(row=2, column=0, sticky=tk.W, pady=2)
        results_path_var = tk.StringVar(value=config.PATHS['results_dir'])
        results_path_entry = ttk.Entry(general_frame, textvariable=results_path_var, width=40)
        results_path_entry.grid(row=2, column=1, pady=2, padx=5, sticky=tk.W)
        ttk.Button(general_frame, text="Gözat", command=lambda: self.browse_directory(results_path_var)).grid(row=2,
                                                                                                              column=2,
                                                                                                              pady=2,
                                                                                                              padx=5)

        # Dizinleri Güncelleme Butonu
        ttk.Button(general_frame, text="Dizinleri Güncelle", command=lambda: self.update_paths(
            models_path_var.get(), data_path_var.get(), results_path_var.get()
        )).grid(row=3, column=0, columnspan=3, pady=10)

    def browse_directory(self, path_var):
        """
        Dizin seçme penceresi açar

        Args:
            path_var: Dizin yolu değişkeni
        """
        directory = filedialog.askdirectory(initialdir=path_var.get())
        if directory:
            path_var.set(directory)

    def update_paths(self, models_dir, data_dir, results_dir):
        """
        Dizin yollarını günceller

        Args:
            models_dir (str): Model dizini
            data_dir (str): Veri dizini
            results_dir (str): Sonuç dizini
        """
        # Dizinleri oluştur
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)

        # Config'i güncelle
        config.PATHS['models_dir'] = models_dir
        config.PATHS['data_dir'] = data_dir
        config.PATHS['results_dir'] = results_dir

        # Log
        self.log_callback("Dizin yolları güncellendi.")

        # Ayarları kaydet
        settings = {
            'api_key': self.api_key_var.get(),
            'api_secret': self.api_secret_var.get(),
            'testnet': self.testnet_var.get(),
            'paths': {
                'models_dir': models_dir,
                'data_dir': data_dir,
                'results_dir': results_dir
            }
        }

        self.save_settings_to_file(settings)

    def connect_api(self):
        """API bağlantısını kurar"""
        api_key = self.api_key_var.get().strip()
        api_secret = self.api_secret_var.get().strip()
        testnet = self.testnet_var.get()

        if not api_key or not api_secret:
            self.log_callback("API anahtarlarını girmelisiniz!", error=True)
            messagebox.showerror("Hata", "API anahtarlarını girmelisiniz!")
            return

        # API yöneticisini ayarla
        self.api_manager.api_key = api_key
        self.api_manager.api_secret = api_secret
        self.api_manager.testnet = testnet

        # Bağlantı kur
        self.log_callback(f"Binance API bağlantısı kuruluyor... ({'' if not testnet else 'TestNet'})")

        if self.api_manager.connect():
            self.log_callback("Binance API bağlantısı başarıyla kuruldu!")
            self.status_callback("Bağlantı kuruldu", "Aktif" if not testnet else "TestNet")
            messagebox.showinfo("Başarılı", "Binance API bağlantısı başarıyla kuruldu!")
        else:
            self.log_callback("Binance API bağlantısı kurulamadı!", error=True)
            self.status_callback("Bağlantı hatası", "Yok")
            messagebox.showerror("Hata", "Binance API bağlantısı kurulamadı! API anahtarlarınızı kontrol edin.")

    def save_settings(self):
        """Ayarları kaydeder"""
        settings = {
            'api_key': self.api_key_var.get(),
            'api_secret': self.api_secret_var.get(),
            'testnet': self.testnet_var.get(),
            'paths': config.PATHS
        }

        self.save_settings_to_file(settings)
        self.log_callback("Ayarlar kaydedildi.")
        messagebox.showinfo("Bilgi", "Ayarlar başarıyla kaydedildi.")

    def save_settings_to_file(self, settings):
        """
        Ayarları dosyaya kaydeder

        Args:
            settings (dict): Ayarlar
        """
        try:
            os.makedirs(os.path.dirname(self.settings_file_path), exist_ok=True)

            with open(self.settings_file_path, 'w', encoding='utf-8') as f:
                json.dump(settings, f, indent=4)
        except Exception as e:
            self.log_callback(f"Ayarlar kaydedilirken hata: {str(e)}", error=True)

    def load_settings(self):
        """Ayarları yükler"""
        try:
            if os.path.exists(self.settings_file_path):
                with open(self.settings_file_path, 'r', encoding='utf-8') as f:
                    settings = json.load(f)

                # API ayarları
                if 'api_key' in settings:
                    self.api_key_var.set(settings['api_key'])

                if 'api_secret' in settings:
                    self.api_secret_var.set(settings['api_secret'])

                if 'testnet' in settings:
                    self.testnet_var.set(settings['testnet'])

                # Dizin yolları
                if 'paths' in settings:
                    paths = settings['paths']

                    for key, value in paths.items():
                        if key in config.PATHS and os.path.exists(value):
                            config.PATHS[key] = value

                self.log_callback("Ayarlar yüklendi.")
        except Exception as e:
            self.log_callback(f"Ayarlar yüklenirken hata: {str(e)}", error=True)