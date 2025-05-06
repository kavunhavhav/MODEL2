#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binance LSTM Kripto Tahmin Botu - Sonuçlar Paneli
"""

import os
import logging
import threading
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import config
import utils

logger = logging.getLogger(__name__)


class ResultsPanel(ttk.Frame):
    """Sonuçlar Paneli Sınıfı"""

    def __init__(self, parent, simulator, log_callback, status_callback):
        """
        Sonuçlar panelini başlatır

        Args:
            parent: Üst widget
            simulator: Simülasyon modülü
            log_callback: Log mesajı callback fonksiyonu
            status_callback: Durum güncelleme callback fonksiyonu
        """
        super().__init__(parent, padding=10)
        self.simulator = simulator
        self.log_callback = log_callback
        self.status_callback = status_callback
        self.root = parent  # Ana pencere referansını ekliyoruz

        # Son simülasyon sonuçları için değişkenler
        self.results_df = None
        self.positions_df = None
        self.summary = None
        self.canvas = None  # Grafik canvası için referans

        # GUI oluştur
        self.create_results_view()
        self.create_performance_view()

    def create_results_view(self):
        """Sonuç görüntüleme alanını oluşturur"""
        # Üst çerçeve (İşlem listesi)
        top_frame = ttk.LabelFrame(self, text="İşlem Listesi", padding=10)
        top_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # İşlem listesi için treeview
        columns = ("id", "sembol", "tip", "giris_zamani", "giris_fiyati",
                   "cikis_zamani", "cikis_fiyati", "kar_zarar",
                   "kar_zarar_yuzde", "durum")
        self.transactions_tree = ttk.Treeview(top_frame, columns=columns,
                                              show="headings", height=8)

        # Sütun başlıkları ve genişlikleri
        column_settings = [
            ("id", "ID", 40),
            ("sembol", "Sembol", 80),
            ("tip", "Tip", 60),
            ("giris_zamani", "Giriş Zamanı", 150),
            ("giris_fiyati", "Giriş Fiyatı", 100),
            ("cikis_zamani", "Çıkış Zamanı", 150),
            ("cikis_fiyati", "Çıkış Fiyatı", 100),
            ("kar_zarar", "Kar/Zarar", 100),
            ("kar_zarar_yuzde", "Kar/Zarar %", 100),
            ("durum", "Durum", 70)
        ]

        for col, heading, width in column_settings:
            self.transactions_tree.heading(col, text=heading)
            self.transactions_tree.column(col, width=width)

        # Scrollbar
        scrollbar = ttk.Scrollbar(top_frame, orient=tk.VERTICAL,
                                  command=self.transactions_tree.yview)
        self.transactions_tree.configure(yscrollcommand=scrollbar.set)

        # Widget yerleşimi
        self.transactions_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Alt çerçeve (Özet ve Butonlar)
        bottom_frame = ttk.Frame(self)
        bottom_frame.pack(fill=tk.X, pady=10)

        # Sonuç özeti
        summary_frame = ttk.LabelFrame(bottom_frame, text="Özet İstatistikler", padding=10)
        summary_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 5))

        # Özet etiketleri
        self.summary_labels = {}
        summary_fields = [
            ("Sembol:", "symbol", 0, 0),
            ("Aralık:", "interval", 0, 2),
            ("Toplam İşlem:", "total_trades", 1, 0),
            ("Başarı Oranı:", "win_rate", 1, 2),
            ("Son Bakiye:", "balance", 2, 0),
            ("Kar/Zarar:", "pnl", 2, 2)
        ]

        for label_text, key, row, col in summary_fields:
            ttk.Label(summary_frame, text=label_text).grid(
                row=row, column=col, sticky=tk.W, padx=5, pady=2)
            self.summary_labels[key] = ttk.Label(summary_frame, text="-")
            self.summary_labels[key].grid(
                row=row, column=col + 1, sticky=tk.W, padx=5, pady=2)

        # Butonlar çerçevesi
        button_frame = ttk.LabelFrame(bottom_frame, text="İşlemler", padding=10)
        button_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=(5, 0))

        # Butonlar
        buttons = [
            ("Excel'e Aktar", self.export_to_excel),
            ("Rapor Oluştur", self.create_report),
            ("Sonuçları Yükle", self.load_results),
            ("Karşılaştır", self.compare_results)
        ]

        for text, command in buttons:
            ttk.Button(button_frame, text=text, command=command).pack(
                anchor=tk.W, pady=2)

    def create_performance_view(self):
        """Performans görüntüleme alanını oluşturur"""
        performance_frame = ttk.LabelFrame(self, text="Performans Analizi", padding=10)
        performance_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Grafik için yer tutucu
        self.chart_placeholder = ttk.Label(
            performance_frame,
            text="Sonuçlar yüklendiğinde grafik burada görüntülenecek"
        )
        self.chart_placeholder.pack(fill=tk.BOTH, expand=True)

    def update_results(self, results_df=None, positions_df=None, summary=None):
        """
        Sonuç görünümünü günceller

        Args:
            results_df (pd.DataFrame): Sonuç verileri
            positions_df (pd.DataFrame): Pozisyon verileri
            summary (dict): Özet istatistikler
        """
        # Sonuçları kaydet
        self.results_df = results_df
        self.positions_df = positions_df
        self.summary = summary

        # İşlem listesini temizle
        self.transactions_tree.delete(*self.transactions_tree.get_children())

        # Pozisyonları ekle
        if positions_df is not None and not positions_df.empty:
            for i, row in positions_df.iterrows():
                # Zaman değerlerini formatlama
                entry_time = self._format_timestamp(row.get('entry_time'))
                exit_time = self._format_timestamp(row.get('exit_time'))

                # Renk ayarları için etiketler
                pnl = row.get('pnl', 0)
                tags = ('positive',) if pnl > 0 else ('negative',) if pnl < 0 else tuple()

                # Ağaca ekle
                self.transactions_tree.insert("", tk.END,
                                              values=(
                                                  row.get('id', i + 1),
                                                  row.get('symbol', ''),
                                                  row.get('type', '').upper(),
                                                  entry_time,
                                                  f"{row.get('entry_price', 0):.8f}",
                                                  exit_time,
                                                  f"{row.get('exit_price', 0):.8f}" if row.get('exit_price') else '',
                                                  f"{pnl:.2f}",
                                                  f"{row.get('pnl_percent', 0) * 100:.2f}%",
                                                  row.get('status', '').upper()
                                              ),
                                              tags=tags
                                              )

            # Kar/Zarar renklendirmesi
            self.transactions_tree.tag_configure('positive', foreground='green')
            self.transactions_tree.tag_configure('negative', foreground='red')

        # Özet bilgileri güncelle
        if summary is not None:
            symbol = getattr(getattr(self.simulator, 'model_manager', None), 'symbol', "-")
            interval = getattr(getattr(self.simulator, 'model_manager', None), 'interval', "-")

            self.summary_labels['symbol'].config(text=symbol)
            self.summary_labels['interval'].config(text=interval)
            self.summary_labels['total_trades'].config(text=str(summary.get('total_trades', 0)))

            # Başarı oranı
            win_rate = summary.get('win_rate', 0) * 100
            self.summary_labels['win_rate'].config(text=f"{win_rate:.2f}%")

            # Bakiye ve kar/zarar
            balance = getattr(self.simulator, 'balance', 0)
            initial_balance = getattr(self.simulator, 'initial_balance', balance)
            pnl = balance - initial_balance
            pnl_percent = (pnl / initial_balance) * 100 if initial_balance > 0 else 0

            self.summary_labels['balance'].config(text=f"${balance:.2f}")
            self.summary_labels['pnl'].config(text=f"${pnl:.2f} ({pnl_percent:+.2f}%)")

            # Renk ayarla
            color = "green" if pnl > 0 else "red" if pnl < 0 else "black"
            self.summary_labels['pnl'].config(foreground=color)

        # Performans grafiği çiz
        if results_df is not None and not results_df.empty:
            self.draw_performance_chart(results_df, positions_df)

    def _format_timestamp(self, timestamp):
        """Zaman damgasını formatlar"""
        if isinstance(timestamp, pd.Timestamp):
            return timestamp.strftime('%Y-%m-%d %H:%M:%S')
        return timestamp if timestamp else ''

    def draw_performance_chart(self, results_df, positions_df=None):
        """
        Performans grafiği çizer

        Args:
            results_df (pd.DataFrame): Sonuç verileri
            positions_df (pd.DataFrame): Pozisyon verileri
        """
        # Mevcut grafiği temizle
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.get_tk_widget().destroy()

        if hasattr(self, 'chart_placeholder'):
            self.chart_placeholder.pack_forget()

        # Figür oluştur
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                       gridspec_kw={'height_ratios': [3, 1]})

        # Üst grafik: Fiyat ve tahminler
        if 'predicted_close' in results_df.columns:
            ax1.plot(results_df.index, results_df['close'],
                     label='Gerçek Fiyat', color='blue')
            ax1.plot(results_df.index, results_df['predicted_close'],
                     label='Tahmin', color='red', linestyle='--')
        else:
            ax1.plot(results_df.index, results_df['close'],
                     label='Fiyat', color='blue')

        # Pozisyonları ekle
        if positions_df is not None and not positions_df.empty:
            for _, pos in positions_df.iterrows():
                if pos['status'] == 'closed':
                    entry_time = pos['entry_time']
                    exit_time = pos['exit_time']
                    entry_price = pos['entry_price']
                    exit_price = pos['exit_price']

                    # Pozisyon tipine göre renk
                    if pos['type'] == 'long':
                        marker_color, exit_color = 'green', 'red'
                    else:  # short
                        marker_color, exit_color = 'red', 'green'

                    # Giriş ve çıkış işaretleri
                    ax1.scatter(entry_time, entry_price, color=marker_color,
                                marker='^' if pos['type'] == 'long' else 'v',
                                s=100, zorder=5)
                    ax1.scatter(exit_time, exit_price, color=exit_color,
                                marker='o', s=80, zorder=5)

        # Alt grafik: Bakiye
        balance_data = results_df['balance'] if 'balance' in results_df.columns else \
            results_df['equity'] if 'equity' in results_df.columns else None

        if balance_data is not None:
            ax2.plot(results_df.index, balance_data, label='Bakiye', color='green')

        # Zaman formatlama
        date_format = '%Y-%m-%d %H:%M' if len(results_df) <= 100 else '%Y-%m-%d'
        ax1.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(date_format))
        ax2.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter(date_format))
        plt.xticks(rotation=45)

        # Grafik başlıkları ve ayarları
        symbol = getattr(getattr(self.simulator, 'model_manager', None), 'symbol', "")
        interval = getattr(getattr(self.simulator, 'model_manager', None), 'interval', "")

        ax1.set_title(f'{symbol} - {interval} (Performans Analizi)')
        ax1.set_ylabel('Fiyat')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        ax2.set_xlabel('Tarih')
        ax2.set_ylabel('Bakiye ($)')
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        # Figürü Tkinter'a yerleştir
        chart_frame = self.chart_placeholder.master
        self.canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def export_to_excel(self):
        """Sonuçları Excel'e aktarır"""
        if self.results_df is None or self.positions_df is None:
            messagebox.showinfo("Bilgi", "Dışa aktarılacak sonuç yok.")
            return

        try:
            # Dosya kaydet iletişim kutusu
            symbol = getattr(getattr(self.simulator, 'model_manager', None), 'symbol', "sonuclar")
            interval = getattr(getattr(self.simulator, 'model_manager', None), 'interval', "")

            filename = f"{symbol}_{interval}_sim_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
            filepath = filedialog.asksaveasfilename(
                title="Sonuçları Kaydet",
                initialfile=filename,
                defaultextension=".xlsx",
                filetypes=[("Excel Dosyası", "*.xlsx"), ("CSV Dosyası", "*.csv"), ("Tüm Dosyalar", "*.*")]
            )

            if not filepath:
                return

            self.log_callback(f"Sonuçlar dışa aktarılıyor: {filepath}")
            self.status_callback("Sonuçlar dışa aktarılıyor...", "Aktif")

            # Simülatörün export fonksiyonunu kullan
            self.simulator.export_results_to_excel(self.results_df, self.positions_df, filepath)

            self.log_callback(f"Sonuçlar kaydedildi: {filepath}")
            messagebox.showinfo("Başarılı", f"Sonuçlar başarıyla kaydedildi:\n{filepath}")
            self.status_callback("Hazır", "Aktif")

        except Exception as e:
            self.log_callback(f"Sonuçlar dışa aktarılırken hata: {str(e)}", error=True)
            messagebox.showerror("Hata", f"Sonuçlar dışa aktarılırken hata oluştu:\n{str(e)}")
            self.status_callback("Hazır", "Aktif")

    def create_report(self):
        """Rapor oluşturur"""
        if self.results_df is None or self.positions_df is None:
            messagebox.showinfo("Bilgi", "Rapor oluşturulacak sonuç yok.")
            return

        try:
            self.log_callback("Rapor oluşturuluyor...")
            self.status_callback("Rapor oluşturuluyor...", "Aktif")

            # Rapor oluşturma işlemi
            def create_report_thread():
                try:
                    html_content = utils.generate_trading_report(
                        results_df=self.results_df,
                        positions_df=self.positions_df,
                        summary_stats=self.summary
                    )

                    # HTML dosyasını kaydet
                    symbol = getattr(getattr(self.simulator, 'model_manager', None), 'symbol', "rapor")
                    interval = getattr(getattr(self.simulator, 'model_manager', None), 'interval', "")

                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    report_filename = f"{symbol}_{interval}_report_{timestamp}.html"
                    report_path = os.path.join(config.PATHS['results_dir'], report_filename)

                    with open(report_path, 'w', encoding='utf-8') as f:
                        f.write(html_content)

                    # Tarayıcıda aç
                    import webbrowser
                    webbrowser.open(f"file://{os.path.abspath(report_path)}")

                    self.root.after(0, lambda: self.log_callback(f"Rapor oluşturuldu ve açıldı: {report_path}"))
                    self.root.after(0, lambda: self.status_callback("Hazır", "Aktif"))
                except Exception as e:
                    self.root.after(0, lambda: self.log_callback(
                        f"Rapor oluşturulurken hata: {str(e)}", error=True))
                    self.root.after(0, lambda: messagebox.showerror(
                        "Hata", f"Rapor oluşturulurken hata: {str(e)}"))
                    self.root.after(0, lambda: self.status_callback("Hazır", "Aktif"))

            # Thread başlat
            thread = threading.Thread(target=create_report_thread)
            thread.daemon = True
            thread.start()

        except Exception as e:
            self.log_callback(f"Rapor oluşturma hazırlığı sırasında hata: {str(e)}", error=True)
            messagebox.showerror("Hata", f"Rapor oluşturma hazırlığı sırasında hata: {str(e)}")
            self.status_callback("Hazır", "Aktif")

    def load_results(self):
        """Önceden kaydedilmiş sonuçları yükler"""
        try:
            # Dosya seçme iletişim kutusu
            filepath = filedialog.askopenfilename(
                title="Sonuç Dosyası Seç",
                initialdir=config.PATHS['results_dir'],
                filetypes=[("Excel Dosyası", "*.xlsx"), ("CSV Dosyası", "*.csv"), ("Tüm Dosyalar", "*.*")]
            )

            if not filepath:
                return

            self.log_callback(f"Sonuçlar yükleniyor: {filepath}")
            self.status_callback("Sonuçlar yükleniyor...", "Aktif")

            # Excel dosyasını oku
            if filepath.endswith('.xlsx'):
                results_df = pd.read_excel(filepath, sheet_name=config.EXCEL_SETTINGS['sheet_names']['sonuclar'])

                # İşlemler sayfasını kontrol et
                try:
                    positions_df = pd.read_excel(filepath, sheet_name="İşlemler")
                except Exception:
                    positions_df = pd.DataFrame()  # Boş DataFrame

                # Özet sayfasını kontrol et
                try:
                    summary_df = pd.read_excel(filepath, sheet_name=config.EXCEL_SETTINGS['sheet_names']['ozet'])
                    summary = dict(zip(summary_df.iloc[:, 0], summary_df.iloc[:, 1]))
                except Exception:
                    summary = {}  # Boş dict

            elif filepath.endswith('.csv'):
                # CSV'den sadece sonuçları yükle
                results_df = pd.read_csv(filepath)
                positions_df = pd.DataFrame()
                summary = {}
            else:
                raise ValueError("Desteklenmeyen dosya formatı")

            # Tarih sütununu dönüştür
            if 'timestamp' in results_df.columns:
                results_df['timestamp'] = pd.to_datetime(results_df['timestamp'])
                results_df.set_index('timestamp', inplace=True)

            # İşlem tarihlerini dönüştür
            if not positions_df.empty:
                for col in ['entry_time', 'exit_time']:
                    if col in positions_df.columns:
                        positions_df[col] = pd.to_datetime(positions_df[col])

            # Sonuçları güncelle
            self.update_results(results_df, positions_df, summary)

            self.log_callback(f"Sonuçlar başarıyla yüklendi: {filepath}")
            self.status_callback("Hazır", "Aktif")

        except Exception as e:
            self.log_callback(f"Sonuçlar yüklenirken hata: {str(e)}", error=True)
            messagebox.showerror("Hata", f"Sonuçlar yüklenirken hata: {str(e)}")
            self.status_callback("Hazır", "Aktif")

    def compare_results(self):
        """Farklı sonuçları karşılaştırır"""
        if self.results_df is None:
            messagebox.showinfo("Bilgi", "Önce bir sonuç yüklemelisiniz.")
            return

        try:
            # Dosya seçme iletişim kutusu
            filepath = filedialog.askopenfilename(
                title="Karşılaştırılacak Sonuç Dosyası Seç",
                initialdir=config.PATHS['results_dir'],
                filetypes=[("Excel Dosyası", "*.xlsx"), ("CSV Dosyası", "*.csv"), ("Tüm Dosyalar", "*.*")]
            )

            if not filepath:
                return

            self.log_callback(f"Karşılaştırma için sonuçlar yükleniyor: {filepath}")

            # Excel dosyasını oku
            if filepath.endswith('.xlsx'):
                compare_df = pd.read_excel(filepath, sheet_name=config.EXCEL_SETTINGS['sheet_names']['sonuclar'])

                # Özet sayfasını kontrol et
                try:
                    summary_df = pd.read_excel(filepath, sheet_name=config.EXCEL_SETTINGS['sheet_names']['ozet'])
                    compare_summary = dict(zip(summary_df.iloc[:, 0], summary_df.iloc[:, 1]))
                except Exception:
                    compare_summary = {}

            elif filepath.endswith('.csv'):
                compare_df = pd.read_csv(filepath)
                compare_summary = {}
            else:
                raise ValueError("Desteklenmeyen dosya formatı")

            # Tarih sütununu dönüştür
            if 'timestamp' in compare_df.columns:
                compare_df['timestamp'] = pd.to_datetime(compare_df['timestamp'])
                compare_df.set_index('timestamp', inplace=True)

            # Karşılaştırma görselini oluştur
            self.draw_comparison_chart(self.results_df, compare_df, self.summary, compare_summary)

            self.log_callback("Karşılaştırma tamamlandı.")

        except Exception as e:
            self.log_callback(f"Karşılaştırma sırasında hata: {str(e)}", error=True)
            messagebox.showerror("Hata", f"Karşılaştırma sırasında hata: {str(e)}")

    def draw_comparison_chart(self, df1, df2, summary1=None, summary2=None):
        """
        Karşılaştırma grafiği çizer

        Args:
            df1 (pd.DataFrame): İlk sonuç verileri
            df2 (pd.DataFrame): İkinci sonuç verileri
            summary1 (dict): İlk özet istatistikler
            summary2 (dict): İkinci özet istatistikler
        """
        # Yeni pencere aç
        comparison_window = tk.Toplevel(self)
        comparison_window.title("Sonuç Karşılaştırması")
        comparison_window.geometry("900x700")
        comparison_window.minsize(900, 700)

        # Ana çerçeve
        main_frame = ttk.Frame(comparison_window, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Üst bilgi paneli
        info_frame = ttk.LabelFrame(main_frame, text="Karşılaştırma Özeti", padding=10)
        info_frame.pack(fill=tk.X, padx=5, pady=5)

        # Model bilgileri
        symbol1 = getattr(getattr(self.simulator, 'model_manager', None), 'symbol', "Model1")
        symbol2 = "Model2"

        # Özet bilgileri
        ttk.Label(info_frame, text="Model 1:").grid(row=0, column=0, sticky=tk.W, pady=2)
        ttk.Label(info_frame, text=symbol1).grid(row=0, column=1, sticky=tk.W, pady=2)

        ttk.Label(info_frame, text="Model 2:").grid(row=0, column=2, sticky=tk.W, pady=2)
        ttk.Label(info_frame, text=symbol2).grid(row=0, column=3, sticky=tk.W, pady=2)

        # Karşılaştırma sonuçları
        if summary1 and summary2:
            # Toplam işlem
            ttk.Label(info_frame, text="Toplam İşlem:").grid(row=1, column=0, sticky=tk.W, pady=2)
            total_trades1 = summary1.get('total_trades', 0)
            ttk.Label(info_frame, text=str(total_trades1)).grid(row=1, column=1, sticky=tk.W, pady=2)

            ttk.Label(info_frame, text="Toplam İşlem:").grid(row=1, column=2, sticky=tk.W, pady=2)
            total_trades2 = summary2.get('total_trades', 0)
            ttk.Label(info_frame, text=str(total_trades2)).grid(row=1, column=3, sticky=tk.W, pady=2)

            # Başarı oranı
            ttk.Label(info_frame, text="Başarı Oranı:").grid(row=2, column=0, sticky=tk.W, pady=2)
            win_rate1 = summary1.get('win_rate', 0) * 100
            ttk.Label(info_frame, text=f"{win_rate1:.2f}%").grid(row=2, column=1, sticky=tk.W, pady=2)

            ttk.Label(info_frame, text="Başarı Oranı:").grid(row=2, column=2, sticky=tk.W, pady=2)
            win_rate2 = summary2.get('win_rate', 0) * 100
            ttk.Label(info_frame, text=f"{win_rate2:.2f}%").grid(row=2, column=3, sticky=tk.W, pady=2)

            # Toplam kar/zarar
            ttk.Label(info_frame, text="Toplam Kar/Zarar:").grid(row=3, column=0, sticky=tk.W, pady=2)
            pnl1 = summary1.get('total_pnl', 0)
            pnl_label1 = ttk.Label(info_frame, text=f"${pnl1:.2f}")
            pnl_label1.grid(row=3, column=1, sticky=tk.W, pady=2)

            ttk.Label(info_frame, text="Toplam Kar/Zarar:").grid(row=3, column=2, sticky=tk.W, pady=2)
            pnl2 = summary2.get('total_pnl', 0)
            pnl_label2 = ttk.Label(info_frame, text=f"${pnl2:.2f}")
            pnl_label2.grid(row=3, column=3, sticky=tk.W, pady=2)

            # Renklendirme
            pnl_label1.configure(foreground="green" if pnl1 > 0 else "red" if pnl1 < 0 else "black")
            pnl_label2.configure(foreground="green" if pnl2 > 0 else "red" if pnl2 < 0 else "black")

        # Grafik paneli
        chart_frame = ttk.Frame(main_frame)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Grafik çiz
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8),
                                       gridspec_kw={'height_ratios': [1, 1]})

        # Üst grafik: Fiyat karşılaştırması
        if 'close' in df1.columns:
            ax1.plot(df1.index, df1['close'], label=f'{symbol1} Fiyat', color='blue')
        if 'close' in df2.columns:
            ax1.plot(df2.index, df2['close'], label=f'{symbol2} Fiyat', color='green')

        # Alt grafik: Bakiye karşılaştırması
        if 'balance' in df1.columns:
            ax2.plot(df1.index, df1['balance'], label=f'{symbol1} Bakiye', color='blue')
        elif 'equity' in df1.columns:
            ax2.plot(df1.index, df1['equity'], label=f'{symbol1} Bakiye', color='blue')

        if 'balance' in df2.columns:
            ax2.plot(df2.index, df2['balance'], label=f'{symbol2} Bakiye', color='green')
        elif 'equity' in df2.columns:
            ax2.plot(df2.index, df2['equity'], label=f'{symbol2} Bakiye', color='green')

        # Grafik ayarları
        ax1.set_title('Fiyat Karşılaştırması')
        ax1.set_ylabel('Fiyat')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left')

        ax2.set_title('Bakiye Karşılaştırması')
        ax2.set_xlabel('Tarih')
        ax2.set_ylabel('Bakiye ($)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')

        plt.tight_layout()

        # Figürü Tkinter'a yerleştir
        canvas = FigureCanvasTkAgg(fig, master=chart_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Butonlar
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)

        ttk.Button(button_frame, text="Kapat", command=comparison_window.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(button_frame, text="Grafiği Kaydet",
                   command=lambda: self.save_comparison_chart(fig)).pack(side=tk.RIGHT, padx=5)

    def save_comparison_chart(self, fig):
        """
        Karşılaştırma grafiğini kaydeder

        Args:
            fig: Matplotlib figürü
        """
        try:
            # Dosya kaydet iletişim kutusu
            filepath = filedialog.asksaveasfilename(
                title="Grafiği Kaydet",
                initialdir=config.PATHS['results_dir'],
                defaultextension=".png",
                filetypes=[("PNG Dosyası", "*.png"), ("PDF Dosyası", "*.pdf"), ("Tüm Dosyalar", "*.*")]
            )

            if not filepath:
                return

            # Grafiği kaydet
            fig.savefig(filepath, dpi=300, bbox_inches='tight')

            self.log_callback(f"Grafik kaydedildi: {filepath}")
            messagebox.showinfo("Başarılı", f"Grafik başarıyla kaydedildi:\n{filepath}")

        except Exception as e:
            self.log_callback(f"Grafik kaydedilirken hata: {str(e)}", error=True)
            messagebox.showerror("Hata", f"Grafik kaydedilirken hata: {str(e)}")