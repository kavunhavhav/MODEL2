#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Yardımcı fonksiyonlar ve araçlar
"""

import os
import logging
import json
import time
import csv
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import date2num
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import matplotlib.ticker as mticker

import config

logger = logging.getLogger(__name__)


def setup_logging(log_dir=None, level=logging.INFO, console=True, file=True):
    """
    Log ayarlarını yapılandırır

    Args:
        log_dir (str): Log dizini
        level: Log seviyesi
        console (bool): Konsola log yazılsın mı
        file (bool): Dosyaya log yazılsın mı
    """
    if log_dir is None:
        log_dir = config.PATHS['logs_dir']

    os.makedirs(log_dir, exist_ok=True)

    # Root logger ayarları
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Tüm eski handlerleri temizle
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Konsol handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # Dosya handler
    if file:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(log_dir, f'crypto_bot_{timestamp}.log')
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    logger.info(f"Loglama ayarları yapılandırıldı. Seviye: {level}, Dosya: {file}")
    return root_logger


def create_timestamp_folders():
    """
    Zaman damgalı klasörler oluşturur

    Returns:
        dict: Oluşturulan klasör yolları
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    paths = {}
    for key, path in config.PATHS.items():
        timestamped_path = os.path.join(path, timestamp)
        os.makedirs(timestamped_path, exist_ok=True)
        paths[key] = timestamped_path

    logger.info(f"Zaman damgalı klasörler oluşturuldu: {timestamp}")
    return paths


def export_dataframe_to_csv(df, filename, directory=None):
    """
    DataFrame'i CSV dosyasına kaydeder

    Args:
        df (pd.DataFrame): Kaydedilecek veri
        filename (str): Dosya adı
        directory (str): Dizin

    Returns:
        str: Dosya yolu
    """
    if directory is None:
        directory = config.PATHS['data_dir']

    os.makedirs(directory, exist_ok=True)

    if not filename.endswith('.csv'):
        filename += '.csv'

    filepath = os.path.join(directory, filename)
    df.to_csv(filepath, index=True)

    logger.info(f"DataFrame CSV olarak kaydedildi: {filepath}")
    return filepath


def save_figure(fig, filename, directory=None, dpi=300, formats=None):
    """
    Matplotlib figürünü kaydeder

    Args:
        fig: Matplotlib figürü
        filename (str): Dosya adı
        directory (str): Dizin
        dpi (int): DPI değeri
        formats (list): Dosya formatları

    Returns:
        list: Kaydedilen dosya yolları
    """
    if directory is None:
        directory = config.PATHS['results_dir']

    os.makedirs(directory, exist_ok=True)

    if formats is None:
        formats = ['png', 'pdf']

    filepaths = []

    for fmt in formats:
        filepath = os.path.join(directory, f"{filename}.{fmt}")
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight')
        filepaths.append(filepath)

    logger.info(f"Grafik kaydedildi: {', '.join(filepaths)}")
    return filepaths


def create_candlestick_chart(df, title=None, volume=True, indicators=None, figsize=(12, 8)):
    """
    Mum grafiği oluşturur

    Args:
        df (pd.DataFrame): Veri çerçevesi
        title (str): Grafik başlığı
        volume (bool): Hacim gösterilsin mi
        indicators (dict): Gösterilecek göstergeler
        figsize (tuple): Grafik boyutu

    Returns:
        fig: Matplotlib figürü
    """
    df = df.copy()

    # Plot büyüklüğüne karar ver
    if volume:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    # Mum grafiği çiz
    width = 0.6
    width2 = 0.05

    # Renkler
    up = 'green'
    down = 'red'

    # Tarih formatı
    if len(df) > 100:
        date_format = '%Y-%m-%d'
    else:
        date_format = '%Y-%m-%d %H:%M'

    for idx, row in df.iterrows():
        if row['close'] >= row['open']:
            color = up
            body_start = row['open']
            body_height = row['close'] - row['open']
        else:
            color = down
            body_start = row['close']
            body_height = row['open'] - row['close']

        # OHLC işaretleri
        date_num = date2num(idx)
        rect = Rectangle(
            xy=(date_num - width / 2, body_start),
            width=width,
            height=body_height,
            facecolor=color,
            edgecolor='black',
            alpha=0.8
        )
        ax1.add_patch(rect)

        # Fitiller
        ax1.plot([date_num, date_num], [row['low'], min(row['open'], row['close'])], color='black', linewidth=1)
        ax1.plot([date_num, date_num], [max(row['open'], row['close']), row['high']], color='black', linewidth=1)

    # Eksen formatları
    ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Göstergeler
    if indicators:
        for name, values in indicators.items():
            if isinstance(values, pd.Series):
                values = values.values

            if 'SMA' in name or 'EMA' in name:
                ax1.plot(df.index, values, label=name)
            elif 'RSI' in name:
                # RSI eksenini ekle
                ax_rsi = ax1.twinx()
                ax_rsi.set_ylim(0, 100)
                ax_rsi.plot(df.index, values, label=name, color='purple', alpha=0.7)
                ax_rsi.axhline(y=70, color='r', linestyle='--', alpha=0.5)
                ax_rsi.axhline(y=30, color='g', linestyle='--', alpha=0.5)
                ax_rsi.set_ylabel('RSI')

                # RSI etiketini ekle
                lines, labels = ax_rsi.get_legend_handles_labels()
                ax1_lines, ax1_labels = ax1.get_legend_handles_labels()
                ax1.legend(ax1_lines + lines, ax1_labels + labels, loc='upper left')
            elif 'MACD' in name:
                # MACD eksenini ekle
                if 'ax_macd' not in locals():
                    ax_macd = ax1.twinx()
                    ax_macd.spines['right'].set_position(('outward', 60))
                    ax_macd.set_ylabel('MACD')

                ax_macd.plot(df.index, values, label=name, alpha=0.7)

                # MACD etiketini ekle
                if not 'ax_rsi' in locals():
                    lines, labels = ax_macd.get_legend_handles_labels()
                    ax1_lines, ax1_labels = ax1.get_legend_handles_labels()
                    ax1.legend(ax1_lines + lines, ax1_labels + labels, loc='upper left')

    # Hacim grafiği
    if volume:
        # Hacim çubuklarını çiz
        for idx, row in df.iterrows():
            date_num = date2num(idx)
            if row['close'] >= row['open']:
                color = up
            else:
                color = down

            rect = Rectangle(
                xy=(date_num - width / 2, 0),
                width=width,
                height=row['volume'],
                facecolor=color,
                edgecolor='black',
                alpha=0.5
            )
            ax2.add_patch(rect)

        ax2.set_ylabel('Hacim')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Grafik ayarları
    if title:
        ax1.set_title(title)

    ax1.set_ylabel('Fiyat')
    ax1.grid(True, alpha=0.3)

    # Grafikleri döndür
    plt.tight_layout()
    return fig


def create_performance_chart(df, balance_data=None, title=None, figsize=(12, 8)):
    """
    Performans grafiği oluşturur

    Args:
        df (pd.DataFrame): Fiyat verileri
        balance_data (pd.Series): Bakiye verileri
        title (str): Grafik başlığı
        figsize (tuple): Grafik boyutu

    Returns:
        fig: Matplotlib figürü
    """
    # Figür oluştur
    if balance_data is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, gridspec_kw={'height_ratios': [3, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=figsize)

    # Fiyat grafiği
    ax1.plot(df.index, df['close'], label='Fiyat', color='blue')

    if 'predicted_close' in df.columns:
        ax1.plot(df.index, df['predicted_close'], label='Tahmin', color='red', linestyle='--', alpha=0.7)

    # Eksen formatları
    if len(df) > 100:
        date_format = '%Y-%m-%d'
    else:
        date_format = '%Y-%m-%d %H:%M'

    ax1.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
    ax1.xaxis.set_major_locator(mdates.AutoDateLocator())

    # Grafik ayarları
    if title:
        ax1.set_title(title)
    else:
        ax1.set_title('Performans Grafiği')

    ax1.set_ylabel('Fiyat')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Bakiye grafiği
    if balance_data is not None:
        ax2.plot(balance_data.index, balance_data.values, label='Bakiye', color='green')
        ax2.set_ylabel('Bakiye ($)')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left')
        ax2.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())

    plt.tight_layout()
    return fig


def format_large_number(num):
    """
    Büyük sayıları formatlar (K, M, B)

    Args:
        num (float): Formatlanacak sayı

    Returns:
        str: Formatlanmış sayı
    """
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f}B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f}M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f}K"
    else:
        return f"{num:.2f}"


def get_unique_filename(base_name, extension, directory=None):
    """
    Benzersiz dosya adı oluşturur

    Args:
        base_name (str): Temel ad
        extension (str): Uzantı
        directory (str): Dizin

    Returns:
        str: Benzersiz dosya yolu
    """
    if directory is None:
        directory = config.PATHS['results_dir']

    os.makedirs(directory, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices('abcdefghijklmnopqrstuvwxyz0123456789', k=5))

    if not extension.startswith('.'):
        extension = '.' + extension

    filename = f"{base_name}_{timestamp}_{random_str}{extension}"
    filepath = os.path.join(directory, filename)

    return filepath


def calculate_drawdown(returns):
    """
    Maksimum drawdown hesaplar

    Args:
        returns (pd.Series): Getiri serisi

    Returns:
        tuple: (max_drawdown, max_drawdown_duration)
    """
    # Kümülatif getiri
    cum_returns = (1 + returns).cumprod()

    # Kümülatif maksimum
    running_max = cum_returns.cummax()

    # Drawdown serisi
    drawdown = (cum_returns / running_max - 1)

    # Maksimum drawdown
    max_drawdown = drawdown.min()

    # Maksimum drawdown süresi
    is_drawdown = drawdown < 0
    drawdown_periods = []
    current_period = 0

    for is_dd in is_drawdown:
        if is_dd:
            current_period += 1
        else:
            if current_period > 0:
                drawdown_periods.append(current_period)
                current_period = 0

    if current_period > 0:
        drawdown_periods.append(current_period)

    max_drawdown_duration = max(drawdown_periods) if drawdown_periods else 0

    return max_drawdown, max_drawdown_duration


def calculate_sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Sharpe oranı hesaplar

    Args:
        returns (pd.Series): Getiri serisi
        risk_free_rate (float): Risksiz faiz oranı
        periods_per_year (int): Yıl başına periyot sayısı

    Returns:
        float: Sharpe oranı
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    return np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()


def calculate_sortino_ratio(returns, risk_free_rate=0.0, periods_per_year=252):
    """
    Sortino oranı hesaplar

    Args:
        returns (pd.Series): Getiri serisi
        risk_free_rate (float): Risksiz faiz oranı
        periods_per_year (int): Yıl başına periyot sayısı

    Returns:
        float: Sortino oranı
    """
    excess_returns = returns - risk_free_rate / periods_per_year
    downside_returns = excess_returns[excess_returns < 0]

    if len(downside_returns) == 0 or downside_returns.std() == 0:
        return float('inf')

    return np.sqrt(periods_per_year) * excess_returns.mean() / downside_returns.std()


def generate_trading_report(results_df, positions_df, summary_stats=None, filename=None):
    """
    İşlem raporu oluşturur

    Args:
        results_df (pd.DataFrame): Sonuç verileri
        positions_df (pd.DataFrame): Pozisyon verileri
        summary_stats (dict): Özet istatistikler
        filename (str): Dosya adı

    Returns:
        str: Rapor HTML içeriği
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"trading_report_{timestamp}.html"

    filepath = os.path.join(config.PATHS['results_dir'], filename)

    # HTML içerik
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>İşlem Raporu</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; color: #333; }}
            h1, h2, h3 {{ color: #2c3e50; }}
            .container {{ max-width: 1200px; margin: 0 auto; }}
            .summary {{ background-color: #f9f9f9; padding: 15px; border-radius: 5px; margin-bottom: 20px; }}
            .stats {{ display: flex; flex-wrap: wrap; gap: 10px; }}
            .stat-card {{ background-color: #fff; border: 1px solid #ddd; border-radius: 5px; padding: 15px; width: calc(25% - 10px); box-sizing: border-box; }}
            .positive {{ color: green; }}
            .negative {{ color: red; }}
            table {{ border-collapse: collapse; width: 100%; margin-bottom: 20px; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            tr:hover {{ background-color: #f5f5f5; }}
            .chart {{ margin-bottom: 20px; max-width: 100%; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Kripto İşlem Raporu</h1>
            <p>Oluşturulma Tarihi: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <div class="summary">
                <h2>Özet</h2>
                <div class="stats">
    """

    # Özet istatistikler
    if summary_stats:
        for key, value in summary_stats.items():
            if key in ['total_pnl', 'balance']:
                css_class = "positive" if value > 0 else "negative"
                formatted_value = f"${value:.2f}"
            elif key in ['win_rate', 'total_pnl_percent']:
                css_class = "positive" if value > 0 else "negative"
                formatted_value = f"{value:.2f}%"
            elif isinstance(value, float):
                css_class = ""
                formatted_value = f"{value:.2f}"
            else:
                css_class = ""
                formatted_value = str(value)

            html_content += f"""
                    <div class="stat-card">
                        <h3>{key.replace('_', ' ').title()}</h3>
                        <p class="{css_class}">{formatted_value}</p>
                    </div>
            """

    html_content += """
                </div>
            </div>

            <h2>İşlemler</h2>
    """

    # Pozisyonlar tablosu
    if not positions_df.empty:
        html_content += """
            <table>
                <tr>
                    <th>ID</th>
                    <th>Sembol</th>
                    <th>Tip</th>
                    <th>Giriş Fiyatı</th>
                    <th>Çıkış Fiyatı</th>
                    <th>Miktar</th>
                    <th>Kar/Zarar</th>
                    <th>Kar/Zarar %</th>
                    <th>Durum</th>
                </tr>
        """

        for _, row in positions_df.iterrows():
            pnl = row.get('pnl', 0)
            pnl_pct = row.get('pnl_percent', 0) * 100
            pnl_class = "positive" if pnl > 0 else "negative" if pnl < 0 else ""

            html_content += f"""
                <tr>
                    <td>{row.get('id', '')}</td>
                    <td>{row.get('symbol', '')}</td>
                    <td>{row.get('type', '').upper()}</td>
                    <td>${row.get('entry_price', 0):.6f}</td>
                    <td>${row.get('exit_price', 0):.6f if row.get('exit_price') else ''}</td>
                    <td>${row.get('size', 0):.2f}</td>
                    <td class="{pnl_class}">${pnl:.2f}</td>
                    <td class="{pnl_class}">{pnl_pct:.2f}%</td>
                    <td>{row.get('status', '').upper()}</td>
                </tr>
            """

        html_content += """
            </table>
        """

    # Sonuçlar
    html_content += """
            <h2>Performans</h2>
    """

    if not results_df.empty:
        # Bakiye grafiği
        if 'balance' in results_df.columns:
            # HTML'de görüntülenmek üzere base64 formatında grafik oluştur
            import io
            import base64

            fig = plt.figure(figsize=(10, 6))
            plt.plot(results_df.index, results_df['balance'], label='Bakiye', color='green')
            plt.title('Bakiye Değişimi')
            plt.xlabel('Zaman')
            plt.ylabel('Bakiye ($)')
            plt.grid(True, alpha=0.3)
            plt.legend()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=72)
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            html_content += f"""
                <div class="chart">
                    <img src="data:image/png;base64,{img_data}" alt="Bakiye Grafiği" style="max-width:100%;">
                </div>
            """

        # Fiyat ve tahmin grafiği
        if all(col in results_df.columns for col in ['close', 'predicted_close']):
            fig = plt.figure(figsize=(10, 6))
            plt.plot(results_df.index, results_df['close'], label='Gerçek Fiyat', color='blue')
            plt.plot(results_df.index, results_df['predicted_close'], label='Tahmin', color='red', linestyle='--')
            plt.title('Fiyat ve Tahminler')
            plt.xlabel('Zaman')
            plt.ylabel('Fiyat')
            plt.grid(True, alpha=0.3)
            plt.legend()

            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=72)
            buf.seek(0)
            img_data = base64.b64encode(buf.read()).decode('utf-8')
            plt.close(fig)

            html_content += f"""
                <div class="chart">
                    <img src="data:image/png;base64,{img_data}" alt="Fiyat ve Tahmin Grafiği" style="max-width:100%;">
                </div>
            """

    html_content += """
        </div>
    </body>
    </html>
    """

    # HTML dosyasını kaydet
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"İşlem raporu oluşturuldu: {filepath}")
    return html_content


def clear_old_logs_and_data(max_age_days=30):
    """
    Eski log ve veri dosyalarını temizler

    Args:
        max_age_days (int): Maksimum dosya yaşı (gün)
    """
    for path_name, path in config.PATHS.items():
        if os.path.exists(path):
            cleaned = 0
            current_time = datetime.now()

            for root, dirs, files in os.walk(path):
                for file in files:
                    file_path = os.path.join(root, file)

                    # Dosya yaşını kontrol et
                    file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                    age = (current_time - file_time).days

                    if age > max_age_days:
                        try:
                            os.remove(file_path)
                            cleaned += 1
                        except Exception as e:
                            logger.error(f"Dosya silinemedi: {file_path} - {str(e)}")

            logger.info(f"{path_name} dizininden {cleaned} eski dosya temizlendi.")


def generate_random_key():
    """
    Rastgele API anahtarı oluşturur (test için)

    Returns:
        tuple: (api_key, api_secret)
    """
    import secrets

    api_key = ''.join(secrets.token_hex(16))
    api_secret = ''.join(secrets.token_hex(32))

    return api_key, api_secret


def parse_timeframe_to_minutes(timeframe):
    """
    Zaman dilimini dakika cinsine çevirir

    Args:
        timeframe (str): Zaman dilimi (örn. '1h', '30m')

    Returns:
        int: Dakika cinsinden süre
    """
    if timeframe.endswith('m'):
        return int(timeframe[:-1])
    elif timeframe.endswith('h'):
        return int(timeframe[:-1]) * 60
    elif timeframe.endswith('d'):
        return int(timeframe[:-1]) * 60 * 24
    elif timeframe.endswith('w'):
        return int(timeframe[:-1]) * 60 * 24 * 7
    else:
        raise ValueError(f"Bilinmeyen zaman dilimi formatı: {timeframe}")


def get_next_candle_time(interval, current_time=None):
    """
    Sonraki mum zamanını hesaplar

    Args:
        interval (str): Zaman aralığı
        current_time: Mevcut zaman

    Returns:
        datetime: Sonraki mum zamanı
    """
    if current_time is None:
        current_time = datetime.now()

    minutes = parse_timeframe_to_minutes(interval)

    if interval.endswith('m'):
        # Dakika başına ayarla
        next_time = current_time.replace(second=0, microsecond=0)
        # Sonraki aralığa kadar ekle
        remainder = next_time.minute % minutes
        if remainder > 0:
            next_time += timedelta(minutes=minutes - remainder)
        else:
            next_time += timedelta(minutes=minutes)

    elif interval.endswith('h'):
        # Saate ayarla
        next_time = current_time.replace(minute=0, second=0, microsecond=0)
        # Sonraki saate kadar ekle
        hours = int(interval[:-1])
        remainder = next_time.hour % hours
        if remainder > 0:
            next_time += timedelta(hours=hours - remainder)
        else:
            next_time += timedelta(hours=hours)

    elif interval.endswith('d'):
        # Günlük zaman aralığı genellikle UTC gece yarısında başlar
        next_time = (current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1))

    elif interval.endswith('w'):
        # Haftalık zaman aralığı genellikle Pazartesi UTC gece yarısında başlar
        days_to_monday = (7 - current_time.weekday()) % 7
        if days_to_monday == 0 and current_time.hour > 0:
            days_to_monday = 7

        next_time = (current_time.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=days_to_monday))

    else:
        raise ValueError(f"Desteklenmeyen zaman aralığı: {interval}")

    return next_time