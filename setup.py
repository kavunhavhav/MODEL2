#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Binance LSTM Kripto Tahmin Botu - Proje Kurulum Scripti
Bu script, proje dosya yapısını oluşturur ve gerekli kütüphaneleri yükler.
"""

import os
import sys
import subprocess
import platform
import argparse
from pathlib import Path

# Oluşturulacak dosya yapısı
PROJECT_STRUCTURE = {
    "ana_klasörler": [
        "models",
        "data",
        "logs",
        "results"
    ],
    "modüller": [
        "config.py",
        "api_manager.py",
        "data_fetcher.py",
        "indicator_calculator.py",
        "model_manager.py",
        "simulator.py",
        "scanner.py",
        "utils.py",
        "main.py"
    ],
    "gui_klasörü": [
        "gui/__init__.py",
        "gui/main_window.py",
        "gui/settings_panel.py",
        "gui/scanner_panel.py",
        "gui/simulation_panel.py",
        "gui/results_panel.py"
    ]
}

# Gerekli kütüphaneler
REQUIREMENTS = [
    "numpy",
    "pandas",
    "matplotlib",
    "scipy",
    "scikit-learn",
    "tensorflow",
    "keras",
    "python-binance",
    "tqdm",
    "ta",
    "openpyxl",
    "xlsxwriter",
    "pillow"
]

# TA-Lib kurulumu için özel komutlar
TALIB_INSTALL = {
    "Windows": "pip install https://download.lfd.uci.edu/pythonlibs/archived/TA_Lib-0.4.24-cp39-cp39-win_amd64.whl",
    "Linux": "pip install TA-Lib",
    "Darwin": "pip install TA-Lib"  # macOS
}


def create_project_structure(base_path):
    """Proje dosya yapısını oluşturur"""
    print("Proje yapısı oluşturuluyor...")

    # Ana klasörleri oluştur
    for folder in PROJECT_STRUCTURE["ana_klasörler"]:
        folder_path = os.path.join(base_path, folder)
        os.makedirs(folder_path, exist_ok=True)
        print(f"✓ Klasör oluşturuldu: {folder_path}")

    # GUI klasörünü oluştur
    os.makedirs(os.path.join(base_path, "gui"), exist_ok=True)

    # Modül dosyalarını oluştur
    for module in PROJECT_STRUCTURE["modüller"]:
        module_path = os.path.join(base_path, module)
        if not os.path.exists(module_path):
            with open(module_path, 'w', encoding='utf-8') as f:
                f.write(
                    f'#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n\n"""\nBinance LSTM Kripto Tahmin Botu - {module}\n"""\n\n')
            print(f"✓ Modül oluşturuldu: {module_path}")

    # GUI modüllerini oluştur
    for gui_module in PROJECT_STRUCTURE["gui_klasörü"]:
        gui_module_path = os.path.join(base_path, gui_module)
        os.makedirs(os.path.dirname(gui_module_path), exist_ok=True)
        if not os.path.exists(gui_module_path):
            with open(gui_module_path, 'w', encoding='utf-8') as f:
                module_name = os.path.basename(gui_module)
                f.write(
                    f'#!/usr/bin/env python\n# -*- coding: utf-8 -*-\n\n"""\nBinance LSTM Kripto Tahmin Botu - GUI: {module_name}\n"""\n\n')
            print(f"✓ GUI modülü oluşturuldu: {gui_module_path}")

    print("\nProje yapısı başarıyla oluşturuldu!")


def setup_virtual_environment(base_path, env_name="venv"):
    """Sanal ortamı oluşturur ve aktifleştirir"""
    env_path = os.path.join(base_path, env_name)

    if os.path.exists(env_path):
        print(f"'{env_path}' zaten mevcut. Yeni sanal ortam oluşturulmayacak.")
        return env_path

    print(f"Sanal ortam oluşturuluyor: {env_path}")
    try:
        subprocess.run([sys.executable, "-m", "venv", env_path], check=True)
        print(f"✓ Sanal ortam başarıyla oluşturuldu: {env_path}")
        return env_path
    except subprocess.CalledProcessError as e:
        print(f"Sanal ortam oluşturulurken hata: {e}")
        sys.exit(1)


def install_requirements(env_path):
    """Gerekli kütüphaneleri yükler"""
    print("\nGerekli kütüphaneler yükleniyor...")

    # Sanal ortamdaki pip yolunu belirle
    if platform.system() == "Windows":
        pip_path = os.path.join(env_path, "Scripts", "pip.exe")
    else:
        pip_path = os.path.join(env_path, "bin", "pip")

    # pip'i güncelle
    try:
        subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        print("✓ pip güncellendi")
    except subprocess.CalledProcessError as e:
        print(f"pip güncellenirken hata: {e}")

    # Gereklilikleri yükle
    for req in REQUIREMENTS:
        try:
            subprocess.run([pip_path, "install", req], check=True)
            print(f"✓ {req} yüklendi")
        except subprocess.CalledProcessError as e:
            print(f"{req} yüklenirken hata: {e}")

    # TA-Lib kurulumu
    try:
        system = platform.system()
        if system in TALIB_INSTALL:
            talib_command = TALIB_INSTALL[system].split()
            if talib_command[0] == "pip":
                talib_command[0] = pip_path
            subprocess.run(talib_command, check=True)
            print("✓ TA-Lib yüklendi")
        else:
            print(f"⚠️ TA-Lib için {system} işletim sistemi desteklenmiyor. Manuel yükleme gerekebilir.")
    except subprocess.CalledProcessError as e:
        print(f"TA-Lib yüklenirken hata: {e}")
        print("⚠️ TA-Lib kurulumu başarısız oldu. Manuel kurulum gerekebilir.")

    print("\nKütüphane kurulumu tamamlandı!")


def create_activation_script(base_path, env_path):
    """Sanal ortamı aktifleştiren script oluşturur"""
    if platform.system() == "Windows":
        script_path = os.path.join(base_path, "activate.bat")
        with open(script_path, 'w') as f:
            f.write(f'@echo off\n')
            f.write(f'call "{os.path.join(env_path, "Scripts", "activate.bat")}"\n')
            f.write(f'echo Sanal ortam aktifleştirildi: {env_path}\n')
            f.write(f'cmd /k\n')
    else:
        script_path = os.path.join(base_path, "activate.sh")
        with open(script_path, 'w') as f:
            f.write(f'#!/bin/bash\n')
            f.write(f'source "{os.path.join(env_path, "bin", "activate")}"\n')
            f.write(f'echo "Sanal ortam aktifleştirildi: {env_path}"\n')

        # Dosyayı çalıştırılabilir yap
        os.chmod(script_path, 0o755)

    print(f"✓ Aktivasyon scripti oluşturuldu: {script_path}")


def create_requirements_file(base_path):
    """requirements.txt dosyası oluşturur"""
    req_path = os.path.join(base_path, "requirements.txt")
    with open(req_path, 'w') as f:
        for req in REQUIREMENTS:
            f.write(f"{req}\n")

    print(f"✓ requirements.txt dosyası oluşturuldu: {req_path}")


def main():
    parser = argparse.ArgumentParser(description="Binance LSTM Kripto Tahmin Botu kurulum scripti")
    parser.add_argument('--path', type=str, default=os.getcwd(),
                        help='Projenin kurulacağı yol (varsayılan: mevcut dizin)')
    parser.add_argument('--env-name', type=str, default='venv',
                        help='Sanal ortam adı (varsayılan: venv)')
    parser.add_argument('--skip-venv', action='store_true',
                        help='Sanal ortam kurulumunu atla')
    parser.add_argument('--skip-deps', action='store_true',
                        help='Bağımlılık kurulumunu atla')

    args = parser.parse_args()

    base_path = os.path.abspath(args.path)

    print(f"Binance LSTM Kripto Tahmin Botu kurulumu başlıyor...")
    print(f"Proje dizini: {base_path}")

    # Proje yapısını oluştur
    create_project_structure(base_path)

    # requirements.txt oluştur
    create_requirements_file(base_path)

    # Sanal ortam kurulumu
    if not args.skip_venv:
        env_path = setup_virtual_environment(base_path, args.env_name)
        create_activation_script(base_path, env_path)

        # Bağımlılıkları yükle
        if not args.skip_deps:
            install_requirements(env_path)

    print("\n=== Kurulum tamamlandı! ===")
    print(f"Projenin temel yapısı '{base_path}' dizininde oluşturuldu.")

    if not args.skip_venv:
        if platform.system() == "Windows":
            print("Sanal ortamı aktifleştirmek için 'activate.bat' dosyasını çalıştırın.")
        else:
            print("Sanal ortamı aktifleştirmek için 'source activate.sh' komutunu çalıştırın.")

    print("\nProjeyi başlatmak için:")
    print(f"cd {base_path}")
    if not args.skip_venv:
        if platform.system() == "Windows":
            print(f"{os.path.join(env_path, 'Scripts', 'python.exe')} main.py")
        else:
            print(f"{os.path.join(env_path, 'bin', 'python')} main.py")
    else:
        print("python main.py")


if __name__ == "__main__":
    main()