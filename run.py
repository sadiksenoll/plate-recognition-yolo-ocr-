#!/usr/bin/env python3
"""
Plaka Okuma Sistemi Başlatıcı
"""

import sys
import os

# Mevcut dizini Python path'ine ekle
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from plaka_okuma import main
    if __name__ == "__main__":
        main()
except ImportError as e:
    print(f"Import hatası: {e}")
    print("Lütfen requirements.txt'deki kütüphaneleri yükleyin:")
    print("pip install -r requirements.txt")
    print("\nNot: GPU desteği için CUDA uyumlu PyTorch kurun:")
    print("https://pytorch.org/get-started/locally/")
    sys.exit(1)
except Exception as e:
    print(f"Uygulama başlatılırken hata oluştu: {e}")
    sys.exit(1)
