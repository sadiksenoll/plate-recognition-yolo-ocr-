# Plaka Okuma Sistemi

YOLO ve EasyOCR tabanlı akıllı plaka tanıma sistemi.

## Özellikler

- **Otomatik Plaka Tespiti**: YOLO modeli ile plaka alanlarını tespit etme
- **OCR ile Metin Okuma**: EasyOCR ile plaka metinlerini tanıma
- **Gerçek Zamanlı İşleme**: Webcam veya görüntü dosyaları üzerinde çalışma
- **Tkinter Arayüz**: Kullanıcı dostu grafik arayüz
- **GPU Desteği**: CUDA destekli sistemlerde hızlı işlem

## Gereksinimler

### Python Kütüphaneleri
```bash
pip install opencv-python ultralytics torch easyocr pillow numpy
```

### Sistem Gereksinimleri
- Python 3.8+
- Windows/Linux/macOS
- Webcam (isteğe bağlı)
- GPU (isteğe bağlı, daha hızlı işlem için)

## Kurulum

1. Python'u yükleyin (3.8 veya üzeri)
2. Gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```
3. Uygulamayı çalıştırın:
```bash
python plaka_okuma.py
```

## Kullanım

1. **Görüntü Seç**: Plaka okumak için görüntü dosyası seçin
2. **Kamera Başlat**: Webcam üzerinden gerçek zamanlı plaka okuma
3. **Sonuçları Gör**: Tespit edilen plakaları ve okunan metinleri görüntüle

## Teknolojiler

- **YOLO**: Nesne tespiti için
- **EasyOCR**: Metin tanıma için
- **OpenCV**: Görüntü işleme
- **Tkinter**: GUI framework
- **PyTorch**: Derin öğrenme altyapısı

## Model Bilgisi

- **YOLOv8**: Plaka tespiti için eğitilmiş model
- **EasyOCR**: 80+ dil desteği ile OCR

## Performans

- **CPU**: ~2-3 saniye/görüntü
- **GPU**: ~0.1-0.5 saniye/görüntü

## Lisans

MIT Lisansı

## Katkıda Bulunma

Pull request'ler bekleniyor!

## İletişim

Proje hakkında sorularınız için: *(iletişim bilgileri)*
