# DocScan Pro — Akıllı Belge Tarama ve İyileştirme Sistemi

> Telefon kamerasıyla çekilmiş belgeleri; perspektif düzeltme, kontrast iyileştirme ve gürültü giderme adımlarıyla tarayıcı kalitesine yükselten, **sıfırdan** kodlanmış bir görüntü işleme uygulaması.

---

## Proje Hakkında

Bu proje, temel görüntü işleme algoritmalarının hiçbir hazır kütüphane fonksiyonu kullanılmadan, yalnızca **Python ve NumPy** ile sıfırdan implementasyonunu içermektedir. OpenCV yalnızca dosya okuma/yazma amacıyla kullanılmıştır.

| | |
|---|---|
| **Dil** | Python 3.10+ |
| **Arayüz** | CustomTkinter (koyu tema) |
| **Görüntü İşleme** | Tamamen Manuel — NumPy |
| **Akademik Kural** | `cv2.Canny`, `cv2.threshold`, `cv2.GaussianBlur` vb. **yasak** |

---

## Özellikler

### CamScanner Modu
- Görüntü yüklendiğinde 4 sürüklenebilir köşe noktası
- **Büyüteç**: köşeyi sürüklerken hizalama için gerçek zamanlı zoom
- Canny tabanlı **Otomatik Köşe Tespiti**
- **Perspektif Düzeltme** (Manuel DLT Homografi)
- **Tam İyileştirme**: Histogram Germe → Arka Plan Normalize → Sauvola Adaptif Eşikleme → Morfolojik Kapama

### Teknik İşlemler Paneli (15 Başlık)
Tüm parametreler slider ile anlık ayarlanabilir:

| # | Başlık | Fonksiyonlar |
|---|---|---|
| 1 | Temel Dönüşümler | Gri, Binary, RGB→HSV, RGB→LAB |
| 2 | Geometrik İşlemler | Döndürme, Ölçekleme (bilineer interpolasyon) |
| 3 | İstatistiksel & Kontrast | Histogram Hesap, Histogram Germe, Kontrast Azalt/Artır, Çıkar, Çarp |
| 4 | Filtreleme & Konvolüsyon | Mean, Median, Motion filtreler; genel 2D konvolüsyon |
| 5 | Gürültü Analizi | Salt & Pepper ekleme + Median ile temizleme |
| 6 | Segmentasyon & Kenar | Canny (Gaussian→Sobel→NMS→Histerezis), Çift Eşikleme |
| 7 | Morfolojik İşlemler | Dilation, Erosion, Opening, Closing |

### Uygulama Özellikleri
- **Geri Al** — 25 adım geçmişi
- **Karşılaştırma Modu** — Orijinal / İşlenmiş yan yana
- **JPG / PNG / PDF** çıktı
- Tüm işlemler `cikti/` klasörüne kaydedilebilir

---

## Kurulum

```bash
# 1. Repoyu klonla
git clone https://github.com/KULLANICI_ADI/docscan-pro.git
cd docscan-pro

# 2. Bağımlılıkları kur
pip install numpy opencv-python customtkinter Pillow matplotlib
```

### Gereksinimler
```
numpy>=1.24
opencv-python>=4.8
customtkinter>=5.2
Pillow>=10.0
matplotlib>=3.7
```

---

## Kullanım

### Masaüstü Uygulaması
```bash
python app.py
```

### Komut Satırı (Toplu İşleme)
```bash
python main.py --girdi belge.jpg --cikti cikti/
```

---

## Proje Yapısı

```
docscan-pro/
├── lib_goruntu.py      # Tüm görüntü işleme algoritmaları (sıfırdan NumPy)
│   ├── TemelDonusumler        — Gri, Binary, HSV, LAB
│   ├── GeometrikIslemler      — Döndürme, Kırpma, Ölçekleme
│   ├── KontrastIslemleri      — Histogram, Kontrast, Aritmetik
│   ├── FiltreIslemleri        — Konvolüsyon, Mean, Median, Motion
│   ├── GurultuAnalizi         — Salt & Pepper
│   ├── KenarBulma             — Canny, Çift Eşikleme
│   ├── MorfolojikIslemler     — Dilation, Erosion, Opening, Closing
│   ├── PerspektifDuzeltme     — DLT Homografi, Warp
│   ├── AdaptifEsikleme        — Sauvola (Integral Image)
│   └── GorunteIsleyici        — Merkez sınıf
│
├── app.py              # CustomTkinter masaüstü uygulaması
├── main.py             # Komut satırı akış dosyası
└── cikti/              # İşlenmiş görüntüler (otomatik oluşur)
```

---

## Algoritma Detayları

### Perspektif Düzeltme (Homografi)
4 nokta çiftinden **DLT (Direct Linear Transform)** ile 3×3 homografi matrisi hesaplanır. SVD çözümü ile elde edilen matris, ters eşleme + bilineer interpolasyon kullanılarak perspektif dönüşümü gerçekleştirir.

### Sauvola Adaptif Eşikleme
Her piksel için yerel ortalama (μ) ve standart sapma (σ) hesaplanarak dinamik eşik belirlenir:

```
T(i,j) = μ(i,j) · [1 + k · (σ(i,j)/R − 1)]
```

**Integral görüntü (SAT)** kullanılarak pencere boyutundan bağımsız O(N) karmaşıklıkta çalışır.

### Canny Kenar Tespiti
1. Gaussian pürüzsüzleştirme (manuel çekirdek)
2. Sobel gradyan hesabı (Gx, Gy)
3. Non-Maximum Suppression (4 yön)
4. Çift eşikleme + Histerezis bağlama

---

## Akademik Not

> Bu projede `cv2.Canny`, `cv2.threshold`, `cv2.GaussianBlur`, `cv2.dilate`, `cv2.erode`, `cv2.filter2D` ve benzeri tüm OpenCV görüntü işleme fonksiyonları **kullanılmamıştır**.
>
> Yalnızca `cv2.imread`, `cv2.imwrite`, `cv2.imshow` dosya I/O için kullanılmaktadır.
> Tüm algoritmalar Python + NumPy ile matris seviyesinde, sıfırdan kodlanmıştır.

---

## Lisans

MIT License — Akademik kullanım serbesttir.
