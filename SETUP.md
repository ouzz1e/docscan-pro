# DocScan Pro — Kurulum ve Çalıştırma Rehberi

## Gereksinimler

- **Python 3.10+** (3.11 önerilir)
- **macOS / Windows / Linux** (macOS'ta test edilmiştir)

Python kurulu değilse → [python.org/downloads](https://www.python.org/downloads/)

---

## 1. Projeyi İndir

```bash
git clone <repo-url>
cd docscan-pro
```

Ya da ZIP olarak indirip klasörü aç.

---

## 2. Sanal Ortam Oluştur (önerilir)

```bash
python3 -m venv .venv
```

**macOS / Linux:**
```bash
source .venv/bin/activate
```

**Windows:**
```bash
.venv\Scripts\activate
```

---

## 3. Bağımlılıkları Kur

```bash
pip install customtkinter==5.2.2 \
            pillow==12.2.0 \
            opencv-python==4.13.0.92 \
            numpy==2.4.4 \
            pillow-heif==1.3.0 \
            tkinterdnd2==0.4.3 \
            matplotlib==3.10.9
```

Veya tek satırda:

```bash
pip install customtkinter pillow opencv-python numpy pillow-heif tkinterdnd2 matplotlib
```

---

## 4. Çalıştır

### Masaüstü Uygulaması (GUI)

```bash
python3 app.py
```

### Komut Satırı Pipeline (CLI)

```bash
python3 main.py --girdi belge.jpg --cikti cikti/
```

`cikti/` klasörü otomatik oluşturulur. İşlem çıktıları (gri, Canny, LAB vb.) oraya kaydedilir.

---

## Paket Açıklamaları

| Paket | Neden gerekli |
|---|---|
| `customtkinter` | Modern koyu tema UI |
| `pillow` | Görüntü okuma/yazma, HEIC dışı formatlar |
| `opencv-python` | `cv2.imread` / `cv2.imwrite` (yalnızca dosya I/O) |
| `numpy` | Tüm görüntü işleme algoritmaları |
| `pillow-heif` | iPhone HEIC/HEIF formatı desteği |
| `tkinterdnd2` | Canvas'a sürükle-bırak ile görüntü açma |
| `matplotlib` | RGB histogram grafiği (isteğe bağlı) |

> `matplotlib` kurulu değilse uygulama çalışır; yalnızca "Hist. Grafik" butonu hata mesajı gösterir.

---

## Klavye Kısayolları

| Kısayol | İşlem |
|---|---|
| `Ctrl+O` | Görüntü aç |
| `Ctrl+Z` | Geri al |
| `Ctrl+S` | JPG kaydet |
| `Ctrl+Shift+S` | PDF kaydet |
| `R` | Orijinal görüntüye sıfırla |

**Canvas:**
- **Fare tekerleği** → zoom in/out (fare konumuna odaklı)
- **Orta tuş sürükle** → pan (kaydır)
- **Sol tuş sürükle** → köşe noktası taşı

---

## Sık Karşılaşılan Sorunlar

### `ModuleNotFoundError: No module named 'customtkinter'`
Bağımlılıklar kurulmamış. Adım 3'ü tekrar çalıştır.

### macOS'ta `tkinter` bulunamıyor
```bash
brew install python-tk@3.11
```

### HEIC dosyası açılmıyor
`pillow-heif` kurulu olduğunu doğrula:
```bash
python3 -c "import pillow_heif; print('OK')"
```

### Windows'ta `tkinterdnd2` çalışmıyor
Sürükle-bırak olmadan uygulama yine de çalışır. Görüntü açmak için `Ctrl+O` veya toolbar butonu kullanılabilir.

---

## Dosya Yapısı

```
docscan-pro/
├── app.py          — Masaüstü uygulaması (GUI)
├── lib_goruntu.py  — Görüntü işleme algoritmaları (sıfırdan, cv2 yasak)
├── main.py         — CLI pipeline (10 adım)
├── SETUP.md        — Bu dosya
└── cikti/          — Çıktı klasörü (otomatik oluşur)
```
