# DocScan Pro — Progress Tracker

## Status: FAZ 6 COMPLETE — TÜM FAZLAR BİTTİ

---

## DONE

### FAZ 1 ✅
- 1.1 Çift tıklama fix — `CTkScrollableFrame._parent_canvas takefocus=False`
- 1.2 Döndürme bbox genişletmesi — `dondur()` artık köşe kırpmıyor
- 1.3 Köşe sürükleme undo — `_surukle_baslat` → `_durum_kaydet()`
- 1.4 HEIC desteği — `pillow-heif` kuruldu, `goruntu_ac` PIL fallback

### FAZ 2 ✅
- 2.1 Tüm 11 slider → slider + entry iki yönlü bağlama (`_slider` helper yeniden yazıldı)
- 2.2 Canlı döndürme önizlemesi — `_canli_dondur_cb`, `_rot_base` taban mekanizması
- 2.3 Bölüm bilgi popup'ları — 7 bölüm + etiket açıklamaları, `_bilgi_goster`, `_section(bilgi=)`

### FAZ 3 ✅ (Performans)
| İşlem | Öncesi | Sonrası |
|---|---|---|
| Konvolüsyon 5×5 | ~45 sn | ~45 ms |
| Median filtre | ~30 sn | ~38 ms |
| Morfoloji | ~25 sn | ~38 ms |
| Canny | ~60 sn | ~106 ms |
| Histogram | ~2 sn | ~1 ms |

- `_konvolusyon_2d` → `sliding_window_view` + `np.einsum`
- `median_filtre` → `sliding_window_view` + `np.median(axis=(2,3))`
- `genisle/asin` → `sliding_window_view` + `.max/.min(axis=2)`
- `_nms` → vektörel 4-yön mask
- `cift_esikleme` → iteratif `np.maximum.reduce`
- `histogram_hesapla` → `np.bincount`
- `_isle_async` threading — ağır işlemler UI dondurmaz, ⏳ göstergesi

---

## TODO

### FAZ 4 ✅ — Otomatik Köşe Tespiti İyileştirme
- [x] **Otsu eşiği** + büyük blur ile parlak belge bölgesi tespiti (Canny değil — karmaşık arka plan desteği)
- [x] Morfolojik kapama (metin deliklerini doldur) + kenar marjı maskesi
- [x] Konveks Zarf (Andrew's monotone chain, NumPy)
- [x] Douglas-Peucker sadeleştirme → epsilon artırarak tam 4 köşeye indir
- [x] 4 köşeyi saat yönünde sırala (sol-üst, sağ-üst, sağ-alt, sol-alt)
- [x] Fallback: zarftan 4 uç nokta seç (xs±ys min/max)
- `lib_goruntu.KoseDetektoru` — `_otsu_esik`, `konveks_zarf`, `douglas_peucker`, `_dp_kapali`, `dort_kose_sec`, `kose_tespit`
- `app.oto_kose` → `_isle_async` ile arka planda çalışır, UI donmaz

### FAZ 5 ✅ — Davranış Düzeltmeleri
- [x] `_kanvasi_yenile` → `s = min(cw/W, ch/H)` (1.0 limiti kaldıldı, küçük görüntüler büyütülür)
- [x] `op_cikar / op_carp` → aynı boyutta ise renkli (kanal kanal), farklı boyutta ise gri fallback
- [x] Kontrast slider etiketi: `◀ azalt` (sarı) / `─ eşit ─` (soluk) / `artır ▶` (yeşil) — dinamik
- [x] `cikti/` klasörü → `os.path.dirname(os.path.abspath(__file__))` bazlı mutlak yol
- [x] `karsilastirma_ac` → `win.update()` sonrası direkt yükleme (`after(150)` hack kaldırıldı)
- [x] `main.py` → LAB çıktısı eklendi (adım 10/10), adım sayısı güncellendi

### FAZ 6 ✅ — UX İyileştirmeleri
- [x] Klavye kısayolları: Ctrl+Z, Ctrl+O, Ctrl+S (JPG), Ctrl+Shift+S (PDF), R (sıfırla) — entry odağında devredışı
- [x] Canvas scroll wheel zoom (fare merkezli, ×0.5–10) + orta tuş pan
- [x] Köşe drag hit-test → `tk scaling` ile DPI'a oranlı yarıçap
- [x] Drag & drop görüntü açma — `tkinterdnd2` (pip ile kuruldu, graceful fallback)
- [x] RGB kanalı bazlı histogram — renkli görüntüde 3 çizgi (R/G/B), gride tek çizgi
- Bonus: `_goruntu_yukle()` ortak yükleme metodu (diyalog + DnD paylaşır)
- Bonus: `_surukle_devam` `orijinal` → `mevcut` boyut bufixi

---

## KNOWN BUGS
_(tümü düzeltildi)_

- ✅ Arka arkaya döndürmede görüntü büyümesi — `op_dondur` sonrası slider 0'a sıfırlanır, `_rot_base` temizlenir
- ✅ `tam_iyilestir` Sauvola parametreleri — CamScanner bölümüne `pencere` (7–75) ve `k` (0.05–0.50) sliderlari eklendi
- ✅ Büyük median filtre performansı — `H×W×k×k > 80M` olunca tile'lı işleme devreye girer (512×512 tile)

---

## FILE MAP
```
app.py          — UI (CTk), tüm işlem methodları, _isle_async
lib_goruntu.py  — NumPy algoritmalar (sıfırdan, cv2 yasak)
main.py         — CLI pipeline (10 adım)
cikti/          — Çıktı klasörü (otomatik oluşur)
```

## RUN
```bash
python3 app.py
python3 main.py --girdi belge.jpg --cikti cikti/
```
