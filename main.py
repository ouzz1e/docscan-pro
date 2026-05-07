"""
Akıllı Belge Tarama ve İyileştirme Sistemi
Ana akış dosyası

Kullanım:
    python main.py --girdi belge.jpg --cikti iyilestirilmis.jpg

Tüm görüntü işleme adımları lib_goruntu.py'deki sınıflar aracılığıyla
gerçekleştirilir. cv2 yalnızca dosya I/O için kullanılmaktadır.
"""

import argparse
import os
import cv2
import numpy as np

from lib_goruntu import GorunteIsleyici


def goruntu_oku(yol: str) -> np.ndarray:
    goruntu = cv2.imread(yol)
    if goruntu is None:
        raise FileNotFoundError(f"Görüntü okunamadı: {yol}")
    # cv2 BGR döndürür → RGB'ye çevir
    return goruntu[:, :, ::-1].copy()


def goruntu_kaydet(goruntu: np.ndarray, yol: str) -> None:
    # RGB → BGR
    cv2.imwrite(yol, goruntu[:, :, ::-1])
    print(f"Kaydedildi: {yol}")


def goruntu_goster(baslik: str, goruntu: np.ndarray) -> None:
    if goruntu.ndim == 2:
        cv2.imshow(baslik, goruntu)
    else:
        cv2.imshow(baslik, goruntu[:, :, ::-1])
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def belge_iyilestir(girdi_yolu: str, cikti_klasoru: str = "cikti") -> None:
    os.makedirs(cikti_klasoru, exist_ok=True)
    gi = GorunteIsleyici()

    # ── 1. Ham görüntüyü oku ────────────────────────────────────────────────
    print("[1/9] Görüntü okunuyor...")
    orijinal = goruntu_oku(girdi_yolu)
    goruntu_kaydet(orijinal, os.path.join(cikti_klasoru, "00_orijinal.jpg"))

    # ── 2. Gri dönüşüm ──────────────────────────────────────────────────────
    print("[2/9] Gri dönüşüm uygulanıyor...")
    gri = gi.temel.griye_donustur(orijinal)
    goruntu_kaydet(np.stack([gri, gri, gri], axis=2),
                   os.path.join(cikti_klasoru, "01_gri.jpg"))

    # ── 3. Histogram germe (kontrast iyileştirme) ────────────────────────────
    print("[3/9] Histogram germe uygulanıyor...")
    gerilmis = gi.kontrast.histogram_ger(gri)
    goruntu_kaydet(np.stack([gerilmis] * 3, axis=2),
                   os.path.join(cikti_klasoru, "02_histogram_gerilmis.jpg"))

    # ── 4. Salt & Pepper gürültüsü ekle ve temizle ──────────────────────────
    print("[4/9] Gürültü ekleniyor ve temizleniyor...")
    gurultulu = gi.gurultu.salt_pepper_ekle(gerilmis, yogunluk=0.04, tohum=42)
    goruntu_kaydet(np.stack([gurultulu] * 3, axis=2),
                   os.path.join(cikti_klasoru, "03_salt_pepper.jpg"))
    temizlenmis = gi.gurultu.salt_pepper_temizle(gurultulu, boyut=3)
    goruntu_kaydet(np.stack([temizlenmis] * 3, axis=2),
                   os.path.join(cikti_klasoru, "04_gurultu_temizlenmis.jpg"))

    # ── 5. Mean filtre (hafif bulanıklaştırma) ──────────────────────────────
    print("[5/9] Mean filtre uygulanıyor...")
    yumusatilmis = gi.filtre.mean_filtre(temizlenmis, boyut=3)
    goruntu_kaydet(np.stack([yumusatilmis] * 3, axis=2),
                   os.path.join(cikti_klasoru, "05_mean_filtre.jpg"))

    # ── 6. Canny kenar tespiti ───────────────────────────────────────────────
    print("[6/9] Canny kenar tespiti yapılıyor...")
    kenarlar = gi.kenar.canny_kenar(yumusatilmis, dusuk_esik=40, yuksek_esik=120)
    goruntu_kaydet(np.stack([kenarlar] * 3, axis=2),
                   os.path.join(cikti_klasoru, "06_canny_kenar.jpg"))

    # ── 7. Morfolojik kapama (kenarları güçlendir) ───────────────────────────
    print("[7/9] Morfolojik işlemler uygulanıyor...")
    cekirdek = gi.morfoloji.kare_cekirdek(3)
    kapatilmis = gi.morfoloji.kapat(kenarlar, cekirdek)
    goruntu_kaydet(np.stack([kapatilmis] * 3, axis=2),
                   os.path.join(cikti_klasoru, "07_morfolojik_kapama.jpg"))

    # ── 8. Binary dönüşüm (son belge görünümü) ───────────────────────────────
    print("[8/9] Binary dönüşüm uygulanıyor...")
    binary = gi.temel.binary_donustur(yumusatilmis, esik=128)
    goruntu_kaydet(np.stack([binary] * 3, axis=2),
                   os.path.join(cikti_klasoru, "08_binary.jpg"))

    # ── 9. Renk uzayı dönüşümleri (bilgi amaçlı) ────────────────────────────
    print("[9/9] Renk uzayı dönüşümleri hesaplanıyor...")
    hsv = gi.temel.rgb_to_hsv(orijinal)
    goruntu_kaydet(hsv, os.path.join(cikti_klasoru, "09_hsv.jpg"))

    print("\nTüm adımlar tamamlandı.")
    print(f"Çıktılar '{cikti_klasoru}/' klasörüne kaydedildi.")


# ── Komut satırı arayüzü ────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Akıllı Belge Tarama ve İyileştirme Sistemi"
    )
    parser.add_argument("--girdi",  required=True,  help="Ham belge görüntüsü yolu")
    parser.add_argument("--cikti",  default="cikti", help="Çıktı klasörü (varsayılan: cikti)")
    args = parser.parse_args()

    belge_iyilestir(args.girdi, args.cikti)
