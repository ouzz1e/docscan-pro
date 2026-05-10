"""
Akıllı Belge Tarama ve İyileştirme Sistemi
Görüntü İşleme Kütüphanesi

KURAL: cv2 yalnızca imread / imshow / imwrite için kullanılabilir.
Tüm algoritmalar Python + NumPy ile sıfırdan implementedir.
"""

import numpy as np


# ─────────────────────────────────────────────
# 1. TEMEL DÖNÜŞÜMLER
# ─────────────────────────────────────────────

class TemelDonusumler:
    """Gri dönüşüm, binary dönüşüm ve renk uzayı dönüşümleri."""

    def griye_donustur(self, goruntu: np.ndarray) -> np.ndarray:
        """
        RGB görüntüyü gri tonlamaya çevirir.
        Formül: Y = 0.299*R + 0.587*G + 0.114*B  (ITU-R BT.601)
        Girdi : (H, W, 3) uint8
        Çıktı : (H, W)    uint8
        """
        if goruntu.ndim == 2:
            return goruntu.copy()

        R = goruntu[:, :, 0].astype(np.float64)
        G = goruntu[:, :, 1].astype(np.float64)
        B = goruntu[:, :, 2].astype(np.float64)

        gri = 0.299 * R + 0.587 * G + 0.114 * B
        return np.clip(gri, 0, 255).astype(np.uint8)

    def binary_donustur(self, gri: np.ndarray, esik: int = 128) -> np.ndarray:
        """
        Gri görüntüyü eşik değerine göre siyah-beyaza (0/255) dönüştürür.
        piksel > esik  →  255 (beyaz)
        piksel <= esik →    0 (siyah)
        Girdi : (H, W) uint8
        Çıktı : (H, W) uint8
        """
        binary = np.where(gri > esik, 255, 0).astype(np.uint8)
        return binary

    def rgb_to_hsv(self, goruntu: np.ndarray) -> np.ndarray:
        """
        RGB → HSV renk uzayı dönüşümü (manuel).
        H ∈ [0°, 360°)   S ∈ [0, 1]   V ∈ [0, 1]
        Çıktı değerleri uint8'e sıkıştırılmıştır:
            H → [0, 180]  (OpenCV standardı: /2)
            S → [0, 255]
            V → [0, 255]
        Girdi : (H, W, 3) uint8 RGB
        Çıktı : (H, W, 3) uint8 HSV
        """
        img = goruntu.astype(np.float64) / 255.0
        R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]

        Cmax = np.maximum(np.maximum(R, G), B)
        Cmin = np.minimum(np.minimum(R, G), B)
        delta = Cmax - Cmin

        # Value
        V = Cmax

        # Saturation
        S = np.where(Cmax != 0, delta / Cmax, 0.0)

        # Hue
        H = np.zeros_like(R)
        mask_r = (Cmax == R) & (delta != 0)
        mask_g = (Cmax == G) & (delta != 0)
        mask_b = (Cmax == B) & (delta != 0)

        H[mask_r] = (60.0 * ((G[mask_r] - B[mask_r]) / delta[mask_r])) % 360.0
        H[mask_g] = (60.0 * ((B[mask_g] - R[mask_g]) / delta[mask_g]) + 120.0) % 360.0
        H[mask_b] = (60.0 * ((R[mask_b] - G[mask_b]) / delta[mask_b]) + 240.0) % 360.0

        # OpenCV uyumlu ölçekleme
        H_out = np.clip(H / 2.0, 0, 180).astype(np.uint8)
        S_out = np.clip(S * 255.0, 0, 255).astype(np.uint8)
        V_out = np.clip(V * 255.0, 0, 255).astype(np.uint8)

        return np.stack([H_out, S_out, V_out], axis=2)

    def rgb_to_lab(self, goruntu: np.ndarray) -> np.ndarray:
        """
        RGB → CIE L*a*b* renk uzayı dönüşümü (manuel).
        Adımlar: RGB → lineer RGB → XYZ (D65) → Lab
        Çıktı float64: L ∈ [0,100], a ve b ∈ yaklaşık [-128, 127]
        Girdi : (H, W, 3) uint8 RGB
        Çıktı : (H, W, 3) float64 Lab
        """
        img = goruntu.astype(np.float64) / 255.0

        # Gamma düzeltmesi (sRGB → lineer)
        lineer = np.where(img <= 0.04045,
                          img / 12.92,
                          ((img + 0.055) / 1.055) ** 2.4)

        R, G, B = lineer[:, :, 0], lineer[:, :, 1], lineer[:, :, 2]

        # XYZ (D65 beyaz nokta, Bradford)
        X = 0.4124564 * R + 0.3575761 * G + 0.1804375 * B
        Y = 0.2126729 * R + 0.7151522 * G + 0.0721750 * B
        Z = 0.0193339 * R + 0.1191920 * G + 0.9503041 * B

        # D65 beyaz referans normalizasyonu
        X /= 0.95047
        Y /= 1.00000
        Z /= 1.08883

        # f(t) fonksiyonu
        eps = 0.008856
        kappa = 903.3
        f = np.where(X > eps, X ** (1.0 / 3.0), (kappa * X + 16.0) / 116.0)
        g = np.where(Y > eps, Y ** (1.0 / 3.0), (kappa * Y + 16.0) / 116.0)
        h = np.where(Z > eps, Z ** (1.0 / 3.0), (kappa * Z + 16.0) / 116.0)

        L = 116.0 * g - 16.0
        a = 500.0 * (f - g)
        b = 200.0 * (g - h)

        return np.stack([L, a, b], axis=2)


# ─────────────────────────────────────────────
# 2. GEOMETRİK İŞLEMLER
# ─────────────────────────────────────────────

class GeometrikIslemler:
    """Döndürme, kırpma ve ölçekleme."""

    def dondur(self, goruntu: np.ndarray, aci_derece: float,
               merkez: tuple = None) -> np.ndarray:
        """
        Görüntüyü verilen açıda saat yönünün tersine döndürür (bilineer interpolasyon).
        Girdi : (H, W) veya (H, W, C)
        Çıktı : aynı boyut
        """
        aci = np.deg2rad(aci_derece)
        cos_a, sin_a = np.cos(aci), np.sin(aci)

        H, W = goruntu.shape[:2]
        if merkez is None:
            cy, cx = H / 2.0, W / 2.0
        else:
            cx, cy = merkez

        # Çıktı piksellerinin ızgarası
        ys, xs = np.mgrid[0:H, 0:W]

        # Döndürme matrisinin tersi (çıktı → girdi eşleme)
        xs_src = cos_a * (xs - cx) + sin_a * (ys - cy) + cx
        ys_src = -sin_a * (xs - cx) + cos_a * (ys - cy) + cy

        return self._bilineer_interpolasyon(goruntu, xs_src, ys_src)

    def kirp(self, goruntu: np.ndarray,
             x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
        """
        Görüntüyü (x1,y1)-(x2,y2) dikdörtgeniyle kırpar.
        x → sütun, y → satır eksenine karşılık gelir.
        """
        return goruntu[y1:y2, x1:x2].copy()

    def olcekle(self, goruntu: np.ndarray,
                olcek_x: float, olcek_y: float = None) -> np.ndarray:
        """
        Görüntüyü ölçekler (bilineer interpolasyon).
        olcek > 1 → büyütme, olcek < 1 → küçültme
        """
        if olcek_y is None:
            olcek_y = olcek_x

        H, W = goruntu.shape[:2]
        yeni_H = int(round(H * olcek_y))
        yeni_W = int(round(W * olcek_x))

        ys_src = np.linspace(0, H - 1, yeni_H)
        xs_src = np.linspace(0, W - 1, yeni_W)
        xs_grid, ys_grid = np.meshgrid(xs_src, ys_src)

        return self._bilineer_interpolasyon(goruntu, xs_grid, ys_grid)

    # ── yardımcı ──────────────────────────────
    def _bilineer_interpolasyon(self, goruntu: np.ndarray,
                                xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """
        Verilen kaynak koordinatlarına (xs, ys) göre bilineer interpolasyon uygular.
        Sınır dışı koordinatlar sıfır (siyah) ile doldurulur.
        """
        H, W = goruntu.shape[:2]
        renkli = goruntu.ndim == 3

        x0 = np.floor(xs).astype(int)
        y0 = np.floor(ys).astype(int)
        x1 = x0 + 1
        y1 = y0 + 1

        tx = xs - x0
        ty = ys - y0

        # Sınır kontrolü
        gecerli = (x0 >= 0) & (y0 >= 0) & (x1 < W) & (y1 < H)

        x0c = np.clip(x0, 0, W - 1)
        y0c = np.clip(y0, 0, H - 1)
        x1c = np.clip(x1, 0, W - 1)
        y1c = np.clip(y1, 0, H - 1)

        if renkli:
            C = goruntu.shape[2]
            sonuc = np.zeros((ys.shape[0], xs.shape[1], C), dtype=np.float64)
            for c in range(C):
                Ia = goruntu[y0c, x0c, c].astype(np.float64)
                Ib = goruntu[y1c, x0c, c].astype(np.float64)
                Ic = goruntu[y0c, x1c, c].astype(np.float64)
                Id = goruntu[y1c, x1c, c].astype(np.float64)
                interpolated = (Ia * (1 - tx) * (1 - ty) +
                                Ic * tx * (1 - ty) +
                                Ib * (1 - tx) * ty +
                                Id * tx * ty)
                sonuc[:, :, c] = np.where(gecerli, interpolated, 0)
        else:
            Ia = goruntu[y0c, x0c].astype(np.float64)
            Ib = goruntu[y1c, x0c].astype(np.float64)
            Ic = goruntu[y0c, x1c].astype(np.float64)
            Id = goruntu[y1c, x1c].astype(np.float64)
            interpolated = (Ia * (1 - tx) * (1 - ty) +
                            Ic * tx * (1 - ty) +
                            Ib * (1 - tx) * ty +
                            Id * tx * ty)
            sonuc = np.where(gecerli, interpolated, 0)

        return np.clip(sonuc, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 3. İSTATİSTİKSEL VE KONTRAST İŞLEMLERİ
# ─────────────────────────────────────────────

class KontrastIslemleri:
    """Histogram hesaplama, germe, kontrast azaltma, aritmetik işlemler."""

    def histogram_hesapla(self, gri: np.ndarray) -> np.ndarray:
        """
        Gri görüntünün [0-255] histogram dizisini döndürür.
        Çıktı: (256,) int array — her indeks o parlaklık değerinin piksel sayısı
        """
        hist = np.zeros(256, dtype=np.int64)
        duzlenmis = gri.ravel()
        for piksel in duzlenmis:
            hist[piksel] += 1
        return hist

    def arkaplan_normalize(self, gri: np.ndarray,
                           pencere: int = 81) -> np.ndarray:
        """
        Büyük pencereli yerel ortalama ile arka plan aydınlanma dengesizliğini giderir.
        Fotoğraftaki gölge veya ışık farklarından bağımsız olarak kağıt beyaza çekilir.
        Her piksel: normalize = (piksel / yerel_ort) × 220 — integral görüntü ile O(N).
        Girdi : (H, W) uint8 gri
        Çıktı : (H, W) uint8 gri (arka plan ~255, metin korunur)
        """
        img = gri.astype(np.float64)
        H, W = img.shape
        pad = pencere // 2

        dolgulu = np.pad(img, pad, mode='reflect')
        I = np.zeros((H + 2*pad + 1, W + 2*pad + 1), dtype=np.float64)
        I[1:, 1:] = np.cumsum(np.cumsum(dolgulu, axis=0), axis=1)

        r = np.arange(H)[:, None]
        c = np.arange(W)[None, :]

        toplam = (I[r + pencere, c + pencere]
                - I[r,           c + pencere]
                - I[r + pencere, c]
                + I[r,           c])

        yerel_ort = toplam / (pencere * pencere)
        normalize = img / (yerel_ort + 1e-6) * 220
        return np.clip(normalize, 0, 255).astype(np.uint8)

    def histogram_ger(self, gri: np.ndarray) -> np.ndarray:
        """
        Histogram germe (contrast stretching):
        piksel_yeni = (piksel - min) / (max - min) * 255
        Görüntünün tüm dinamik aralığı [0, 255]'e yayılır.
        """
        img = gri.astype(np.float64)
        p_min = img.min()
        p_max = img.max()
        if p_max == p_min:
            return gri.copy()
        gerilmis = (img - p_min) / (p_max - p_min) * 255.0
        return np.clip(gerilmis, 0, 255).astype(np.uint8)

    def kontrast_azalt(self, gri: np.ndarray, faktor: float = 0.5) -> np.ndarray:
        """
        Kontrastı azaltır: piksel_yeni = 128 + faktor * (piksel - 128)
        faktor < 1 → kontrast azalır, faktor > 1 → kontrast artar
        """
        img = gri.astype(np.float64)
        sonuc = 128.0 + faktor * (img - 128.0)
        return np.clip(sonuc, 0, 255).astype(np.uint8)

    def aritmetik_cikar(self, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
        """İki gri görüntüyü çıkarır; negatif değerler 0'a sıkıştırılır."""
        fark = g1.astype(np.int16) - g2.astype(np.int16)
        return np.clip(fark, 0, 255).astype(np.uint8)

    def aritmetik_carp(self, g1: np.ndarray, g2: np.ndarray) -> np.ndarray:
        """
        İki gri görüntüyü piksel piksel çarpar.
        Her iki görüntü [0,1]'e normalize edilip çarpılır, ardından [0,255]'e döndürülür.
        """
        a = g1.astype(np.float64) / 255.0
        b = g2.astype(np.float64) / 255.0
        carpim = a * b * 255.0
        return np.clip(carpim, 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────
# 4. FİLTRELEME VE KONVOLÜSYON
# ─────────────────────────────────────────────

class FiltreIslemleri:
    """Genel konvolüsyon, mean, median ve motion filtreler."""

    # ── çekirdek fabrikaları ──────────────────
    @staticmethod
    def mean_cekirdek(boyut: int = 3) -> np.ndarray:
        """Uniform ortalama çekirdeği."""
        return np.ones((boyut, boyut), dtype=np.float64) / (boyut * boyut)

    @staticmethod
    def motion_cekirdek(uzunluk: int = 15, aci_derece: float = 0.0) -> np.ndarray:
        """
        Hareket bulanıklığı çekirdeği:
        Belirtilen açıda uzunluk kadar çizgisel çekirdek oluşturur.
        """
        cekirdek = np.zeros((uzunluk, uzunluk), dtype=np.float64)
        merkez = uzunluk // 2
        aci = np.deg2rad(aci_derece)
        for i in range(uzunluk):
            ofs = i - merkez
            col = int(round(merkez + ofs * np.cos(aci)))
            row = int(round(merkez + ofs * np.sin(aci)))
            if 0 <= row < uzunluk and 0 <= col < uzunluk:
                cekirdek[row, col] = 1.0
        toplam = cekirdek.sum()
        if toplam > 0:
            cekirdek /= toplam
        return cekirdek

    # ── çekirdek tabanlı konvolüsyon ─────────
    def konvolusyon(self, goruntu: np.ndarray,
                    cekirdek: np.ndarray) -> np.ndarray:
        """
        Genel 2D çapraz korelasyon (görüntü işlemde "konvolüsyon" olarak anılır).
        Simetrik çekirdekler için matematiksel konvolüsyonla özdeştir.
        Birden fazla kanal varsa her kanala ayrı ayrı uygulanır.
        Girdi : (H, W) veya (H, W, C)
        Çıktı : aynı boyut, uint8
        """
        if goruntu.ndim == 3:
            kanallar = [self._konvolusyon_2d(goruntu[:, :, c].astype(np.float64), cekirdek)
                        for c in range(goruntu.shape[2])]
            sonuc = np.stack(kanallar, axis=2)
        else:
            sonuc = self._konvolusyon_2d(goruntu.astype(np.float64), cekirdek)

        return np.clip(sonuc, 0, 255).astype(np.uint8)

    def _konvolusyon_2d(self, kanal: np.ndarray,
                        cekirdek: np.ndarray) -> np.ndarray:
        """
        Tek kanallı float64 matris üzerinde 2D çapraz korelasyon.
        Zero-padding kullanılır; çıktı girdi ile aynı boyuttadır.
        """
        H, W = kanal.shape
        kH, kW = cekirdek.shape
        pad_h = kH // 2
        pad_w = kW // 2

        dolgulu = np.pad(kanal, ((pad_h, pad_h), (pad_w, pad_w)),
                         mode='constant', constant_values=0)
        sonuc = np.zeros((H, W), dtype=np.float64)

        for i in range(H):
            for j in range(W):
                bolge = dolgulu[i: i + kH, j: j + kW]
                sonuc[i, j] = np.sum(bolge * cekirdek)

        return sonuc

    def mean_filtre(self, goruntu: np.ndarray, boyut: int = 3) -> np.ndarray:
        """
        Ortalama (mean) filtre — her pikseli komşularının ortalamasıyla değiştirir.
        Boyut tek sayı olmalıdır (3, 5, 7 …).
        """
        cekirdek = self.mean_cekirdek(boyut)
        return self.konvolusyon(goruntu, cekirdek)

    def median_filtre(self, goruntu: np.ndarray, boyut: int = 3) -> np.ndarray:
        """
        Medyan filtresi — her pikseli komşuların ortanca değeriyle değiştirir.
        Salt & pepper gürültüsünde mean'e göre çok daha etkilidir.
        Konvolüsyon tabanlı değil, pencere sıralama tabanlıdır.
        """
        H, W = goruntu.shape[:2]
        pad = boyut // 2
        renkli = goruntu.ndim == 3

        if renkli:
            C = goruntu.shape[2]
            sonuc = np.zeros_like(goruntu)
            for c in range(C):
                dolgulu = np.pad(goruntu[:, :, c], pad, mode='constant')
                for i in range(H):
                    for j in range(W):
                        pencere = dolgulu[i: i + boyut, j: j + boyut].ravel()
                        # Manuel sıralama ile medyan
                        siralı = np.sort(pencere)
                        sonuc[i, j, c] = siralı[len(siralı) // 2]
        else:
            sonuc = np.zeros_like(goruntu)
            dolgulu = np.pad(goruntu, pad, mode='constant')
            for i in range(H):
                for j in range(W):
                    pencere = dolgulu[i: i + boyut, j: j + boyut].ravel()
                    siralı = np.sort(pencere)
                    sonuc[i, j] = siralı[len(siralı) // 2]

        return sonuc

    def motion_filtre(self, goruntu: np.ndarray,
                      uzunluk: int = 15, aci_derece: float = 0.0) -> np.ndarray:
        """Hareket bulanıklığı filtresi."""
        cekirdek = self.motion_cekirdek(uzunluk, aci_derece)
        return self.konvolusyon(goruntu, cekirdek)


# ─────────────────────────────────────────────
# 5. GÜRÜLTÜ ANALİZİ
# ─────────────────────────────────────────────

class GurultuAnalizi:
    """Salt & pepper gürültü ekleme ve temizleme."""

    def salt_pepper_ekle(self, goruntu: np.ndarray,
                         yogunluk: float = 0.05,
                         tohum: int = None) -> np.ndarray:
        """
        Görüntüye rastgele 'tuz ve biber' gürültüsü ekler.
        yogunluk : toplam etkilenen piksel oranı (0.0 – 1.0)
        Tuz (beyaz) ve biber (siyah) piksel sayısı eşit tutulur.
        """
        if tohum is not None:
            np.random.seed(tohum)

        sonuc = goruntu.copy()
        H, W = goruntu.shape[:2]
        toplam_piksel = H * W
        etkilenen = int(toplam_piksel * yogunluk)

        # Siyah (biber)
        satirlar = np.random.randint(0, H, etkilenen // 2)
        sutunlar = np.random.randint(0, W, etkilenen // 2)
        if goruntu.ndim == 3:
            sonuc[satirlar, sutunlar] = 0
        else:
            sonuc[satirlar, sutunlar] = 0

        # Beyaz (tuz)
        satirlar = np.random.randint(0, H, etkilenen // 2)
        sutunlar = np.random.randint(0, W, etkilenen // 2)
        if goruntu.ndim == 3:
            sonuc[satirlar, sutunlar] = 255
        else:
            sonuc[satirlar, sutunlar] = 255

        return sonuc

    def salt_pepper_temizle(self, gurultulu: np.ndarray,
                            boyut: int = 3) -> np.ndarray:
        """
        Medyan filtresiyle salt & pepper gürültüsünü temizler.
        FiltreIslemleri.median_filtre ile aynı algoritmayı kullanır.
        """
        filtre = FiltreIslemleri()
        return filtre.median_filtre(gurultulu, boyut)


# ─────────────────────────────────────────────
# 6. SEGMENTASYON VE KENAR BULMA
# ─────────────────────────────────────────────

class KenarBulma:
    """Canny kenar algılama ve çift eşikleme."""

    def _gaussian_cekirdek(self, boyut: int = 5,
                            sigma: float = 1.4) -> np.ndarray:
        """Gaussian pürüzsüzleştirme çekirdeği (manuel)."""
        merkez = boyut // 2
        cekirdek = np.zeros((boyut, boyut), dtype=np.float64)
        for i in range(boyut):
            for j in range(boyut):
                x = i - merkez
                y = j - merkez
                cekirdek[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
        cekirdek /= cekirdek.sum()
        return cekirdek

    def _sobel_gradyan(self, gri: np.ndarray):
        """
        Sobel operatörüyle gradyan büyüklüğü ve yönünü hesaplar.
        Geri döndürür: (buyukluk, aci_radyan)
        """
        filtre = FiltreIslemleri()

        Kx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]], dtype=np.float64)
        Ky = np.array([[-1, -2, -1],
                       [ 0,  0,  0],
                       [ 1,  2,  1]], dtype=np.float64)

        Gx = filtre._konvolusyon_2d(gri.astype(np.float64), Kx)
        Gy = filtre._konvolusyon_2d(gri.astype(np.float64), Ky)

        buyukluk = np.sqrt(Gx**2 + Gy**2)
        aci = np.arctan2(Gy, Gx)
        return buyukluk, aci

    def _nms(self, buyukluk: np.ndarray, aci: np.ndarray) -> np.ndarray:
        """
        Non-Maximum Suppression (Maksimum-Olmayan Bastırma).
        Gradyan yönünde yerel maksimum olmayan pikselleri sıfırlar.
        """
        H, W = buyukluk.shape
        sonuc = np.zeros((H, W), dtype=np.float64)
        aci_d = np.rad2deg(aci) % 180  # 0°–180° aralığına sıkıştır

        for i in range(1, H - 1):
            for j in range(1, W - 1):
                a = aci_d[i, j]
                m = buyukluk[i, j]

                # Gradyan yönüne göre komşu seçimi (4 temel yön)
                if (0 <= a < 22.5) or (157.5 <= a <= 180):
                    k1, k2 = buyukluk[i, j - 1], buyukluk[i, j + 1]
                elif 22.5 <= a < 67.5:
                    k1, k2 = buyukluk[i + 1, j - 1], buyukluk[i - 1, j + 1]
                elif 67.5 <= a < 112.5:
                    k1, k2 = buyukluk[i - 1, j], buyukluk[i + 1, j]
                else:
                    k1, k2 = buyukluk[i - 1, j - 1], buyukluk[i + 1, j + 1]

                if m >= k1 and m >= k2:
                    sonuc[i, j] = m

        return sonuc

    def cift_esikleme(self, nms: np.ndarray,
                      dusuk_esik: float, yuksek_esik: float) -> np.ndarray:
        """
        Çift eşikleme (hysteresis):
          > yüksek_esik → kesinlikle kenar (255)
          > düşük_esik  → zayıf kenar (128), güçlü kenara bağlıysa kabul edilir
          diğer         → 0 (kenar değil)
        """
        H, W = nms.shape
        sonuc = np.zeros((H, W), dtype=np.uint8)

        guclu = nms >= yuksek_esik
        zayif = (nms >= dusuk_esik) & ~guclu

        sonuc[guclu] = 255
        sonuc[zayif] = 128

        # Histerezis: zayıf piksel 8-komşusunda güçlü piksel varsa kabul et
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if sonuc[i, j] == 128:
                    bolge = sonuc[i - 1: i + 2, j - 1: j + 2]
                    if np.any(bolge == 255):
                        sonuc[i, j] = 255
                    else:
                        sonuc[i, j] = 0

        return sonuc

    def canny_kenar(self, gri: np.ndarray,
                    dusuk_esik: float = 50.0,
                    yuksek_esik: float = 150.0,
                    gaussian_boyut: int = 5,
                    sigma: float = 1.4) -> np.ndarray:
        """
        Tam Canny kenar algılama boru hattı (manuel):
        1. Gaussian pürüzsüzleştirme
        2. Sobel gradyanı (büyüklük + yön)
        3. Non-Maximum Suppression
        4. Çift eşikleme + histerezis
        Girdi : (H, W) uint8 gri
        Çıktı : (H, W) uint8 — kenarlar 255, arka plan 0
        """
        filtre = FiltreIslemleri()
        gauss = self._gaussian_cekirdek(gaussian_boyut, sigma)
        yumusatilmis = filtre._konvolusyon_2d(gri.astype(np.float64), gauss)

        buyukluk, aci = self._sobel_gradyan(yumusatilmis)
        nms = self._nms(buyukluk, aci)

        return self.cift_esikleme(nms, dusuk_esik, yuksek_esik)


# ─────────────────────────────────────────────
# 7. MORFOLOJİK İŞLEMLER
# ─────────────────────────────────────────────

class MorfolojikIslemler:
    """Dilation, Erosion, Opening, Closing — binary veya gri görüntüler için."""

    @staticmethod
    def kare_cekirdek(boyut: int = 3) -> np.ndarray:
        """Tam dolu kare yapısal eleman."""
        return np.ones((boyut, boyut), dtype=np.uint8)

    @staticmethod
    def cizgi_cekirdek(uzunluk: int = 5,
                       yatay: bool = True) -> np.ndarray:
        """Yatay veya dikey çizgi yapısal elemanı."""
        if yatay:
            return np.ones((1, uzunluk), dtype=np.uint8)
        return np.ones((uzunluk, 1), dtype=np.uint8)

    def genisle(self, goruntu: np.ndarray,
                cekirdek: np.ndarray = None) -> np.ndarray:
        """
        Dilation (Genişleme):
        Her pikseli, yapısal elemanın örttüğü bölgedeki maksimum değerle değiştirir.
        Binary görüntülerde beyaz bölgeleri büyütür.
        """
        if cekirdek is None:
            cekirdek = self.kare_cekirdek(3)

        H, W = goruntu.shape[:2]
        kH, kW = cekirdek.shape
        pad_h, pad_w = kH // 2, kW // 2

        dolgulu = np.pad(goruntu, ((pad_h, pad_h), (pad_w, pad_w)),
                         mode='constant', constant_values=0)
        sonuc = np.zeros_like(goruntu)

        for i in range(H):
            for j in range(W):
                bolge = dolgulu[i: i + kH, j: j + kW]
                # Yalnızca yapısal elemanın 1 olan konumları kullan
                degerler = bolge[cekirdek == 1]
                sonuc[i, j] = degerler.max() if degerler.size > 0 else 0

        return sonuc

    def asin(self, goruntu: np.ndarray,
             cekirdek: np.ndarray = None) -> np.ndarray:
        """
        Erosion (Aşınma):
        Her pikseli, yapısal elemanın örttüğü bölgedeki minimum değerle değiştirir.
        Binary görüntülerde beyaz bölgeleri küçültür.
        """
        if cekirdek is None:
            cekirdek = self.kare_cekirdek(3)

        H, W = goruntu.shape[:2]
        kH, kW = cekirdek.shape
        pad_h, pad_w = kH // 2, kW // 2

        dolgulu = np.pad(goruntu, ((pad_h, pad_h), (pad_w, pad_w)),
                         mode='constant', constant_values=0)
        sonuc = np.zeros_like(goruntu)

        for i in range(H):
            for j in range(W):
                bolge = dolgulu[i: i + kH, j: j + kW]
                degerler = bolge[cekirdek == 1]
                sonuc[i, j] = degerler.min() if degerler.size > 0 else 0

        return sonuc

    def ac(self, goruntu: np.ndarray,
           cekirdek: np.ndarray = None) -> np.ndarray:
        """
        Opening (Açma) = Erosion → Dilation
        Küçük beyaz gürültü noktalarını temizler, şekilleri korur.
        """
        if cekirdek is None:
            cekirdek = self.kare_cekirdek(3)
        return self.genisle(self.asin(goruntu, cekirdek), cekirdek)

    def kapat(self, goruntu: np.ndarray,
              cekirdek: np.ndarray = None) -> np.ndarray:
        """
        Closing (Kapama) = Dilation → Erosion
        Küçük delikleri ve boşlukları kapatır.
        """
        if cekirdek is None:
            cekirdek = self.kare_cekirdek(3)
        return self.asin(self.genisle(goruntu, cekirdek), cekirdek)


# ─────────────────────────────────────────────
# 8. PERSPEKTİF DÜZELTME (HOMOGRAFİ)
# ─────────────────────────────────────────────

class PerspektifDuzeltme:
    """Manuel DLT homografi hesabı ve perspektif dönüşümü."""

    def homografi_hesapla(self, kaynak: np.ndarray,
                          hedef: np.ndarray) -> np.ndarray:
        """
        4 nokta çiftinden 3×3 homografi matrisi hesaplar (DLT algoritması).
        kaynak, hedef : (4, 2) float array — [(x, y), ...]
        Döndürür      : (3, 3) float64
        """
        A = []
        for (xs, ys), (xd, yd) in zip(kaynak, hedef):
            A.append([-xs, -ys, -1,   0,   0,  0,  xd*xs, xd*ys, xd])
            A.append([  0,   0,  0, -xs, -ys, -1,  yd*xs, yd*ys, yd])
        A = np.array(A, dtype=np.float64)

        # SVD: en küçük tekil değerin sağ vektörü çözümdür
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        H /= H[2, 2]
        return H

    def perspektif_donustur(self, goruntu: np.ndarray,
                             H: np.ndarray,
                             cikti_gen: int,
                             cikti_yuk: int) -> np.ndarray:
        """
        H ile görüntüyü perspektif dönüşümüne uğratır.
        Ters eşleme (çıktı → kaynak) + bilineer interpolasyon.
        Girdi : (H, W) veya (H, W, C)
        Çıktı : (cikti_yuk, cikti_gen) aynı kanal sayısıyla
        """
        H_inv = np.linalg.inv(H)

        ys_g, xs_g = np.mgrid[0:cikti_yuk, 0:cikti_gen]
        ones = np.ones(cikti_yuk * cikti_gen, dtype=np.float64)

        pts = np.stack([xs_g.ravel().astype(np.float64),
                        ys_g.ravel().astype(np.float64),
                        ones], axis=0)  # (3, N)

        src = H_inv @ pts
        src /= src[2:3, :]  # homojen → kartezyen

        xs_src = src[0].reshape(cikti_yuk, cikti_gen)
        ys_src = src[1].reshape(cikti_yuk, cikti_gen)

        geo = GeometrikIslemler()
        return geo._bilineer_interpolasyon(goruntu, xs_src, ys_src)


# ─────────────────────────────────────────────
# 9. ADAPTİF EŞİKLEME (SAUVOLA)
# ─────────────────────────────────────────────

class AdaptifEsikleme:
    """Sauvola yerel adaptif eşikleme — integral görüntü ile O(N) hız."""

    def sauvola(self, gri: np.ndarray,
                pencere: int = 25,
                k: float = 0.2,
                R: float = 128.0) -> np.ndarray:
        """
        Sauvola eşikleme:
          T(i,j) = μ(i,j) · [1 + k · (σ(i,j)/R − 1)]
        Piksel > T → 255 (beyaz), aksi → 0 (siyah).
        İntegral görüntü (SAT) ile her piksel O(1) hesaplanır.
        Girdi : (H, W) uint8 gri
        Çıktı : (H, W) uint8 binary
        """
        img = gri.astype(np.float64)
        H, W = img.shape
        pad = pencere // 2

        # Kenar yansımalı dolgu
        dolgulu   = np.pad(img,      pad, mode='reflect')
        dolgulu_k = np.pad(img**2,   pad, mode='reflect')

        # Kümülatif toplamlar (sıfır satır/sütun eklenmiş)
        I   = np.zeros((H + 2*pad + 1, W + 2*pad + 1), dtype=np.float64)
        I2  = np.zeros_like(I)
        I[1:,  1:]  = np.cumsum(np.cumsum(dolgulu,   axis=0), axis=1)
        I2[1:, 1:]  = np.cumsum(np.cumsum(dolgulu_k, axis=0), axis=1)

        # Piksel (i,j)'nin penceresi dolgulu'da [i, i+pencere) × [j, j+pencere)
        # SAT formülü: sum(y1..y1+p-1, x1..x1+p-1) = I[y1+p, x1+p] - I[y1, x1+p] - I[y1+p, x1] + I[y1, x1]
        r = np.arange(H)[:, None]   # (H, 1)
        c = np.arange(W)[None, :]   # (1, W)

        toplam = (I[r + pencere, c + pencere]
                - I[r,           c + pencere]
                - I[r + pencere, c]
                + I[r,           c])

        toplam_k = (I2[r + pencere, c + pencere]
                  - I2[r,           c + pencere]
                  - I2[r + pencere, c]
                  + I2[r,           c])

        n       = pencere * pencere
        mu      = toplam / n
        varyans = np.maximum(toplam_k / n - mu**2, 0.0)
        sigma   = np.sqrt(varyans)

        esik    = mu * (1.0 + k * (sigma / R - 1.0))
        return np.where(img > esik, 255, 0).astype(np.uint8)


# ─────────────────────────────────────────────
# MERKEZ SINIF — tüm modüllere tek erişim noktası
# ─────────────────────────────────────────────

class GorunteIsleyici:
    """
    Tüm görüntü işleme modüllerini tek çatı altında toplar.
    Kullanım:
        gi = GorunteIsleyici()
        gri = gi.temel.griye_donustur(goruntu)
        kenarlar = gi.kenar.canny_kenar(gri)
    """

    def __init__(self):
        self.temel      = TemelDonusumler()
        self.geometri   = GeometrikIslemler()
        self.kontrast   = KontrastIslemleri()
        self.filtre     = FiltreIslemleri()
        self.gurultu    = GurultuAnalizi()
        self.kenar      = KenarBulma()
        self.morfoloji  = MorfolojikIslemler()
        self.perspektif = PerspektifDuzeltme()
        self.adaptif    = AdaptifEsikleme()
