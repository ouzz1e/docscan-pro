"""
Akıllı Belge Tarama ve İyileştirme Sistemi — Masaüstü Uygulaması
Çalıştırmak için: python app.py
"""

import os
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageTk

from lib_goruntu import GorunteIsleyici

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

# ─── Renk paleti ──────────────────────────────────────────────────────────────
C = {
    "bg":        "#0d1117",
    "panel":     "#161b22",
    "card":      "#21262d",
    "border":    "#30363d",
    "accent":    "#58a6ff",
    "green":     "#3fb950",
    "yellow":    "#d29922",
    "red":       "#f85149",
    "text":      "#e6edf3",
    "muted":     "#8b949e",
    "highlight": "#1f6feb",
}


# ─── Yardımcı bileşenler ──────────────────────────────────────────────────────

def _btn(parent, text, cmd, color=None, **kw):
    return ctk.CTkButton(
        parent, text=text, command=cmd,
        fg_color=color or C["highlight"],
        hover_color="#388bfd",
        corner_radius=6, height=30,
        font=ctk.CTkFont(size=11),
        **kw
    )


def _lbl(parent, text, size=10, color=None, bold=False):
    return ctk.CTkLabel(
        parent, text=text,
        font=ctk.CTkFont(size=size, weight="bold" if bold else "normal"),
        text_color=color or C["muted"]
    )


def _slider(parent, var, lo, hi, steps):
    return ctk.CTkSlider(
        parent, from_=lo, to=hi, number_of_steps=steps,
        variable=var, button_color=C["accent"],
        progress_color=C["highlight"], height=16
    )


def _section(parent, title):
    f = ctk.CTkFrame(parent, fg_color=C["card"], corner_radius=8)
    f.pack(fill="x", padx=6, pady=4)
    ctk.CTkLabel(
        f, text=title,
        font=ctk.CTkFont(size=11, weight="bold"),
        text_color=C["accent"]
    ).pack(anchor="w", padx=10, pady=(8, 2))
    ctk.CTkFrame(f, height=1, fg_color=C["border"]).pack(fill="x", padx=10, pady=(0, 6))
    return f


# ══════════════════════════════════════════════════════════════════════════════
# ANA UYGULAMA
# ══════════════════════════════════════════════════════════════════════════════

class BelgeTaramaApp(ctk.CTk):

    # ── Başlatma ──────────────────────────────────────────────────────────────

    def __init__(self):
        super().__init__()
        self.gi = GorunteIsleyici()

        # Durum
        self.orijinal: np.ndarray | None = None   # RGB uint8
        self.mevcut:   np.ndarray | None = None   # RGB uint8 (işlenmiş)
        self._photo = None                         # ImageTk referansı
        self.kose_img: np.ndarray | None = None   # (4,2) float — görüntü koord.
        self._surukle_idx: int | None = None
        self._goruntu_olcek   = 1.0
        self._goruntu_offset  = (0, 0)

        # Geri al stack'i
        self._gecmis: list = []          # her eleman: (mevcut_kopya, kose_kopya)
        self._gecmis_limit = 25

        # Büyüteç ImageTk referansı (GC'den korunmak için)
        self._mag_photo = None

        self._pencere_kur()
        self._ui_kur()

    def _pencere_kur(self):
        self.title("DocScan Pro — Akıllı Belge Tarama Sistemi")
        self.geometry("1440x860")
        self.minsize(1100, 700)
        self.configure(fg_color=C["bg"])

    # ── UI iskelet ────────────────────────────────────────────────────────────

    def _ui_kur(self):
        self._toolbar_kur()
        ana = ctk.CTkFrame(self, fg_color=C["bg"], corner_radius=0)
        ana.pack(fill="both", expand=True, padx=8, pady=(4, 8))
        ana.columnconfigure(0, weight=3)
        ana.columnconfigure(1, weight=1, minsize=290)
        ana.rowconfigure(0, weight=1)
        self._canvas_panel_kur(ana)
        self._sag_panel_kur(ana)
        self._statusbar_kur()

    # ── Toolbar ───────────────────────────────────────────────────────────────

    def _toolbar_kur(self):
        tb = ctk.CTkFrame(self, fg_color=C["panel"], height=48, corner_radius=0)
        tb.pack(fill="x")
        tb.pack_propagate(False)

        _lbl(tb, "  DocScan Pro", size=15, color=C["accent"], bold=True).pack(
            side="left", padx=4)

        separator = ctk.CTkFrame(tb, width=1, fg_color=C["border"])
        separator.pack(side="left", fill="y", padx=8, pady=8)

        for text, cmd, color in [
            ("  Görüntü Aç",    self.goruntu_ac,            C["highlight"]),
            ("  JPG Kaydet",    self.jpg_kaydet,             "#1a4731"),
            ("  PDF Kaydet",    self.pdf_kaydet,             "#1a4731"),
            ("  Karşılaştır",   self.karsilastirma_ac,       "#3d2b00"),
            ("  Geri Al",       self.geri_al,                "#3d2400"),
            ("  Sıfırla",       self.sifirla,                "#3d1212"),
        ]:
            ctk.CTkButton(
                tb, text=text, command=cmd,
                fg_color=color, hover_color=C["highlight"],
                corner_radius=6, height=32, width=120,
                font=ctk.CTkFont(size=11)
            ).pack(side="left", padx=4, pady=8)

    # ── Canvas paneli ─────────────────────────────────────────────────────────

    def _canvas_panel_kur(self, parent):
        sol = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=10)
        sol.grid(row=0, column=0, sticky="nsew", padx=(0, 4))

        _lbl(sol,
             "Belge Görünümü  —  Köşe noktalarını sürükleyerek seçimi düzelt",
             size=11, color=C["muted"]).pack(pady=(8, 4))

        self.kanvas = tk.Canvas(sol, bg=C["bg"], highlightthickness=0,
                                cursor="crosshair")
        self.kanvas.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        self.kanvas.bind("<ButtonPress-1>",   self._surukle_baslat)
        self.kanvas.bind("<B1-Motion>",       self._surukle_devam)
        self.kanvas.bind("<ButtonRelease-1>", self._surukle_bitis)
        self.kanvas.bind("<Configure>",       lambda _: self._kanvasi_yenile())

    # ── Sağ kaydırmalı panel ──────────────────────────────────────────────────

    def _sag_panel_kur(self, parent):
        sag_dis = ctk.CTkFrame(parent, fg_color=C["panel"], corner_radius=10)
        sag_dis.grid(row=0, column=1, sticky="nsew")

        _lbl(sag_dis, "Teknik İşlemler", size=13, color=C["text"], bold=True).pack(
            pady=(10, 4))
        ctk.CTkFrame(sag_dis, height=1, fg_color=C["border"]).pack(
            fill="x", padx=8, pady=(0, 4))

        scroll = ctk.CTkScrollableFrame(sag_dis, fg_color=C["bg"], corner_radius=8)
        scroll.pack(fill="both", expand=True, padx=6, pady=(0, 6))

        self._camscanner_bolum(scroll)
        self._temel_bolum(scroll)
        self._geometri_bolum(scroll)
        self._kontrast_bolum(scroll)
        self._filtre_bolum(scroll)
        self._gurultu_bolum(scroll)
        self._kenar_bolum(scroll)
        self._morfo_bolum(scroll)

    # ── Status bar ────────────────────────────────────────────────────────────

    def _statusbar_kur(self):
        self._durum_var = ctk.StringVar(value="Hazır — 'Görüntü Aç' ile başlayın.")
        sb = ctk.CTkFrame(self, fg_color=C["panel"], height=26, corner_radius=0)
        sb.pack(fill="x", side="bottom")
        sb.pack_propagate(False)
        ctk.CTkLabel(sb, textvariable=self._durum_var,
                     font=ctk.CTkFont(size=10),
                     text_color=C["muted"]).pack(side="left", padx=12)

    # ══════════════════════════════════════════════════════════════════════════
    # PANEL BÖLÜMLERI
    # ══════════════════════════════════════════════════════════════════════════

    def _camscanner_bolum(self, parent):
        f = ctk.CTkFrame(parent, fg_color="#0d2137", corner_radius=10,
                         border_width=1, border_color="#1f6feb")
        f.pack(fill="x", padx=6, pady=6)

        ctk.CTkLabel(f, text="CamScanner Modu",
                     font=ctk.CTkFont(size=12, weight="bold"),
                     text_color=C["green"]).pack(pady=(8, 4))

        for text, cmd in [
            ("Otomatik Köşe Tespiti",             self.oto_kose),
            ("Perspektif Düzelt",                  self.perspektif_duzelt),
            ("Tam İyileştirme  (Hist + Sauvola + Morfo)", self.tam_iyilestir),
        ]:
            ctk.CTkButton(
                f, text=text, command=cmd,
                fg_color="#0d3321", hover_color="#1a4731",
                corner_radius=6, height=32,
                font=ctk.CTkFont(size=11)
            ).pack(fill="x", padx=10, pady=3)

        ctk.CTkFrame(f, height=6, fg_color="transparent").pack()

    def _temel_bolum(self, parent):
        f = _section(parent, "1 · Temel Dönüşümler")

        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(0, 4))
        for text, cmd in [("Gri", self.op_gri), ("Binary", self.op_binary),
                           ("HSV", self.op_hsv),  ("LAB",  self.op_lab)]:
            _btn(row, text, cmd).pack(side="left", expand=True, fill="x", padx=2)

        self._v_binary_esik = ctk.IntVar(value=128)
        _lbl(f, "Binary Eşik").pack(anchor="w", padx=10)
        _slider(f, self._v_binary_esik, 0, 255, 255).pack(fill="x", padx=10, pady=(0, 8))

    def _geometri_bolum(self, parent):
        f = _section(parent, "2 · Geometrik İşlemler")

        self._v_donus = ctk.DoubleVar(value=0)
        _lbl(f, "Döndürme Açısı (°)").pack(anchor="w", padx=10)
        _slider(f, self._v_donus, -180, 180, 360).pack(fill="x", padx=10, pady=2)
        _btn(f, "Döndürmeyi Uygula", self.op_dondur).pack(
            fill="x", padx=10, pady=(2, 6))

        self._v_olcek = ctk.DoubleVar(value=1.0)
        _lbl(f, "Ölçek Faktörü").pack(anchor="w", padx=10)
        _slider(f, self._v_olcek, 0.1, 3.0, 290).pack(fill="x", padx=10, pady=2)
        _btn(f, "Ölçeklemeyi Uygula", self.op_olcekle).pack(
            fill="x", padx=10, pady=(2, 8))

    def _kontrast_bolum(self, parent):
        f = _section(parent, "3 · İstatistiksel & Kontrast")

        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(0, 4))
        _btn(row, "Hist. Ger",    self.op_hist_ger).pack(
            side="left", expand=True, fill="x", padx=2)
        _btn(row, "Hist. Grafik", self.op_hist_grafik).pack(
            side="left", expand=True, fill="x", padx=2)

        self._v_kontrast = ctk.DoubleVar(value=0.5)
        _lbl(f, "Kontrast Faktörü").pack(anchor="w", padx=10, pady=(4, 0))
        _slider(f, self._v_kontrast, 0.1, 2.0, 190).pack(fill="x", padx=10, pady=2)
        _btn(f, "Kontrast Uygula", self.op_kontrast).pack(
            fill="x", padx=10, pady=(2, 4))

        row2 = ctk.CTkFrame(f, fg_color="transparent")
        row2.pack(fill="x", padx=8, pady=(0, 8))
        _btn(row2, "Çıkar (ori-mev)", self.op_cikar).pack(
            side="left", expand=True, fill="x", padx=2)
        _btn(row2, "Çarp (ori×mev)", self.op_carp).pack(
            side="left", expand=True, fill="x", padx=2)

    def _filtre_bolum(self, parent):
        f = _section(parent, "4 · Filtreleme & Konvolüsyon")

        self._v_filtre_boyut = ctk.IntVar(value=3)
        _lbl(f, "Filtre Boyutu (piksel)").pack(anchor="w", padx=10)
        _slider(f, self._v_filtre_boyut, 3, 15, 6).pack(fill="x", padx=10, pady=2)

        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(2, 4))
        _btn(row, "Mean",   self.op_mean).pack(
            side="left", expand=True, fill="x", padx=2)
        _btn(row, "Median", self.op_median).pack(
            side="left", expand=True, fill="x", padx=2)

        self._v_motion_uzun = ctk.IntVar(value=10)
        self._v_motion_aci  = ctk.DoubleVar(value=0)
        _lbl(f, "Motion Uzunluk").pack(anchor="w", padx=10, pady=(4, 0))
        _slider(f, self._v_motion_uzun, 3, 30, 27).pack(fill="x", padx=10, pady=2)
        _lbl(f, "Motion Açısı (°)").pack(anchor="w", padx=10)
        _slider(f, self._v_motion_aci, 0, 180, 180).pack(fill="x", padx=10, pady=2)
        _btn(f, "Motion Filtre Uygula", self.op_motion).pack(
            fill="x", padx=10, pady=(2, 8))

    def _gurultu_bolum(self, parent):
        f = _section(parent, "5 · Gürültü Analizi (Salt & Pepper)")

        self._v_gurultu = ctk.DoubleVar(value=0.05)
        _lbl(f, "Gürültü Yoğunluğu").pack(anchor="w", padx=10)
        _slider(f, self._v_gurultu, 0.01, 0.30, 29).pack(fill="x", padx=10, pady=2)

        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(2, 8))
        _btn(row, "Gürültü Ekle",      self.op_gurultu_ekle).pack(
            side="left", expand=True, fill="x", padx=2)
        _btn(row, "Temizle (Median)", self.op_gurultu_temizle).pack(
            side="left", expand=True, fill="x", padx=2)

    def _kenar_bolum(self, parent):
        f = _section(parent, "6 · Segmentasyon & Kenar (Canny)")

        self._v_canny_lo = ctk.IntVar(value=50)
        self._v_canny_hi = ctk.IntVar(value=150)
        _lbl(f, "Düşük Eşik").pack(anchor="w", padx=10)
        _slider(f, self._v_canny_lo, 0, 255, 255).pack(fill="x", padx=10, pady=2)
        _lbl(f, "Yüksek Eşik").pack(anchor="w", padx=10)
        _slider(f, self._v_canny_hi, 0, 255, 255).pack(fill="x", padx=10, pady=2)
        _btn(f, "Canny Kenar Tespiti", self.op_canny).pack(
            fill="x", padx=10, pady=(2, 8))

    def _morfo_bolum(self, parent):
        f = _section(parent, "7 · Morfolojik İşlemler")

        self._v_morfo_boyut = ctk.IntVar(value=3)
        _lbl(f, "Yapısal Eleman Boyutu").pack(anchor="w", padx=10)
        _slider(f, self._v_morfo_boyut, 3, 11, 4).pack(fill="x", padx=10, pady=2)

        row1 = ctk.CTkFrame(f, fg_color="transparent")
        row1.pack(fill="x", padx=8, pady=2)
        _btn(row1, "Genişlet",  lambda: self.op_morfo("genisle")).pack(
            side="left", expand=True, fill="x", padx=2)
        _btn(row1, "Aşındır",  lambda: self.op_morfo("asin")).pack(
            side="left", expand=True, fill="x", padx=2)

        row2 = ctk.CTkFrame(f, fg_color="transparent")
        row2.pack(fill="x", padx=8, pady=(0, 8))
        _btn(row2, "Aç (Ero→Dil)",  lambda: self.op_morfo("ac")).pack(
            side="left", expand=True, fill="x", padx=2)
        _btn(row2, "Kapat (Dil→Ero)", lambda: self.op_morfo("kapat")).pack(
            side="left", expand=True, fill="x", padx=2)

    # ══════════════════════════════════════════════════════════════════════════
    # GÖRÜNTÜ YÖNETİMİ
    # ══════════════════════════════════════════════════════════════════════════

    def goruntu_ac(self):
        yol = filedialog.askopenfilename(
            title="Belge Görüntüsü Seç",
            filetypes=[("Görüntü", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp")]
        )
        if not yol:
            return
        bgr = cv2.imread(yol)
        if bgr is None:
            messagebox.showerror("Hata", f"Görüntü okunamadı:\n{yol}")
            return
        self.orijinal = bgr[:, :, ::-1].copy()
        self.mevcut   = self.orijinal.copy()
        H, W = self.orijinal.shape[:2]
        self.kose_img = np.array(
            [[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float64)
        self._gecmis.clear()
        self._kanvasi_yenile()
        self._durum(f"Yüklendi: {os.path.basename(yol)}  —  {W}×{H} piksel")

    def _kanvasi_yenile(self):
        if self.mevcut is None:
            return
        cw = self.kanvas.winfo_width()
        ch = self.kanvas.winfo_height()
        if cw < 10 or ch < 10:
            cw, ch = 800, 580

        H, W = self.mevcut.shape[:2]
        s = min(cw / W, ch / H, 1.0)
        yw, yh = max(int(W * s), 1), max(int(H * s), 1)

        self._goruntu_olcek  = s
        self._goruntu_offset = ((cw - yw) // 2, (ch - yh) // 2)

        pil = Image.fromarray(self.mevcut).resize((yw, yh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)

        self.kanvas.delete("all")
        ox, oy = self._goruntu_offset
        self.kanvas.create_image(ox, oy, anchor="nw", image=self._photo)
        self._kose_ciz()

    def _kose_ciz(self):
        if self.kose_img is None:
            return
        pts = self._img2cv(self.kose_img)

        # Çokgen
        self.kanvas.create_polygon(
            [v for p in pts for v in p],
            outline=C["accent"], fill="", width=2, dash=(6, 4))

        renk = [C["red"], C["yellow"], C["green"], "#a371f7"]
        isim = ["Sol Üst", "Sağ Üst", "Sağ Alt", "Sol Alt"]
        r = 9
        for i, (cx, cy) in enumerate(pts):
            self.kanvas.create_oval(
                cx-r, cy-r, cx+r, cy+r,
                fill=renk[i], outline="white", width=2, tags=f"k{i}")
            self.kanvas.create_text(
                cx, cy - r - 9, text=isim[i],
                fill=renk[i], font=("Segoe UI", 8, "bold"))

    def _img2cv(self, pts):
        ox, oy = self._goruntu_offset
        s = self._goruntu_olcek
        return [(int(x * s + ox), int(y * s + oy)) for x, y in pts]

    def _cv2img(self, cx, cy):
        ox, oy = self._goruntu_offset
        s = self._goruntu_olcek or 1
        return (cx - ox) / s, (cy - oy) / s

    # ── Köşe sürükleme ────────────────────────────────────────────────────────

    def _surukle_baslat(self, e):
        if self.kose_img is None:
            return
        self._surukle_idx = None
        pts = self._img2cv(self.kose_img)
        for i, (cx, cy) in enumerate(pts):
            if abs(e.x - cx) < 14 and abs(e.y - cy) < 14:
                self._surukle_idx = i
                break

    def _surukle_devam(self, e):
        if self._surukle_idx is None or self.orijinal is None:
            return
        ix, iy = self._cv2img(e.x, e.y)
        H, W = self.orijinal.shape[:2]
        self.kose_img[self._surukle_idx] = [
            np.clip(ix, 0, W-1), np.clip(iy, 0, H-1)]
        self.kanvas.delete("all")
        ox, oy = self._goruntu_offset
        self.kanvas.create_image(ox, oy, anchor="nw", image=self._photo)
        self._kose_ciz()
        self._buyutec_goster(e.x, e.y, ix, iy)

    def _surukle_bitis(self, _):
        self._surukle_idx = None
        self.kanvas.delete("buyutec")

    # ══════════════════════════════════════════════════════════════════════════
    # CAMSCANNER İŞLEMLERİ
    # ══════════════════════════════════════════════════════════════════════════

    def oto_kose(self):
        if not self._goruntu_var():
            return
        self._durum("Canny ile köşe tespiti yapılıyor...")
        self.update_idletasks()

        gri   = self.gi.temel.griye_donustur(self.orijinal)
        kucuk = self.gi.geometri.olcekle(gri, 0.3)
        kenar = self.gi.kenar.canny_kenar(kucuk, 30, 100)

        ys_i, xs_i = np.where(kenar > 127)
        if xs_i.size == 0:
            return self._durum("Kenar bulunamadı.")

        s   = 1.0 / 0.3
        H, W = self.orijinal.shape[:2]
        m   = 10  # kenar marjı

        def _nokta(idx):
            return [
                float(np.clip(xs_i[idx] * s, m, W - m)),
                float(np.clip(ys_i[idx] * s, m, H - m))
            ]

        sk = xs_i + ys_i
        sk2 = xs_i - ys_i
        self.kose_img = np.array([
            _nokta(np.argmin(sk)),   # sol üst
            _nokta(np.argmax(sk2)),  # sağ üst
            _nokta(np.argmax(sk)),   # sağ alt
            _nokta(np.argmin(sk2)),  # sol alt
        ], dtype=np.float64)

        self._kanvasi_yenile()
        self._durum("Otomatik köşe tespiti tamamlandı — sürükleyerek ince ayar yapabilirsin.")

    def perspektif_duzelt(self):
        if not self._goruntu_var():
            return
        pts = self.kose_img.astype(np.float64)
        gen = int(max(
            np.linalg.norm(pts[1] - pts[0]),
            np.linalg.norm(pts[2] - pts[3])))
        yuk = int(max(
            np.linalg.norm(pts[3] - pts[0]),
            np.linalg.norm(pts[2] - pts[1])))

        if gen < 10 or yuk < 10:
            return self._durum("Köşe noktaları çok yakın.")

        self._durum(f"Perspektif düzeltiliyor → {gen}×{yuk} ...")
        self.update_idletasks()

        hedef = np.array(
            [[0, 0], [gen-1, 0], [gen-1, yuk-1], [0, yuk-1]], dtype=np.float64)
        H_mat = self.gi.perspektif.homografi_hesapla(pts, hedef)
        duz   = self.gi.perspektif.perspektif_donustur(
            self.orijinal, H_mat, gen, yuk)

        self._durum_kaydet()
        self.mevcut   = duz
        self.kose_img = np.array(
            [[0, 0], [gen-1, 0], [gen-1, yuk-1], [0, yuk-1]], dtype=np.float64)
        self._kanvasi_yenile()
        self._durum(f"Perspektif düzeltme tamamlandı — {gen}×{yuk}")

    def tam_iyilestir(self):
        if not self._goruntu_var():
            return
        self._durum("Tam iyileştirme: Histogram Germe → Sauvola Eşikleme → Morfolojik Açma ...")
        self.update_idletasks()

        gri      = self.gi.temel.griye_donustur(self.mevcut)
        gerilmis = self.gi.kontrast.histogram_ger(gri)
        sauvola  = self.gi.adaptif.sauvola(gerilmis, pencere=25, k=0.2)
        acilmis  = self.gi.morfoloji.ac(sauvola, self.gi.morfoloji.kare_cekirdek(3))

        self._durum_kaydet()
        self.mevcut = np.stack([acilmis] * 3, axis=2)
        self._kanvasi_yenile()
        self._durum("Tam iyileştirme tamamlandı.")

    # ══════════════════════════════════════════════════════════════════════════
    # TEKNİK İŞLEM OPERASYONLARI
    # ══════════════════════════════════════════════════════════════════════════

    def _gri(self):
        return self.gi.temel.griye_donustur(self.mevcut)

    def _guncelle(self, yeni: np.ndarray, msg: str = ""):
        self._durum_kaydet()
        self.mevcut = yeni
        self._kanvasi_yenile()
        if msg:
            self._durum(msg)

    def _goruntu_var(self) -> bool:
        if self.mevcut is None:
            self._durum("Önce görüntü yükleyin!")
            return False
        return True

    def _g3(self, kanal):
        """Tek kanallı → 3 kanallı."""
        return np.stack([kanal] * 3, axis=2)

    # ── Temel ─────────────────────────────────────────────────────────────────

    def op_gri(self):
        if not self._goruntu_var(): return
        self._guncelle(self._g3(self._gri()), "Gri dönüşüm uygulandı.")

    def op_binary(self):
        if not self._goruntu_var(): return
        esik = self._v_binary_esik.get()
        b = self.gi.temel.binary_donustur(self._gri(), esik)
        self._guncelle(self._g3(b), f"Binary dönüşüm — eşik={esik}")

    def op_hsv(self):
        if not self._goruntu_var(): return
        self._guncelle(self.gi.temel.rgb_to_hsv(self.mevcut), "RGB → HSV")

    def op_lab(self):
        if not self._goruntu_var(): return
        lab = self.gi.temel.rgb_to_lab(self.mevcut)
        L = np.clip(lab[:, :, 0] / 100.0 * 255, 0, 255).astype(np.uint8)
        a = np.clip(lab[:, :, 1] + 128,          0, 255).astype(np.uint8)
        b = np.clip(lab[:, :, 2] + 128,          0, 255).astype(np.uint8)
        self._guncelle(np.stack([L, a, b], axis=2), "RGB → LAB (görselleştirilmiş)")

    # ── Geometrik ─────────────────────────────────────────────────────────────

    def op_dondur(self):
        if not self._goruntu_var(): return
        aci = self._v_donus.get()
        self._durum(f"Döndürülüyor {aci:.1f}° ...")
        self.update_idletasks()
        self._guncelle(self.gi.geometri.dondur(self.mevcut, aci),
                       f"Döndürme: {aci:.1f}°")

    def op_olcekle(self):
        if not self._goruntu_var(): return
        s = self._v_olcek.get()
        self._durum(f"Ölçekleniyor ×{s:.2f} ...")
        self.update_idletasks()
        sonuc = self.gi.geometri.olcekle(self.mevcut, s)
        H, W = sonuc.shape[:2]
        self.kose_img = np.array(
            [[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float64)
        self._guncelle(sonuc, f"Ölçekleme ×{s:.2f} → {W}×{H}")

    # ── Kontrast ──────────────────────────────────────────────────────────────

    def op_hist_ger(self):
        if not self._goruntu_var(): return
        self._guncelle(self._g3(self.gi.kontrast.histogram_ger(self._gri())),
                       "Histogram germe uygulandı.")

    def op_hist_grafik(self):
        if not self._goruntu_var(): return
        hist = self.gi.kontrast.histogram_hesapla(self._gri())
        self._histogram_goster(hist)

    def op_kontrast(self):
        if not self._goruntu_var(): return
        f = self._v_kontrast.get()
        self._guncelle(self._g3(self.gi.kontrast.kontrast_azalt(self._gri(), f)),
                       f"Kontrast faktörü: {f:.2f}")

    def op_cikar(self):
        if not self._goruntu_var(): return
        g_ori = self.gi.temel.griye_donustur(self.orijinal)
        self._guncelle(self._g3(self.gi.kontrast.aritmetik_cikar(g_ori, self._gri())),
                       "Aritmetik çıkarma: orijinal − mevcut")

    def op_carp(self):
        if not self._goruntu_var(): return
        g_ori = self.gi.temel.griye_donustur(self.orijinal)
        self._guncelle(self._g3(self.gi.kontrast.aritmetik_carp(g_ori, self._gri())),
                       "Aritmetik çarpma: orijinal × mevcut")

    # ── Filtreler ─────────────────────────────────────────────────────────────

    def _tek_boyut(self, v):
        b = v.get()
        return b if b % 2 == 1 else b + 1

    def op_mean(self):
        if not self._goruntu_var(): return
        b = self._tek_boyut(self._v_filtre_boyut)
        self._durum(f"Mean filtre {b}×{b} ...")
        self.update_idletasks()
        self._guncelle(self._g3(self.gi.filtre.mean_filtre(self._gri(), b)),
                       f"Mean filtre {b}×{b} uygulandı.")

    def op_median(self):
        if not self._goruntu_var(): return
        b = self._tek_boyut(self._v_filtre_boyut)
        self._durum(f"Median filtre {b}×{b} ...")
        self.update_idletasks()
        self._guncelle(self._g3(self.gi.filtre.median_filtre(self._gri(), b)),
                       f"Median filtre {b}×{b} uygulandı.")

    def op_motion(self):
        if not self._goruntu_var(): return
        u = self._v_motion_uzun.get()
        a = self._v_motion_aci.get()
        self._durum(f"Motion filtre (uzunluk={u}, açı={a:.0f}°) ...")
        self.update_idletasks()
        self._guncelle(self._g3(self.gi.filtre.motion_filtre(self._gri(), u, a)),
                       f"Motion filtre uygulandı.")

    # ── Gürültü ───────────────────────────────────────────────────────────────

    def op_gurultu_ekle(self):
        if not self._goruntu_var(): return
        y = self._v_gurultu.get()
        g = self.gi.gurultu.salt_pepper_ekle(self._gri(), y)
        self._guncelle(self._g3(g), f"Salt & Pepper eklendi (%{y*100:.0f})")

    def op_gurultu_temizle(self):
        if not self._goruntu_var(): return
        self._durum("Median ile gürültü temizleniyor ...")
        self.update_idletasks()
        self._guncelle(self._g3(self.gi.gurultu.salt_pepper_temizle(self._gri(), 3)),
                       "Gürültü temizlendi.")

    # ── Kenar ─────────────────────────────────────────────────────────────────

    def op_canny(self):
        if not self._goruntu_var(): return
        lo, hi = self._v_canny_lo.get(), self._v_canny_hi.get()
        self._durum(f"Canny kenar tespiti (lo={lo}, hi={hi}) ...")
        self.update_idletasks()
        self._guncelle(self._g3(self.gi.kenar.canny_kenar(self._gri(), lo, hi)),
                       "Canny tamamlandı.")

    # ── Morfoloji ─────────────────────────────────────────────────────────────

    def op_morfo(self, islem: str):
        if not self._goruntu_var(): return
        b = self._tek_boyut(self._v_morfo_boyut)
        ck = self.gi.morfoloji.kare_cekirdek(b)
        self._durum(f"Morfolojik '{islem}' {b}×{b} ...")
        self.update_idletasks()
        islemler = {
            "genisle": self.gi.morfoloji.genisle,
            "asin":    self.gi.morfoloji.asin,
            "ac":      self.gi.morfoloji.ac,
            "kapat":   self.gi.morfoloji.kapat,
        }
        self._guncelle(self._g3(islemler[islem](self._gri(), ck)),
                       f"Morfoloji '{islem}' {b}×{b} tamamlandı.")

    # ══════════════════════════════════════════════════════════════════════════
    # KAYDETME
    # ══════════════════════════════════════════════════════════════════════════

    def jpg_kaydet(self):
        if not self._goruntu_var(): return
        os.makedirs("cikti", exist_ok=True)
        yol = filedialog.asksaveasfilename(
            initialdir="cikti", defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            title="JPG / PNG olarak kaydet"
        )
        if not yol:
            return
        cv2.imwrite(yol, self.mevcut[:, :, ::-1])
        self._durum(f"Kaydedildi: {os.path.basename(yol)}")

    def pdf_kaydet(self):
        if not self._goruntu_var(): return
        os.makedirs("cikti", exist_ok=True)
        yol = filedialog.asksaveasfilename(
            initialdir="cikti", defaultextension=".pdf",
            filetypes=[("PDF", "*.pdf")],
            title="PDF olarak kaydet"
        )
        if not yol:
            return
        Image.fromarray(self.mevcut).convert("RGB").save(yol, "PDF")
        self._durum(f"PDF kaydedildi: {os.path.basename(yol)}")

    # ══════════════════════════════════════════════════════════════════════════
    # KARŞILAŞTIRMA PENCERESİ
    # ══════════════════════════════════════════════════════════════════════════

    def karsilastirma_ac(self):
        if self.orijinal is None or self.mevcut is None:
            return self._durum("Karşılaştırma için görüntü gerekli.")

        win = ctk.CTkToplevel(self)
        win.title("Karşılaştırma — Orijinal / İşlenmiş")
        win.geometry("1100x580")
        win.configure(fg_color=C["bg"])
        win.grab_set()

        win.columnconfigure(0, weight=1)
        win.columnconfigure(1, weight=1)
        win.rowconfigure(1, weight=1)

        for col, (baslik, renk) in enumerate([
            ("Orijinal",  C["text"]),
            ("İşlenmiş",  C["green"])
        ]):
            ctk.CTkLabel(win, text=baslik,
                         font=ctk.CTkFont(size=13, weight="bold"),
                         text_color=renk).grid(row=0, column=col, pady=6)

        cv_ori = tk.Canvas(win, bg=C["bg"], highlightthickness=0)
        cv_isi = tk.Canvas(win, bg=C["bg"], highlightthickness=0)
        cv_ori.grid(row=1, column=0, sticky="nsew", padx=(8, 4), pady=8)
        cv_isi.grid(row=1, column=1, sticky="nsew", padx=(4, 8), pady=8)

        def _yukle(canvas, arr):
            canvas.update_idletasks()
            cw = canvas.winfo_width()  or 500
            ch = canvas.winfo_height() or 460
            H, W = arr.shape[:2]
            s = min(cw / W, ch / H, 1.0)
            yw, yh = max(int(W*s), 1), max(int(H*s), 1)
            pil   = Image.fromarray(arr).resize((yw, yh), Image.LANCZOS)
            photo = ImageTk.PhotoImage(pil)
            canvas._photo = photo
            canvas.create_image((cw-yw)//2, (ch-yh)//2,
                                 anchor="nw", image=photo)

        win.after(150, lambda: _yukle(cv_ori, self.orijinal))
        win.after(150, lambda: _yukle(cv_isi, self.mevcut))

    # ══════════════════════════════════════════════════════════════════════════
    # HİSTOGRAM GRAFİĞİ
    # ══════════════════════════════════════════════════════════════════════════

    def _histogram_goster(self, hist: np.ndarray):
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4), facecolor=C["bg"])
            ax.set_facecolor(C["panel"])
            ax.bar(range(256), hist, color=C["accent"], width=1.0, alpha=0.85)
            ax.set_xlabel("Piksel Değeri", color=C["text"])
            ax.set_ylabel("Piksel Sayısı",  color=C["text"])
            ax.set_title("Gri Histogram", color=C["text"],
                         fontsize=13, fontweight="bold")
            ax.tick_params(colors=C["muted"])
            for sp in ax.spines.values():
                sp.set_edgecolor(C["border"])
            plt.tight_layout()
            plt.show()
        except ImportError:
            self._durum("Histogram grafiği için matplotlib gerekli.")

    # ══════════════════════════════════════════════════════════════════════════
    # SIFIRLAMA & YARDIMCI
    # ══════════════════════════════════════════════════════════════════════════

    def sifirla(self):
        if self.orijinal is None:
            return
        self._gecmis.clear()
        self.mevcut = self.orijinal.copy()
        H, W = self.orijinal.shape[:2]
        self.kose_img = np.array(
            [[0,0],[W-1,0],[W-1,H-1],[0,H-1]], dtype=np.float64)
        self._kanvasi_yenile()
        self._durum("Orijinal görüntüye dönüldü.")

    # ── Geri Al ───────────────────────────────────────────────────────────────

    def _durum_kaydet(self):
        """İşlem öncesi mevcut durumu stack'e ekler."""
        if self.mevcut is None:
            return
        kose_kop = self.kose_img.copy() if self.kose_img is not None else None
        self._gecmis.append((self.mevcut.copy(), kose_kop))
        if len(self._gecmis) > self._gecmis_limit:
            self._gecmis.pop(0)

    def geri_al(self):
        if not self._gecmis:
            return self._durum("Geri alınacak işlem yok.")
        self.mevcut, self.kose_img = self._gecmis.pop()
        self._kanvasi_yenile()
        kalan = len(self._gecmis)
        self._durum(f"Geri alındı.{f'  ({kalan} adım daha geri alınabilir)' if kalan else ''}")

    # ── Büyüteç ───────────────────────────────────────────────────────────────

    def _buyutec_goster(self, cv_x: int, cv_y: int,
                        img_x: float, img_y: float):
        """Sürükleme sırasında köşe noktasının çevresini büyütülmüş gösterir."""
        if self.mevcut is None:
            return

        MAG_PX   = 160    # Canvas'ta gösterilecek büyüteç boyutu (kare, px)
        YAMA_PX  = 48     # Kaynak görüntüden alınan patch yarıçapı (px)

        H, W = self.mevcut.shape[:2]
        x, y = int(img_x), int(img_y)
        x1 = max(0,   x - YAMA_PX)
        y1 = max(0,   y - YAMA_PX)
        x2 = min(W-1, x + YAMA_PX)
        y2 = min(H-1, y + YAMA_PX)

        yama = self.mevcut[y1:y2+1, x1:x2+1]
        if yama.size == 0:
            return

        pil = Image.fromarray(yama).resize((MAG_PX, MAG_PX), Image.NEAREST)

        # Çapraz kıl (crosshair) ve çerçeve
        from PIL import ImageDraw
        d = ImageDraw.Draw(pil)
        m = MAG_PX // 2
        d.line([(m - 18, m), (m + 18, m)], fill=(255, 60, 60), width=2)
        d.line([(m, m - 18), (m, m + 18)], fill=(255, 60, 60), width=2)
        d.rectangle([0, 0, MAG_PX - 1, MAG_PX - 1],
                    outline=(88, 166, 255), width=3)

        self._mag_photo = ImageTk.PhotoImage(pil)

        # Büyüteç konumu: imleçten kaçacak şekilde hesapla
        cw = self.kanvas.winfo_width()
        ch = self.kanvas.winfo_height()
        bx = cv_x + 22 if cv_x + 22 + MAG_PX < cw else cv_x - MAG_PX - 22
        by = cv_y - MAG_PX - 22 if cv_y - MAG_PX - 22 > 0 else cv_y + 22

        self.kanvas.delete("buyutec")
        self.kanvas.create_image(bx, by, anchor="nw",
                                 image=self._mag_photo, tags="buyutec")
        self.kanvas.create_text(
            bx + MAG_PX // 2, by + MAG_PX + 11,
            text=f"({int(img_x)}, {int(img_y)})",
            fill=C["accent"], font=("Segoe UI", 9, "bold"),
            tags="buyutec")

    def _durum(self, mesaj: str):
        self._durum_var.set(mesaj)
        self.update_idletasks()


# ── Giriş noktası ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = BelgeTaramaApp()
    app.mainloop()
