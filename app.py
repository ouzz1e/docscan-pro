"""
Akıllı Belge Tarama ve İyileştirme Sistemi — Masaüstü Uygulaması
Çalıştırmak için: python app.py
"""

import os
import threading
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import customtkinter as ctk
import numpy as np
from PIL import Image, ImageDraw, ImageTk

try:
    import pillow_heif
    pillow_heif.register_heif_opener()
    _HEIC_DESTEKLI = True
except ImportError:
    _HEIC_DESTEKLI = False

try:
    from tkinterdnd2 import DND_FILES
    _DND_DESTEKLI = True
except ImportError:
    _DND_DESTEKLI = False

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


def _slider(parent, var, lo, hi, steps, command=None):
    """
    Slider + anlık değer kutusu (iki yönlü bağlama).
    Kullanıcı slider'ı sürüklediğinde kutu güncellenir;
    kutuya el ile değer girilip Enter/Tab yapıldığında slider o konuma gider.
    command: slider hareket ettikçe çağrılacak callback(value).
    """
    frame = ctk.CTkFrame(parent, fg_color="transparent")

    is_int = isinstance(var, tk.IntVar)

    def _fmt(v):
        return str(int(round(v))) if is_int else (
            f"{v:.2f}" if abs(float(lo)) < 10 or abs(float(hi)) <= 10 else f"{v:.1f}"
        )

    sl = ctk.CTkSlider(
        frame, from_=lo, to=hi, number_of_steps=steps,
        variable=var, button_color=C["accent"],
        progress_color=C["highlight"], height=16,
        command=command,
    )
    sl.pack(side="left", fill="x", expand=True, padx=(0, 6))

    entry_var = tk.StringVar(value=_fmt(var.get()))
    entry = ctk.CTkEntry(
        frame, textvariable=entry_var,
        width=58, height=26,
        font=ctk.CTkFont(size=11),
        justify="center",
        fg_color=C["card"],
        border_color=C["border"],
        text_color=C["text"],
    )
    entry.pack(side="left")

    # Slider → giriş kutusu
    def _on_var(*_):
        entry_var.set(_fmt(var.get()))
    var.trace_add("write", _on_var)

    # Giriş kutusu → slider  (Enter veya odak kaybolunca)
    def _on_entry(*_):
        try:
            v = float(entry_var.get().replace(",", "."))
            v = max(float(lo), min(float(hi), v))
            if is_int:
                v = int(round(v))
            var.set(v)
            entry_var.set(_fmt(v))
            if command:
                command(v)
        except ValueError:
            entry_var.set(_fmt(var.get()))

    entry.bind("<Return>",   _on_entry)
    entry.bind("<FocusOut>", _on_entry)

    return frame


def _bilgi_goster(baslik: str, metin: str):
    """Bölüm açıklama popup'ı."""
    win = ctk.CTkToplevel()
    win.title(baslik)
    win.geometry("420x260")
    win.resizable(False, False)
    win.configure(fg_color=C["panel"])
    win.grab_set()
    win.lift()
    win.after(50, win.lift)

    ctk.CTkLabel(
        win, text=baslik,
        font=ctk.CTkFont(size=12, weight="bold"),
        text_color=C["accent"],
    ).pack(padx=20, pady=(18, 6), anchor="w")

    ctk.CTkLabel(
        win, text=metin,
        font=ctk.CTkFont(size=11),
        text_color=C["text"],
        justify="left",
        wraplength=380,
        anchor="nw",
    ).pack(padx=20, pady=(0, 10), fill="x")

    ctk.CTkButton(
        win, text="Tamam", command=win.destroy,
        fg_color=C["highlight"], hover_color="#388bfd",
        width=90, height=30, corner_radius=6,
    ).pack(pady=(0, 18))


def _section(parent, title, bilgi: str = None):
    f = ctk.CTkFrame(parent, fg_color=C["card"], corner_radius=8)
    f.pack(fill="x", padx=6, pady=4)

    baslik_satiri = ctk.CTkFrame(f, fg_color="transparent")
    baslik_satiri.pack(fill="x", padx=10, pady=(8, 2))

    ctk.CTkLabel(
        baslik_satiri, text=title,
        font=ctk.CTkFont(size=11, weight="bold"),
        text_color=C["accent"],
    ).pack(side="left")

    if bilgi:
        _t, _b = title, bilgi
        ctk.CTkButton(
            baslik_satiri, text=" ? ", width=22, height=20,
            fg_color=C["border"], hover_color="#3d444d",
            corner_radius=10, font=ctk.CTkFont(size=9),
            command=lambda: _bilgi_goster(_t, _b),
        ).pack(side="right")

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

        # Canlı döndürme için taban görüntü (slider hareket edince ayarlanır)
        self._rot_base: np.ndarray | None = None

        # Arka plan iş parçacığı kilidi (çakışan işlemleri engeller)
        self._isleniyor = False

        # Canvas zoom / pan
        self._zoom_faktor   = 1.0
        self._pan_offset    = [0, 0]
        self._pan_baslangic = None   # (ex, ey, pan_x0, pan_y0)

        self._pencere_kur()
        self._ui_kur()
        self._kisayollar_kur()

        # DnD arka uç başlat (tkinterdnd2 tüm widget'lara dnd metotları ekler)
        if _DND_DESTEKLI:
            try:
                self.tk.call('package', 'require', 'tkdnd')
            except Exception:
                pass

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
                font=ctk.CTkFont(size=11),
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

        # Scroll zoom
        self.kanvas.bind("<MouseWheel>", self._scroll_zoom)   # Windows / macOS
        self.kanvas.bind("<Button-4>",   self._scroll_zoom)   # Linux yukarı
        self.kanvas.bind("<Button-5>",   self._scroll_zoom)   # Linux aşağı

        # Orta tuş pan
        self.kanvas.bind("<ButtonPress-2>",   self._pan_baslat)
        self.kanvas.bind("<B2-Motion>",       self._pan_devam)
        self.kanvas.bind("<ButtonRelease-2>", self._pan_bitis)

        # Drag & drop
        if _DND_DESTEKLI:
            try:
                self.kanvas.drop_target_register(DND_FILES)
                self.kanvas.dnd_bind("<<Drop>>", self._drag_drop_ac)
            except Exception:
                pass

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
        # macOS: scroll frame'in dahili canvas'ı click event'ini yutmasın
        try:
            scroll._parent_canvas.configure(takefocus=False)
        except Exception:
            pass

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
                font=ctk.CTkFont(size=11),
            ).pack(fill="x", padx=10, pady=3)

        ctk.CTkFrame(f, height=1, fg_color="#1f6feb").pack(fill="x", padx=10, pady=(6, 2))

        self._v_sauvola_pencere = ctk.IntVar(value=25)
        self._v_sauvola_k = ctk.DoubleVar(value=0.2)
        _lbl(f, "Sauvola Pencere  (küçük = ince detay, büyük = geniş bölge)",
             color=C["muted"]).pack(anchor="w", padx=10)
        _slider(f, self._v_sauvola_pencere, 7, 75, 34).pack(fill="x", padx=10, pady=2)
        _lbl(f, "Sauvola k  (düşük = koyu metin, yüksek = açık metin)",
             color=C["muted"]).pack(anchor="w", padx=10)
        _slider(f, self._v_sauvola_k, 0.05, 0.50, 45).pack(fill="x", padx=10, pady=(2, 6))

    def _temel_bolum(self, parent):
        f = _section(parent, "1 · Temel Dönüşümler", bilgi=(
            "Renk uzayı dönüşümleri:\n"
            "• Gri — ITU-R BT.601 formülüyle tek kanallı gri tonlama\n"
            "• Binary — Eşik değerine göre siyah-beyaz (0 veya 255)\n"
            "• HSV — Ton / Doygunluk / Parlaklık ayrıştırması\n"
            "• LAB — İnsan gözü algısına yakın CIE L*a*b* renk uzayı"
        ))

        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(0, 4))
        for text, cmd in [("Gri", self.op_gri), ("Binary", self.op_binary),
                           ("HSV", self.op_hsv),  ("LAB",  self.op_lab)]:
            _btn(row, text, cmd).pack(side="left", expand=True, fill="x", padx=2)

        self._v_binary_esik = ctk.IntVar(value=128)
        _lbl(f, "Binary Eşik  (0 = siyah, 255 = beyaz)").pack(anchor="w", padx=10)
        _slider(f, self._v_binary_esik, 0, 255, 255).pack(fill="x", padx=10, pady=(0, 8))

    def _geometri_bolum(self, parent):
        f = _section(parent, "2 · Geometrik İşlemler", bilgi=(
            "Görüntü boyutu ve açısını değiştirir:\n"
            "• Döndürme — Bilineer interpolasyonla döndürme; köşeler kırpılmaz, "
            "çıktı boyutu otomatik genişler. Slider hareket ettikçe önizleme gösterilir.\n"
            "• Ölçekleme — 1.0 = orijinal boyut, >1 büyütme, <1 küçültme."
        ))

        self._v_donus = ctk.DoubleVar(value=0)
        _lbl(f, "Döndürme Açısı  (−180° … +180°)").pack(anchor="w", padx=10)
        _slider(f, self._v_donus, -180, 180, 360,
                command=self._canli_dondur_cb).pack(fill="x", padx=10, pady=2)
        _btn(f, "Döndürmeyi Uygula  (tam çözünürlük)", self.op_dondur).pack(
            fill="x", padx=10, pady=(2, 6))

        self._v_olcek = ctk.DoubleVar(value=1.0)
        _lbl(f, "Ölçek Faktörü  (0.1 = %10 … 3.0 = %300)").pack(anchor="w", padx=10)
        _slider(f, self._v_olcek, 0.1, 3.0, 290).pack(fill="x", padx=10, pady=2)
        _btn(f, "Ölçeklemeyi Uygula", self.op_olcekle).pack(
            fill="x", padx=10, pady=(2, 8))

    def _kontrast_bolum(self, parent):
        f = _section(parent, "3 · İstatistiksel & Kontrast", bilgi=(
            "Parlaklık ve kontrast ayarları:\n"
            "• Hist. Ger — Tüm piksel değerlerini 0–255 aralığına yay (kontrast artırma).\n"
            "• Hist. Grafik — Gri tonlama histogramını matplotlib ile göster.\n"
            "• Kontrast Faktörü: <1.0 azaltır, 1.0 değişmez, >1.0 artırır.\n"
            "• Çıkar — |Orijinal − Mevcut| fark görüntüsü.\n"
            "• Çarp — Orijinal × Mevcut normalleştirilmiş çarpımı."
        ))

        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(0, 4))
        _btn(row, "Hist. Ger",    self.op_hist_ger).pack(
            side="left", expand=True, fill="x", padx=2)
        _btn(row, "Hist. Grafik", self.op_hist_grafik).pack(
            side="left", expand=True, fill="x", padx=2)

        self._v_kontrast = ctk.DoubleVar(value=1.0)
        baslik_satiri = ctk.CTkFrame(f, fg_color="transparent")
        baslik_satiri.pack(fill="x", padx=10, pady=(4, 0))
        _lbl(baslik_satiri, "Kontrast Faktörü").pack(side="left")
        self._lbl_kontrast = ctk.CTkLabel(
            baslik_satiri, text="─ eşit ─  (1.00)",
            font=ctk.CTkFont(size=10), text_color=C["muted"])
        self._lbl_kontrast.pack(side="right")

        def _kontrast_guncelle(*_):
            v = self._v_kontrast.get()
            if v < 0.99:
                self._lbl_kontrast.configure(
                    text=f"◀ azalt  ({v:.2f})", text_color=C["yellow"])
            elif v > 1.01:
                self._lbl_kontrast.configure(
                    text=f"artır ▶  ({v:.2f})", text_color=C["green"])
            else:
                self._lbl_kontrast.configure(
                    text="─ eşit ─  (1.00)", text_color=C["muted"])
        self._v_kontrast.trace_add("write", _kontrast_guncelle)

        _slider(f, self._v_kontrast, 0.1, 2.0, 190).pack(fill="x", padx=10, pady=2)
        _btn(f, "Kontrast Uygula", self.op_kontrast).pack(
            fill="x", padx=10, pady=(2, 4))

        row2 = ctk.CTkFrame(f, fg_color="transparent")
        row2.pack(fill="x", padx=8, pady=(0, 8))
        _btn(row2, "Çıkar (ori−mev)", self.op_cikar).pack(
            side="left", expand=True, fill="x", padx=2)
        _btn(row2, "Çarp (ori×mev)", self.op_carp).pack(
            side="left", expand=True, fill="x", padx=2)

    def _filtre_bolum(self, parent):
        f = _section(parent, "4 · Filtreleme & Konvolüsyon", bilgi=(
            "Görüntüyü yumuşat veya bulanıklaştır:\n"
            "• Mean — Her pikseli komşularının ortalamasıyla değiştir (düz bulanıklık).\n"
            "• Median — Gürültüye karşı dayanıklı ortalama; salt & pepper için idealdir.\n"
            "• Motion — Belirli açıda hareket bulanıklığı simülasyonu.\n"
            "Filtre boyutu artıkça etki güçlenir; tek sayı kullan (3, 5, 7…)."
        ))

        self._v_filtre_boyut = ctk.IntVar(value=3)
        _lbl(f, "Filtre Boyutu  (3–15, tek sayı)").pack(anchor="w", padx=10)
        _slider(f, self._v_filtre_boyut, 3, 15, 6).pack(fill="x", padx=10, pady=2)

        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(2, 4))
        _btn(row, "Mean",   self.op_mean).pack(
            side="left", expand=True, fill="x", padx=2)
        _btn(row, "Median", self.op_median).pack(
            side="left", expand=True, fill="x", padx=2)

        self._v_motion_uzun = ctk.IntVar(value=10)
        self._v_motion_aci  = ctk.DoubleVar(value=0)
        _lbl(f, "Motion Uzunluğu  (3–30 piksel)").pack(anchor="w", padx=10, pady=(4, 0))
        _slider(f, self._v_motion_uzun, 3, 30, 27).pack(fill="x", padx=10, pady=2)
        _lbl(f, "Motion Açısı  (0° … 180°)").pack(anchor="w", padx=10)
        _slider(f, self._v_motion_aci, 0, 180, 180).pack(fill="x", padx=10, pady=2)
        _btn(f, "Motion Filtre Uygula", self.op_motion).pack(
            fill="x", padx=10, pady=(2, 8))

    def _gurultu_bolum(self, parent):
        f = _section(parent, "5 · Gürültü Analizi (Salt & Pepper)", bilgi=(
            "Tuz-Biber gürültüsü simülasyonu ve temizleme:\n"
            "• Gürültü Ekle — Rastgele beyaz (tuz) ve siyah (biber) pikseller ekler.\n"
            "• Temizle — Median filtre ile gürültüyü giderir.\n"
            "Yoğunluk: toplam etkilenen piksel oranı (0.01 = %1, 0.30 = %30)."
        ))

        self._v_gurultu = ctk.DoubleVar(value=0.05)
        _lbl(f, "Gürültü Yoğunluğu  (0.01 … 0.30)").pack(anchor="w", padx=10)
        _slider(f, self._v_gurultu, 0.01, 0.30, 29).pack(fill="x", padx=10, pady=2)

        row = ctk.CTkFrame(f, fg_color="transparent")
        row.pack(fill="x", padx=8, pady=(2, 8))
        _btn(row, "Gürültü Ekle",     self.op_gurultu_ekle).pack(
            side="left", expand=True, fill="x", padx=2)
        _btn(row, "Temizle (Median)", self.op_gurultu_temizle).pack(
            side="left", expand=True, fill="x", padx=2)

    def _kenar_bolum(self, parent):
        f = _section(parent, "6 · Segmentasyon & Kenar (Canny)", bilgi=(
            "Manuel Canny kenar tespiti (4 adım):\n"
            "1. Gaussian pürüzsüzleştirme\n"
            "2. Sobel gradyan hesabı (büyüklük + yön)\n"
            "3. Non-Maximum Suppression\n"
            "4. Çift eşikleme + histerezis\n"
            "Düşük < Yüksek eşik olmalı. Düşük artıkça daha fazla kenar seçilir."
        ))

        self._v_canny_lo = ctk.IntVar(value=50)
        self._v_canny_hi = ctk.IntVar(value=150)
        _lbl(f, "Düşük Eşik  (zayıf kenar sınırı)").pack(anchor="w", padx=10)
        _slider(f, self._v_canny_lo, 0, 255, 255).pack(fill="x", padx=10, pady=2)
        _lbl(f, "Yüksek Eşik  (güçlü kenar sınırı)").pack(anchor="w", padx=10)
        _slider(f, self._v_canny_hi, 0, 255, 255).pack(fill="x", padx=10, pady=2)
        _btn(f, "Canny Kenar Tespiti", self.op_canny).pack(
            fill="x", padx=10, pady=(2, 8))

    def _morfo_bolum(self, parent):
        f = _section(parent, "7 · Morfolojik İşlemler", bilgi=(
            "Binary görüntülerde şekil manipülasyonu:\n"
            "• Genişlet (Dilation) — Beyaz bölgeleri büyütür, yazıyı kalınlaştırır.\n"
            "• Aşındır (Erosion) — Beyaz bölgeleri küçültür, yazıyı inceltir.\n"
            "• Aç (Erosion→Dilation) — Küçük beyaz lekeleri siler, ana şekli korur.\n"
            "• Kapat (Dilation→Erosion) — Yazı içindeki küçük delikleri doldurur.\n"
            "Yapısal eleman boyutu artıkça etki güçlenir."
        ))

        self._v_morfo_boyut = ctk.IntVar(value=3)
        _lbl(f, "Yapısal Eleman Boyutu  (3–11, tek sayı)").pack(anchor="w", padx=10)
        _slider(f, self._v_morfo_boyut, 3, 11, 4).pack(fill="x", padx=10, pady=2)

        row1 = ctk.CTkFrame(f, fg_color="transparent")
        row1.pack(fill="x", padx=8, pady=2)
        _btn(row1, "Genişlet",  lambda: self.op_morfo("genisle")).pack(
            side="left", expand=True, fill="x", padx=2)
        _btn(row1, "Aşındır",  lambda: self.op_morfo("asin")).pack(
            side="left", expand=True, fill="x", padx=2)

        row2 = ctk.CTkFrame(f, fg_color="transparent")
        row2.pack(fill="x", padx=8, pady=(0, 8))
        _btn(row2, "Aç (Ero→Dil)",    lambda: self.op_morfo("ac")).pack(
            side="left", expand=True, fill="x", padx=2)
        _btn(row2, "Kapat (Dil→Ero)", lambda: self.op_morfo("kapat")).pack(
            side="left", expand=True, fill="x", padx=2)

    # ══════════════════════════════════════════════════════════════════════════
    # GÖRÜNTÜ YÖNETİMİ
    # ══════════════════════════════════════════════════════════════════════════

    def goruntu_ac(self):
        heic_filtre = " *.heic *.heif *.HEIC *.HEIF" if _HEIC_DESTEKLI else ""
        yol = filedialog.askopenfilename(
            title="Belge Görüntüsü Seç",
            filetypes=[("Görüntü", f"*.jpg *.jpeg *.png *.bmp *.tiff *.webp{heic_filtre}")]
        )
        if not yol:
            return
        self._goruntu_yukle(yol)

    def _goruntu_yukle(self, yol: str):
        """Verilen yoldan görüntüyü yükler (dosya diyaloğu ve drag & drop için ortak)."""
        bgr = cv2.imread(yol)
        if bgr is not None:
            self.orijinal = bgr[:, :, ::-1].copy()
        else:
            try:
                pil_img = Image.open(yol).convert("RGB")
                self.orijinal = np.array(pil_img)
            except Exception as ex:
                messagebox.showerror("Hata", f"Görüntü okunamadı:\n{yol}\n\n{ex}")
                return

        self.mevcut = self.orijinal.copy()
        self._rot_base = None
        H, W = self.orijinal.shape[:2]
        self.kose_img = np.array(
            [[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float64)
        self._gecmis.clear()
        self._zoom_faktor = 1.0
        self._pan_offset  = [0, 0]
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
        base_s = min(cw / W, ch / H)
        s  = base_s * self._zoom_faktor
        yw, yh = max(int(W * s), 1), max(int(H * s), 1)

        ox = (cw - yw) // 2 + self._pan_offset[0]
        oy = (ch - yh) // 2 + self._pan_offset[1]

        self._goruntu_olcek  = s
        self._goruntu_offset = (ox, oy)

        pil = Image.fromarray(self.mevcut).resize((yw, yh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)

        self.kanvas.delete("all")
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
        try:
            sf = float(self.tk.call('tk', 'scaling'))
        except Exception:
            sf = 1.0
        hit_r = max(12, int(14 * sf))
        for i, (cx, cy) in enumerate(pts):
            if abs(e.x - cx) < hit_r and abs(e.y - cy) < hit_r:
                self._surukle_idx = i
                self._durum_kaydet()
                break

    def _surukle_devam(self, e):
        if self._surukle_idx is None or self.mevcut is None:
            return
        ix, iy = self._cv2img(e.x, e.y)
        H, W = self.mevcut.shape[:2]
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

    # ── Scroll zoom ───────────────────────────────────────────────────────────

    def _scroll_zoom(self, event):
        if self.mevcut is None:
            return
        cw = self.kanvas.winfo_width()  or 800
        ch = self.kanvas.winfo_height() or 580
        H, W = self.mevcut.shape[:2]

        if getattr(event, 'delta', 0):
            factor = 1.15 if event.delta > 0 else 1 / 1.15
        else:
            factor = 1.15 if event.num == 4 else 1 / 1.15

        base_s = min(cw / W, ch / H)
        s  = base_s * self._zoom_faktor
        ox = (cw - int(W * s)) // 2 + self._pan_offset[0]
        oy = (ch - int(H * s)) // 2 + self._pan_offset[1]

        # Fare altındaki görüntü koordinatı sabit kalsın
        ix = (event.x - ox) / s
        iy = (event.y - oy) / s

        new_zoom = max(0.5, min(10.0, self._zoom_faktor * factor))
        new_s    = base_s * new_zoom
        new_cx   = (cw - int(W * new_s)) // 2
        new_cy   = (ch - int(H * new_s)) // 2

        self._pan_offset[0] = event.x - int(ix * new_s) - new_cx
        self._pan_offset[1] = event.y - int(iy * new_s) - new_cy
        self._zoom_faktor   = new_zoom
        self._kanvasi_yenile()

    # ── Orta tuş pan ─────────────────────────────────────────────────────────

    def _pan_baslat(self, event):
        self._pan_baslangic = (event.x, event.y,
                               self._pan_offset[0], self._pan_offset[1])

    def _pan_devam(self, event):
        if self._pan_baslangic is None or self.mevcut is None:
            return
        bx, by, bpx, bpy = self._pan_baslangic
        self._pan_offset[0] = bpx + (event.x - bx)
        self._pan_offset[1] = bpy + (event.y - by)
        self._kanvasi_yenile()

    def _pan_bitis(self, _):
        self._pan_baslangic = None

    # ── Klavye kısayolları ────────────────────────────────────────────────────

    def _kisayollar_kur(self):
        def _guard(fn):
            """Entry odağındaysa kısayolu yoksay."""
            def _inner(_event=None):
                focused = self.focus_get()
                if isinstance(focused, (tk.Entry, ctk.CTkEntry)):
                    return
                fn()
            return _inner

        self.bind("<Control-z>", _guard(self.geri_al))
        self.bind("<Control-Z>", _guard(self.geri_al))
        self.bind("<Control-o>", _guard(self.goruntu_ac))
        self.bind("<Control-O>", _guard(self.goruntu_ac))
        self.bind("<Control-s>", _guard(self.jpg_kaydet))   # Ctrl+S → JPG
        self.bind("<Control-S>", _guard(self.pdf_kaydet))   # Ctrl+Shift+S → PDF
        self.bind("<r>",         _guard(self.sifirla))
        self.bind("<R>",         _guard(self.sifirla))

    # ── Drag & drop ───────────────────────────────────────────────────────────

    def _drag_drop_ac(self, event):
        yol = event.data.strip().strip("{}")   # tkdnd süslü parantez ekleyebilir
        uzanti = os.path.splitext(yol)[1].lower()
        desteklenen = {".jpg", ".jpeg", ".png", ".bmp",
                       ".tiff", ".webp", ".heic", ".heif"}
        if uzanti not in desteklenen:
            self._durum(f"Desteklenmeyen dosya türü: {uzanti}")
            return
        self._goruntu_yukle(yol)

    # ══════════════════════════════════════════════════════════════════════════
    # CAMSCANNER İŞLEMLERİ
    # ══════════════════════════════════════════════════════════════════════════

    def oto_kose(self):
        if not self._goruntu_var():
            return
        orijinal_kopya = self.orijinal.copy()

        def _islem():
            return self.gi.kose.kose_tespit(orijinal_kopya)

        def _bitince(koseler):
            self.kose_img = koseler
            self._kanvasi_yenile()
            self._durum("Otomatik köşe tespiti tamamlandı — sürükleyerek ince ayar yapabilirsin.")

        self._isle_async(
            _islem, _bitince,
            "Canny → morfoloji → konveks zarf → Douglas-Peucker ile köşe tespiti yapılıyor..."
        )

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
            self.mevcut, H_mat, gen, yuk)

        self._durum_kaydet()
        self.mevcut   = duz
        self.kose_img = np.array(
            [[0, 0], [gen-1, 0], [gen-1, yuk-1], [0, yuk-1]], dtype=np.float64)
        self._kanvasi_yenile()
        self._durum(f"Perspektif düzeltme tamamlandı — {gen}×{yuk}")

    def tam_iyilestir(self):
        if not self._goruntu_var():
            return
        mevcut_kopya = self.mevcut.copy()

        pencere = self._v_sauvola_pencere.get()
        k_val   = self._v_sauvola_k.get()

        def _islem():
            gri      = self.gi.temel.griye_donustur(mevcut_kopya)
            gerilmis = self.gi.kontrast.histogram_ger(gri)
            sauvola  = self.gi.adaptif.sauvola(gerilmis, pencere=pencere, k=k_val)
            acilmis  = self.gi.morfoloji.ac(sauvola, self.gi.morfoloji.kare_cekirdek(3))
            return np.stack([acilmis] * 3, axis=2)

        def _bitince(sonuc):
            self._durum_kaydet()
            self.mevcut = sonuc
            self._kanvasi_yenile()
            self._durum("Tam iyileştirme tamamlandı.")

        self._isle_async(_islem, _bitince,
                         f"Hist. Germe → Sauvola (pencere={pencere}, k={k_val:.2f}) → Morfo Açma...")

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

    def _canli_dondur_cb(self, _val=None):
        """Döndürme slider'ı hareket edince canlı önizleme gösterir (self.mevcut değişmez)."""
        if self.mevcut is None:
            return
        if self._rot_base is None:
            self._rot_base = self.mevcut.copy()

        aci = self._v_donus.get()

        # Önizleme için en fazla 700px'e küçült (hız için)
        H, W = self._rot_base.shape[:2]
        if max(H, W) > 700:
            s = 700.0 / max(H, W)
            base = self.gi.geometri.olcekle(self._rot_base, s)
        else:
            base = self._rot_base

        preview = self.gi.geometri.dondur(base, aci)
        Hp, Wp = preview.shape[:2]

        # self.mevcut'a dokunmadan doğrudan kanvasa çiz
        cw = self.kanvas.winfo_width() or 800
        ch = self.kanvas.winfo_height() or 580
        s2 = min(cw / Wp, ch / Hp)
        yw = max(int(Wp * s2), 1)
        yh = max(int(Hp * s2), 1)
        pil = Image.fromarray(preview).resize((yw, yh), Image.LANCZOS)
        self._photo = ImageTk.PhotoImage(pil)
        self.kanvas.delete("all")
        ox, oy = (cw - yw) // 2, (ch - yh) // 2
        self.kanvas.create_image(ox, oy, anchor="nw", image=self._photo)
        self._durum(f"Önizleme: {aci:.1f}°  —  Uygulamak için 'Döndürmeyi Uygula' butonuna bas")

    def op_dondur(self):
        if not self._goruntu_var(): return
        aci = self._v_donus.get()
        base = self._rot_base if self._rot_base is not None else self.mevcut
        self._durum(f"Döndürülüyor {aci:.1f}° (tam çözünürlük) ...")
        self.update_idletasks()
        sonuc = self.gi.geometri.dondur(base, aci)
        H, W = sonuc.shape[:2]
        self.kose_img = np.array(
            [[0, 0], [W-1, 0], [W-1, H-1], [0, H-1]], dtype=np.float64)
        self._rot_base = None
        self._guncelle(sonuc, f"Döndürme: {aci:.1f}°")
        # Slider sıfırla — bir sonraki "Uygula" tekrar aynı açıyı eklemesin
        self._v_donus.set(0)
        self._rot_base = None   # _canli_dondur_cb'nin set ettiğini temizle
        self._kanvasi_yenile()  # köşe noktalarını geri çiz

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
        self._rgb_histogram_goster(self.mevcut)

    def op_kontrast(self):
        if not self._goruntu_var(): return
        f = self._v_kontrast.get()
        self._guncelle(self._g3(self.gi.kontrast.kontrast_azalt(self._gri(), f)),
                       f"Kontrast faktörü: {f:.2f}")

    def op_cikar(self):
        if not self._goruntu_var(): return
        if self.orijinal.shape == self.mevcut.shape:
            self._guncelle(
                self.gi.kontrast.aritmetik_cikar(self.orijinal, self.mevcut),
                "Aritmetik çıkarma: orijinal − mevcut (renkli)")
        else:
            g_ori = self.gi.temel.griye_donustur(self.orijinal)
            self._guncelle(self._g3(self.gi.kontrast.aritmetik_cikar(g_ori, self._gri())),
                           "Aritmetik çıkarma: orijinal − mevcut (gri, boyut farkı)")

    def op_carp(self):
        if not self._goruntu_var(): return
        if self.orijinal.shape == self.mevcut.shape:
            self._guncelle(
                self.gi.kontrast.aritmetik_carp(self.orijinal, self.mevcut),
                "Aritmetik çarpma: orijinal × mevcut (renkli)")
        else:
            g_ori = self.gi.temel.griye_donustur(self.orijinal)
            self._guncelle(self._g3(self.gi.kontrast.aritmetik_carp(g_ori, self._gri())),
                           "Aritmetik çarpma: orijinal × mevcut (gri, boyut farkı)")

    # ── Filtreler ─────────────────────────────────────────────────────────────

    def _tek_boyut(self, v):
        b = v.get()
        return b if b % 2 == 1 else b + 1

    def op_mean(self):
        if not self._goruntu_var(): return
        b = self._tek_boyut(self._v_filtre_boyut)
        gri = self._gri()
        self._isle_async(
            lambda: self._g3(self.gi.filtre.mean_filtre(gri, b)),
            lambda r: self._guncelle(r, f"Mean filtre {b}×{b} uygulandı."),
            f"Mean filtre {b}×{b} hesaplanıyor..."
        )

    def op_median(self):
        if not self._goruntu_var(): return
        b = self._tek_boyut(self._v_filtre_boyut)
        gri = self._gri()
        self._isle_async(
            lambda: self._g3(self.gi.filtre.median_filtre(gri, b)),
            lambda r: self._guncelle(r, f"Median filtre {b}×{b} uygulandı."),
            f"Median filtre {b}×{b} hesaplanıyor..."
        )

    def op_motion(self):
        if not self._goruntu_var(): return
        u = self._v_motion_uzun.get()
        a = self._v_motion_aci.get()
        gri = self._gri()
        self._isle_async(
            lambda: self._g3(self.gi.filtre.motion_filtre(gri, u, a)),
            lambda r: self._guncelle(r, "Motion filtre uygulandı."),
            f"Motion filtre (uzunluk={u}, açı={a:.0f}°) hesaplanıyor..."
        )

    # ── Gürültü ───────────────────────────────────────────────────────────────

    def op_gurultu_ekle(self):
        if not self._goruntu_var(): return
        y = self._v_gurultu.get()
        g = self.gi.gurultu.salt_pepper_ekle(self._gri(), y)
        self._guncelle(self._g3(g), f"Salt & Pepper eklendi (%{y*100:.0f})")

    def op_gurultu_temizle(self):
        if not self._goruntu_var(): return
        gri = self._gri()
        self._isle_async(
            lambda: self._g3(self.gi.gurultu.salt_pepper_temizle(gri, 3)),
            lambda r: self._guncelle(r, "Gürültü temizlendi."),
            "Median ile gürültü temizleniyor..."
        )

    # ── Kenar ─────────────────────────────────────────────────────────────────

    def op_canny(self):
        if not self._goruntu_var(): return
        lo, hi = self._v_canny_lo.get(), self._v_canny_hi.get()
        gri = self._gri()
        self._isle_async(
            lambda: self._g3(self.gi.kenar.canny_kenar(gri, lo, hi)),
            lambda r: self._guncelle(r, "Canny tamamlandı."),
            f"Canny kenar tespiti (lo={lo}, hi={hi}) hesaplanıyor..."
        )

    # ── Morfoloji ─────────────────────────────────────────────────────────────

    def op_morfo(self, islem: str):
        if not self._goruntu_var(): return
        b = self._tek_boyut(self._v_morfo_boyut)
        ck = self.gi.morfoloji.kare_cekirdek(b)
        islemler = {
            "genisle": self.gi.morfoloji.genisle,
            "asin":    self.gi.morfoloji.asin,
            "ac":      self.gi.morfoloji.ac,
            "kapat":   self.gi.morfoloji.kapat,
        }
        fn = islemler[islem]
        gri = self._gri()
        self._isle_async(
            lambda: self._g3(fn(gri, ck)),
            lambda r: self._guncelle(r, f"Morfoloji '{islem}' {b}×{b} tamamlandı."),
            f"Morfolojik '{islem}' {b}×{b} hesaplanıyor..."
        )

    # ══════════════════════════════════════════════════════════════════════════
    # KAYDETME
    # ══════════════════════════════════════════════════════════════════════════

    def jpg_kaydet(self):
        if not self._goruntu_var(): return
        cikti_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cikti")
        os.makedirs(cikti_dir, exist_ok=True)
        yol = filedialog.asksaveasfilename(
            initialdir=cikti_dir, defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png")],
            title="JPG / PNG olarak kaydet"
        )
        if not yol:
            return
        cv2.imwrite(yol, self.mevcut[:, :, ::-1])
        self._durum(f"Kaydedildi: {os.path.basename(yol)}")

    def pdf_kaydet(self):
        if not self._goruntu_var(): return
        cikti_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cikti")
        os.makedirs(cikti_dir, exist_ok=True)
        yol = filedialog.asksaveasfilename(
            initialdir=cikti_dir, defaultextension=".pdf",
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
            cw = canvas.winfo_width()  or 500
            ch = canvas.winfo_height() or 460
            H, W = arr.shape[:2]
            s = min(cw / W, ch / H)
            yw, yh = max(int(W*s), 1), max(int(H*s), 1)
            pil   = Image.fromarray(arr).resize((yw, yh), Image.LANCZOS)
            photo = ImageTk.PhotoImage(pil)
            canvas._photo = photo
            canvas.create_image((cw-yw)//2, (ch-yh)//2,
                                 anchor="nw", image=photo)

        win.update()   # pencereyi render et, canvas boyutları hazır olsun
        _yukle(cv_ori, self.orijinal)
        _yukle(cv_isi, self.mevcut)

    # ══════════════════════════════════════════════════════════════════════════
    # HİSTOGRAM GRAFİĞİ
    # ══════════════════════════════════════════════════════════════════════════

    def _rgb_histogram_goster(self, img: np.ndarray):
        try:
            import matplotlib
            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            fig, ax = plt.subplots(figsize=(8, 4), facecolor=C["bg"])
            ax.set_facecolor(C["panel"])
            kanal_bilgi = [
                (0, C["red"],    "R"),
                (1, C["green"],  "G"),
                (2, C["accent"], "B"),
            ]
            for c, renk, isim in kanal_bilgi:
                hist = self.gi.kontrast.histogram_hesapla(img[:, :, c])
                ax.plot(range(256), hist, color=renk,
                        linewidth=1.3, label=isim, alpha=0.85)
            ax.legend(facecolor=C["panel"], edgecolor=C["border"],
                      labelcolor=C["text"])
            ax.set_xlabel("Piksel Değeri", color=C["text"])
            ax.set_ylabel("Piksel Sayısı",  color=C["text"])
            ax.set_title("RGB Histogram", color=C["text"],
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
        self._rot_base = None
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

    # ── Asenkron iş yürütücü ──────────────────────────────────────────────────

    def _isle_async(self, islem_fn, bitince_fn, mesaj: str = "İşleniyor..."):
        """
        islem_fn'yi arka plan thread'inde çalıştırır, UI'ı dondurmaz.
        Sonuç ana thread'e bitince_fn(sonuc) ile iletilir.
        """
        if self._isleniyor:
            self._durum("⏳ Lütfen mevcut işlemin tamamlanmasını bekleyin.")
            return
        self._isleniyor = True
        self._durum(f"⏳ {mesaj}")
        self.update_idletasks()

        def _calis():
            try:
                sonuc = islem_fn()
                self.after(0, lambda: bitince_fn(sonuc))
            except Exception as e:
                self.after(0, lambda: (
                    messagebox.showerror("İşlem Hatası", str(e)),
                    self._durum("Hata oluştu.")
                ))
            finally:
                self.after(0, lambda: setattr(self, '_isleniyor', False))

        threading.Thread(target=_calis, daemon=True).start()


# ── Giriş noktası ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = BelgeTaramaApp()
    app.mainloop()
