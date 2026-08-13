"""Fen Isleri icin PDF is emri (Faz 7).

Belediyenin OPERASYONEL ciktisi budur: yol/altyapi hasari bir bakim is emrine
donusur. Kaza ve ihlal ciktilari ise planlama/bilgilendirme niteligindedir ve
idari islem baslatmaz - bu ayrim sayfanin altindaki zorunlu uyariyla korunur.

Neden matplotlib: PDF icin yeni bagimlilik EKLENMEDI. reportlab/fpdf2'nin
gomulu fontlari Latin-1'dir ve Turkce s/g/I karakterlerini basamaz; resmi bir
is emrinde bozuk metin kabul edilemez. matplotlib zaten ultralytics ile
kuruludur ve DejaVu Sans fontunu paket icinde tasir - Turkce tam destekli.

ponytail: metin yerlesimi elle koordinatli (sabit A4 sablonu). Cok sayfali veya
degisken uzunlukta rapor gerekirse gercek bir PDF kutuphanesine gecilmeli; tek
sayfalik is emri icin bu yeterli ve bagimlilik eklemiyor.
"""

from __future__ import annotations

from pathlib import Path

# A4 (inch). Yerlesim sayfa oranina gore normalize koordinatlarda.
_A4 = (8.27, 11.69)
_SOL, _SAG = 0.08, 0.92

_ALTYAPI_NOTU = (
    "Bu iş emri görüntü analizine dayalı bir ÖN DEĞERLENDİRMEDİR. Hasarın "
    "gerçek boyutu, müdahale yöntemi ve önceliği saha kontrolü ile "
    "doğrulanmalıdır."
)
_KUSUR_NOTU = (
    "Bu çıktı bağlayıcı bir tespit değildir ve idari işleme dayanak "
    "oluşturmaz. Kusur oranı yetkili merciler tarafından belirlenir."
)

_TIP_BASLIK = {
    "ALTYAPI": "YOL / ALTYAPI BAKIM İŞ EMRİ",
    "KAZA": "KAZA ÖN DEĞERLENDİRME FORMU",
    "IHLAL": "İHLAL ÖN DEĞERLENDİRME FORMU",
}


def _siddet_ve_oncelik(olay: dict) -> tuple[str, str]:
    """M5'in urettigi siddet/oncelik notlarindan okur - YENIDEN HESAPLAMAZ."""
    metin = " ".join(olay.get("notlar") or [])
    siddet = next((s for s in ("YUKSEK", "ORTA", "DUSUK") if f"siddet {s}" in metin), "-")
    oncelik = "-"
    if "oncelik " in metin:
        parca = metin.split("oncelik ", 1)[1].strip()
        oncelik = parca[0] if parca[:1].isdigit() else "-"
    return siddet, oncelik


def _kanit_karesi(olay: dict, video: str | Path | None):
    """Olayin kanit karesini okur. Okunamazsa None - uydurma gorsel konmaz."""
    kaynak = video or olay.get("klip") or olay.get("video")
    if not kaynak or not Path(str(kaynak)).exists():
        return None
    import cv2

    cap = cv2.VideoCapture(str(kaynak))
    if not cap.isOpened():
        return None
    try:
        # Klipte kare numarasi klibin basina goredir; tam videoda frame_start.
        klip_mi = str(kaynak) == str(olay.get("klip"))
        hedef = 0 if klip_mi else int(olay.get("frame_start") or 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, hedef))
        ok, img = cap.read()
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB) if ok else None
    finally:
        cap.release()


def _cizgi(fig, y: float):
    from matplotlib.lines import Line2D

    return Line2D([_SOL, _SAG], [y, y], transform=fig.transFigure,
                  color="#999999", linewidth=0.8)


def _sar(metin: str, genislik: int) -> list[str]:
    import textwrap

    return textwrap.wrap(metin, genislik) or [""]


def is_emri_pdf(olay: dict, cikti: str | Path, video: str | Path | None = None,
                kurum: str = "Arnavutköy Belediyesi") -> Path:
    """Tek sayfalik A4 is emri uretir ve yolunu dondurur."""
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    cikti = Path(cikti)
    cikti.parent.mkdir(parents=True, exist_ok=True)

    tip = olay.get("tip", "-")
    siddet, oncelik = _siddet_ve_oncelik(olay)
    gps = olay.get("gps")
    fig = Figure(figsize=_A4, dpi=150)

    y = 0.955
    fig.text(_SOL, y, _TIP_BASLIK.get(tip, "OLAY FORMU"), fontsize=17, weight="bold")
    y -= 0.022
    fig.text(_SOL, y, f"{kurum} — ADGS Trafik Video Analiz Sistemi", fontsize=9.5,
             color="#444444")
    y -= 0.012
    fig.add_artist(_cizgi(fig, y))

    satirlar = [
        ("İş emri no", olay.get("event_id", "-")),
        ("Olay tipi", f"{tip} / {olay.get('alt_tip', '-')}"),
        ("Şiddet", siddet),
        ("Öncelik", f"{oncelik}  (1 = acil, 3 = planlı)" if oncelik != "-" else "-"),
        ("Konum (GPS)", f"{gps[0]:.6f}, {gps[1]:.6f}" if gps else
                        "GPS YOK — konum saha ekibince belirlenmeli"),
        ("Kamera / kaynak", olay.get("camera_id") or olay.get("video") or "-"),
        ("Çekim tarihi", olay.get("cekim_tarihi") or "-"),
        ("Kare aralığı", f"{olay.get('frame_start', '-')} – {olay.get('frame_end', '-')}"),
        ("Güven skoru", f"{olay.get('conf', '-')}"),
    ]
    y -= 0.030
    for etiket, deger in satirlar:
        fig.text(_SOL, y, etiket, fontsize=10, weight="bold")
        fig.text(_SOL + 0.24, y, str(deger), fontsize=10)
        y -= 0.0225

    y -= 0.012
    fig.text(_SOL, y, "Kanıt görüntüsü", fontsize=10, weight="bold")
    img = _kanit_karesi(olay, video)
    gorsel_y = y - 0.31
    ax = fig.add_axes([_SOL, gorsel_y, _SAG - _SOL, 0.295])
    ax.set_xticks([])
    ax.set_yticks([])
    if img is not None:
        ax.imshow(img)
    else:
        ax.text(0.5, 0.5, "Görüntü bulunamadı\n(klip veya kaynak video erişilemedi)",
                ha="center", va="center", fontsize=10, color="#888888")
    y = gorsel_y - 0.025

    fig.text(_SOL, y, "Notlar ve sınırlamalar", fontsize=10, weight="bold")
    y -= 0.018
    for not_metni in (olay.get("notlar") or [])[:6]:
        for parca in _sar(not_metni, 108):
            fig.text(_SOL, y, parca, fontsize=8.2, color="#222222")
            y -= 0.0135
        y -= 0.004

    # Zorunlu uyari - kaldirilamaz.
    uyari = _ALTYAPI_NOTU if tip == "ALTYAPI" else _KUSUR_NOTU
    y = min(y, 0.205)
    fig.add_artist(_cizgi(fig, y + 0.022))
    for parca in _sar(uyari, 100):
        fig.text(_SOL, y, parca, fontsize=8.6, weight="bold", color="#8a0000")
        y -= 0.015

    fig.text(_SOL, 0.075, "Düzenleyen", fontsize=9, weight="bold")
    fig.text(_SOL, 0.045, "____________________", fontsize=9)
    fig.text(0.55, 0.075, "Onaylayan (Fen İşleri)", fontsize=9, weight="bold")
    fig.text(0.55, 0.045, "____________________", fontsize=9)
    fig.text(_SOL, 0.02, "ADGS — karar destek aracı. Bağlayıcı bir tespit üretmez.",
             fontsize=7.5, color="#666666")

    # Bicim uzantidan cikarilir: .png vermek gorsel dogrulamayi mumkun kilar
    # (PDF'ten metin cikarmak font altkumesi yuzunden guvenilir degil).
    fig.savefig(cikti)
    return cikti
