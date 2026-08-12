"""HATALI_PARK ihlali.

Iki kosul: (1) aracin zemin noktasi yasak park poligonunun icinde, (2) orada
kesintisiz N saniye hareketsiz.

Sure esigi neden yuksek (varsayilan 60 sn): kirmizi isikta bekleyen, yolcu
indiren veya trafikte duran her arac kisa sureli hareketsizdir. Duraklama ile
park arasindaki fark suredir; esigi dusurmek yanlis pozitif fabrikasi olur.

Hareketsizlik testi M3'ten (accident.hareketsiz_mi) yeniden kullanilir -
"merkez N kare boyunca esik pikselin icinde kaldi mi" semantigi ayni; ayni
mantigin ikinci kopyasi yazilmaz.
"""

from __future__ import annotations

from adgs.accident import KazaParam, hareketsiz_mi
from adgs.schema import Event, Track
from adgs.violations.base import Baglam, nokta_poligonda, olay, zemin

KOD = "HATALI_PARK"


def _alan(p, alanlar: list) -> dict | None:
    for a in alanlar:
        if nokta_poligonda(p, a.get("poligon") or []):
            return a
    return None


def tespit_et(izler: list[Track], ctx: Baglam) -> list[Event]:
    alanlar = [a for a in (ctx.kamera.get("yasak_park") or []) if a.get("poligon")]
    if not alanlar:
        return ctx.reddet(KOD, "kamera config'inde yasak_park poligonu tanimli degil")

    sure = float(ctx.p(KOD, "sure_sn", 60.0))
    kayma = float(ctx.p(KOD, "kayma_piksel", 8.0))
    p = KazaParam(hareketsizlik_sn=sure, hareketsizlik_piksel=kayma)
    gerekli = max(2, int(sure * ctx.fps))

    olaylar: list[Event] = []
    for iz in izler:
        kareler = sorted(iz.frames)
        for f in kareler:
            a = _alan(zemin(iz.frames[f].bbox), alanlar)
            if a is None:
                continue
            if not hareketsiz_mi(iz, f, ctx.fps, p):
                continue
            son = [k for k in kareler if k >= f][:gerekli][-1]
            olaylar.append(olay(
                KOD, iz, f, ctx,
                conf=iz.frames[f].conf,
                notlar=[
                    f"Iz #{iz.track_id} ({iz.cls}) '{a.get('ad', '?')}' yasak park "
                    f"alaninda kare {f}'den itibaren {sure:.0f} sn hareketsiz kaldi.",
                    "Duraklama ile park ayrimi yalnizca sure esigine dayanir; arac "
                    "arizasi, yolcu indirme veya trafik sikisikligi ayirt edilemez.",
                ],
                f_bas=f, f_son=son,
            ))
            break  # ayni arac icin tek kayit
    return olaylar
