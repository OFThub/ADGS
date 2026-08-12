"""KIRMIZI_ISIK ihlali.

Uc kosul BIRLIKTE aranir:
    1. Isik o karede KIRMIZI
    2. Aracin zemin noktasi dur cizgisi PARCASINI o karede kesiyor
    3. Gecisten sonra arac ilerlemeye devam ediyor

3. kosul olmadan, kirmizida cizgiyi bir tekerlek boyu asip duran arac ihlal
sayilirdi. Ihlal, kirmizida kavsaga GIRMEKtir; cizgiyi hafif asip durmak degil.

Isik durumu ayri bir siniflandirici degil, config'teki isik ROI'sinin HSV
dagilimindan okunur: yanan lamba doygun ve parlaktir, bu yuzden S/V esiginin
uzerindeki piksellerin renk bandi durumu verir. ROI yoksa modul CALISMAZ -
isigi tahmin etmek yanlis ihlal kaydi uretir.

ponytail: HSV esikleme. Gece parlamasi, sari->kirmizi gecisi ve arka arac stop
lambasi yansimasi hatali okunabilir; gerekirse ROI'ye kucuk bir isik
siniflandirici takilir, arayuz (kare -> durum) ayni kalir.
"""

from __future__ import annotations

import math
from pathlib import Path

from adgs.schema import Event, Track
from adgs.violations.base import Baglam, kare_ciftleri, kesisiyor_mu, olay, zemin

KOD = "KIRMIZI_ISIK"

# OpenCV HSV: H 0-179. Kirmizi 0/180 civarinda sarmal yaptigi icin iki bant.
_HSV_BANT = {
    "KIRMIZI": [((0, 120, 110), (10, 255, 255)), ((170, 120, 110), (179, 255, 255))],
    "SARI": [((18, 120, 130), (33, 255, 255))],
    "YESIL": [((40, 90, 90), (90, 255, 255))],
}


def _renk(crop, min_piksel: int) -> str:
    import cv2
    import numpy as np

    if crop is None or crop.size == 0:
        return "BELIRSIZ"
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    sayim = {
        ad: sum(
            int(cv2.countNonZero(cv2.inRange(hsv, np.array(lo, np.uint8),
                                             np.array(hi, np.uint8))))
            for lo, hi in bantlar
        )
        for ad, bantlar in _HSV_BANT.items()
    }
    ad, n = max(sayim.items(), key=lambda kv: kv[1])
    # Yeterli yanan piksel yoksa "karar veremedim" denir; KIRMIZI VARSAYILMAZ.
    return ad if n >= min_piksel else "BELIRSIZ"


def isik_durumu(video: str | Path, roi: list, min_piksel: int = 20) -> dict[int, str]:
    """Her kare icin isik durumu: KIRMIZI | SARI | YESIL | BELIRSIZ."""
    import cv2

    x1, y1, x2, y2 = (int(v) for v in roi)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"video acilamadi: {video}")
    durum: dict[int, str] = {}
    try:
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            durum[idx] = _renk(img[y1:y2, x1:x2], min_piksel)
            idx += 1
    finally:
        cap.release()
    return durum


def _devam_etti_mi(iz: Track, f: int, pencere: int, esik: float) -> bool:
    """Kosul 3: gecisten sonra ilerlemeye devam etti mi."""
    sonraki = [k for k in iz.frames if f < k <= f + pencere]
    if not sonraki:
        return False
    p0 = zemin(iz.frames[f].bbox)
    return max(math.dist(p0, zemin(iz.frames[k].bbox)) for k in sonraki) >= esik


def tespit_et(izler: list[Track], ctx: Baglam,
              durum: dict[int, str] | None = None) -> list[Event]:
    """Kirmizi isik ihlallerini bulur.

    `durum` disaridan verilebilir (test ve yeniden kullanim icin); verilmezse
    videodan okunur.
    """
    cizgi = ctx.kamera.get("dur_cizgisi")
    roi = ctx.kamera.get("isik_roi")
    if not cizgi:
        return ctx.reddet(KOD, "kamera config'inde dur_cizgisi tanimli degil")
    if durum is None and not roi:
        return ctx.reddet(KOD, "kamera config'inde isik_roi tanimli degil")

    pencere = int(ctx.p(KOD, "devam_pencere_kare", 8))
    esik = float(ctx.p(KOD, "devam_piksel", 12.0))
    if durum is None:
        try:
            durum = isik_durumu(ctx.video, roi, int(ctx.p(KOD, "isik_min_piksel", 20)))
        except FileNotFoundError as e:
            return ctx.reddet(KOD, str(e))

    if "KIRMIZI" not in set(durum.values()):
        # Ihlal cikmamasi "ihlal yok" degil "isik hic kirmizi okunmadi" olabilir.
        ctx.uyarilar.append(
            f"{KOD}: videoda hic KIRMIZI kare okunmadi - isik_roi dogru mu? "
            f"(okunan durumlar: {sorted(set(durum.values()))})"
        )

    c1 = (float(cizgi[0][0]), float(cizgi[0][1]))
    c2 = (float(cizgi[1][0]), float(cizgi[1][1]))
    olaylar: list[Event] = []
    for iz in izler:
        for f0, f1 in kare_ciftleri(iz):
            if durum.get(f1) != "KIRMIZI":
                continue
            if not kesisiyor_mu(zemin(iz.frames[f0].bbox), zemin(iz.frames[f1].bbox),
                                c1, c2):
                continue
            if not _devam_etti_mi(iz, f1, pencere, esik):
                continue
            olaylar.append(olay(
                KOD, iz, f1, ctx,
                conf=iz.frames[f1].conf,
                notlar=[
                    f"Iz #{iz.track_id} ({iz.cls}) kare {f1}'de dur cizgisini "
                    f"isik KIRMIZI iken gecti ve ilerlemeye devam etti.",
                    "Isik durumu ROI renk analiziyle okunmustur; sinyalizasyon "
                    "sisteminin kendi kaydi degildir.",
                ],
                f_bas=max(0, f1 - pencere), f_son=f1 + pencere,
            ))
            break  # bir arac icin tek gecis yeterli
    return olaylar
