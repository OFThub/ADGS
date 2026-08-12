"""TERS_YON ihlali.

Aracin hareket vektoru, icinde bulundugu seridin tanimli yon vektoruyle esik
aciyi (varsayilan 120 derece) asarsa ters yondur.

Tek karelik sapma YETMEZ. Park cikisi, manevra, geri gitme ve takip gurultusu
(ID sicramasi, kutu titremesi) tek karede kolayca 180 derece gorunur. Bu yuzden
sapmanin ardisik N karede surmesi aranir - precision recall'dan once gelir.

Kalibrasyon ZORUNLU DEGILDIR: perspektif nedeniyle piksel uzayinda uzak
araclarin hareket vektoru kisalir ama ACISI korunur; bu modul yalnizca aciya
bakar. Serit yon vektoru de goruntu uzayinda tanimlanir.
"""

from __future__ import annotations

import math

from adgs.schema import Event, Track
from adgs.violations.base import Baglam, nokta_poligonda, olay, zemin

KOD = "TERS_YON"


def _aci_derece(v1: tuple[float, float], v2: tuple[float, float]) -> float | None:
    """Iki vektor arasi aci (derece). Vektorlerden biri sifira yakinsa None."""
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    kos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, kos))))


def _serit(p, seritler: list) -> dict | None:
    for s in seritler:
        if nokta_poligonda(p, s.get("poligon") or []):
            return s
    return None


def tespit_et(izler: list[Track], ctx: Baglam) -> list[Event]:
    seritler = ctx.kamera.get("seritler") or []
    yonlu = [s for s in seritler if s.get("yon") and s.get("poligon")]
    if not yonlu:
        return ctx.reddet(
            KOD, "kamera config'inde yon vektoru tanimli serit yok (seritler[].yon)"
        )

    esik = float(ctx.p(KOD, "aci_esik_derece", 120.0))
    pencere = int(ctx.p(KOD, "pencere_kare", 5))
    gerekli = int(ctx.p(KOD, "min_ardisik_kare", 8))
    min_kayma = float(ctx.p(KOD, "min_kayma_piksel", 3.0))

    olaylar: list[Event] = []
    for iz in izler:
        kareler = sorted(iz.frames)
        ardisik, bas = 0, None
        for i, f in enumerate(kareler):
            f_ileri = kareler[min(i + pencere, len(kareler) - 1)]
            p0, p1 = zemin(iz.frames[f].bbox), zemin(iz.frames[f_ileri].bbox)
            hareket = (p1[0] - p0[0], p1[1] - p0[1])
            s = _serit(p0, yonlu)
            aci = None
            if s is not None and math.hypot(*hareket) >= min_kayma:
                aci = _aci_derece(hareket, (float(s["yon"][0]), float(s["yon"][1])))
            # Duran arac veya serit disi konum: sayac sifirlanir. Ters yon,
            # SUREN bir davranistir; kesintili sapma manevradir.
            if aci is None or aci <= esik:
                ardisik, bas = 0, None
                continue
            ardisik += 1
            bas = f if bas is None else bas
            if ardisik >= gerekli:
                olaylar.append(olay(
                    KOD, iz, f, ctx,
                    # Guven: aci esigi ne kadar astiysa o kadar net (esik..180 -> 0..1)
                    conf=min(1.0, (aci - esik) / max(1e-6, 180.0 - esik)),
                    notlar=[
                        f"Iz #{iz.track_id} ({iz.cls}) '{s.get('ad', '?')}' seridinde "
                        f"{ardisik} ardisik karede serit yonuyle {aci:.0f} derece aci "
                        f"yaparak ilerledi (esik {esik:.0f}).",
                        "Serit yon tanimi kamera config'inden gelir; yanlis tanimlanmis "
                        "bir serit yanlis tespit uretir.",
                    ],
                    f_bas=bas, f_son=f,
                ))
                break  # bir arac icin tek kayit
    return olaylar
