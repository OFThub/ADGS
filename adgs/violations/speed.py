"""HIZ_IHLALI - kalibrasyona KOSULLU modul.

Bu modul, kalibrasyon dogrulamayi gecmediyse CALISMAYI REDDEDER. Plan Faz 4
kabul kriteri budur ve test edilmis davranistir (tests/test_violations.py).

Neden bu kadar kati: hiz, homografinin dogruluguna dogrudan bagli bir turevdir.
%20 hatali bir homografi 50 km/s'lik araci 60 km/s gosterir - ve bu sayi ceza
doguran bir cikti olarak dolasima girer. Ultralytics'in kendi hiz tahmini tek
bir metre/piksel skaleri kullanir ve perspektifi hesaba katmaz; bu proje tam
homografi kullanir ve dogrulanmadan hicbir sey uretmez.

Hiz yer duzleminde (Track.world_xy) olculur. world_xy'yi M2 yalnizca gecerli
kalibrasyonda doldurur, yani ikinci bir guvenlik kilidi zaten oradadir.

Tek karelik zirve YETMEZ: takip kutusunun titremesi anlik 100 km/s uretebilir.
Esik asiminin ardisik N karede surmesi aranir.
"""

from __future__ import annotations

import math

from adgs.schema import Event, Track
from adgs.violations.base import Baglam, olay

KOD = "HIZ_IHLALI"


def hiz_kmh(iz: Track, f0: int, f1: int, fps: float) -> float | None:
    """Iki kare arasi yer duzlemi hizi (km/s). Olculemezse None - 0 DEGIL."""
    w = iz.world_xy
    if not w or f0 not in w or f1 not in w or f1 <= f0 or fps <= 0:
        return None
    dt = (f1 - f0) / fps
    return math.dist(w[f0], w[f1]) / dt * 3.6 if dt > 0 else None


def tespit_et(izler: list[Track], ctx: Baglam) -> list[Event]:
    if not ctx.kalibrasyon_gecerli_mi(KOD):
        return []  # sebep ctx.uyarilar'a yazildi

    limit = ctx.kamera.get("hiz_limiti_kmh")
    if not limit:
        return ctx.reddet(KOD, "kamera config'inde hiz_limiti_kmh tanimli degil")
    limit = float(limit)

    tolerans = float(ctx.p(KOD, "tolerans_yuzde", 10.0))
    pencere = int(ctx.p(KOD, "pencere_kare", 10))
    gerekli = int(ctx.p(KOD, "min_ardisik_kare", 5))
    # Tolerans musamaha degil, olcum belirsizligi payidir: kalibrasyon hatasi
    # %10'a kadar kabul edildigi icin esik de o kadar yukari tasinir.
    esik = limit * (1.0 + tolerans / 100.0)

    olaylar: list[Event] = []
    for iz in izler:
        if not iz.world_xy:
            continue
        kareler = sorted(iz.world_xy)
        ardisik, bas, en_yuksek = 0, None, 0.0
        for i, f in enumerate(kareler):
            f_ileri = kareler[min(i + pencere, len(kareler) - 1)]
            v = hiz_kmh(iz, f, f_ileri, ctx.fps)
            if v is None or v <= esik:
                ardisik, bas, en_yuksek = 0, None, 0.0
                continue
            ardisik += 1
            bas = f if bas is None else bas
            en_yuksek = max(en_yuksek, v)
            if ardisik >= gerekli:
                olaylar.append(olay(
                    KOD, iz, f, ctx,
                    conf=min(1.0, (en_yuksek - esik) / max(1e-6, esik)),
                    notlar=[
                        f"Iz #{iz.track_id} ({iz.cls}) {ardisik} ardisik karede "
                        f"tahmini {en_yuksek:.0f} km/s ile ilerledi "
                        f"(limit {limit:.0f}, toleransli esik {esik:.0f}).",
                        f"Hiz, dogrulanmis homografi ile hesaplanmistir (kalibrasyon "
                        f"hatasi %{float(getattr(ctx.kalib, 'hata_yuzde', 0) or 0):.1f}). "
                        "Tahmini bir degerdir; radar/EDS olcumu yerine gecmez.",
                    ],
                    f_bas=bas, f_son=f,
                ))
                break
    return olaylar
