"""TAKIP_MESAFESI ihlali - kalibrasyona KOSULLU modul.

Takip suresi = onundeki araca olan yer duzlemi mesafesi / kendi hizi.
Bu sure guvenli esigin (varsayilan 2 sn) altinda ardisik N karede kalirsa ihlal.

Mesafe metre cinsinden olculmek zorundadir, bu yuzden kalibrasyon sarttir ve
gecersizse modul CALISMAYI REDDEDER (speed.py ile ayni disiplin).

Yanlis pozitifin ana kaynagi duran trafik: kirmizi isikta tampon tampona
bekleyen araclarin takip suresi matematiksel olarak sifira gider. Bu yuzden
min_hiz_kmh altindaki araclar hic degerlendirilmez - sikisiklikta beklemek
takip mesafesi ihlali degildir.

Ikinci filtre yon uyumu: yan seritteki veya karsi yondeki arac "onumdeki arac"
degildir. Yalnizca ayni yone giden (aci < esik), hareket yonunde ONDE ve yanal
sapmasi bir serit genisliginden az olan araclar lider sayilir.
"""

from __future__ import annotations

import math

from adgs.schema import Event, Track
from adgs.violations.base import Baglam, olay
from adgs.violations.speed import hiz_kmh

KOD = "TAKIP_MESAFESI"


def _yon(iz: Track, f: int, pencere: int) -> tuple[float, float] | None:
    """Yer duzleminde hareket yonu (birim vektor). Duran arac icin None."""
    w = iz.world_xy or {}
    ileri = [k for k in w if f < k <= f + pencere]
    if f not in w or not ileri:
        return None
    p0, p1 = w[f], w[max(ileri)]
    d = (p1[0] - p0[0], p1[1] - p0[1])
    n = math.hypot(*d)
    return (d[0] / n, d[1] / n) if n > 1e-6 else None


def tespit_et(izler: list[Track], ctx: Baglam) -> list[Event]:
    if not ctx.kalibrasyon_gecerli_mi(KOD):
        return []

    guvenli = float(ctx.p(KOD, "guvenli_sn", 2.0))
    pencere = int(ctx.p(KOD, "pencere_kare", 10))
    gerekli = int(ctx.p(KOD, "min_ardisik_kare", 5))
    min_hiz = float(ctx.p(KOD, "min_hiz_kmh", 20.0))
    maks_mesafe = float(ctx.p(KOD, "maks_mesafe_m", 50.0))
    yon_esik = float(ctx.p(KOD, "yon_uyum_derece", 30.0))
    yanal = float(ctx.p(KOD, "maks_yanal_m", 2.5))

    araclar = [t for t in izler if t.world_xy]
    olaylar: list[Event] = []

    for takipci in araclar:
        ardisik, bas, en_kisa, lider_id = 0, None, None, None
        for f in sorted(takipci.world_xy):
            v = hiz_kmh(takipci, f, f + pencere, ctx.fps)
            yon = _yon(takipci, f, pencere)
            if v is None or v < min_hiz or yon is None:
                ardisik, bas = 0, None
                continue

            p = takipci.world_xy[f]
            en_yakin = None
            for lider in araclar:
                if lider is takipci or f not in (lider.world_xy or {}):
                    continue
                q = lider.world_xy[f]
                d = (q[0] - p[0], q[1] - p[1])
                ileri_m = d[0] * yon[0] + d[1] * yon[1]
                if ileri_m <= 0 or ileri_m > maks_mesafe:
                    continue  # arkada veya cok uzakta
                if abs(d[0] * -yon[1] + d[1] * yon[0]) > yanal:
                    continue  # yan seritte
                l_yon = _yon(lider, f, pencere)
                if l_yon is not None:
                    kos = max(-1.0, min(1.0, l_yon[0] * yon[0] + l_yon[1] * yon[1]))
                    if math.degrees(math.acos(kos)) > yon_esik:
                        continue  # karsi yon veya farkli manevra
                if en_yakin is None or ileri_m < en_yakin[0]:
                    en_yakin = (ileri_m, lider.track_id)

            if en_yakin is None:
                ardisik, bas = 0, None
                continue
            sure = en_yakin[0] / (v / 3.6)  # m / (m/s)
            if sure >= guvenli:
                ardisik, bas = 0, None
                continue

            ardisik += 1
            bas = f if bas is None else bas
            en_kisa = sure if en_kisa is None else min(en_kisa, sure)
            lider_id = en_yakin[1]
            if ardisik >= gerekli:
                olaylar.append(olay(
                    KOD, takipci, f, ctx,
                    conf=min(1.0, (guvenli - en_kisa) / guvenli),
                    notlar=[
                        f"Iz #{takipci.track_id} ({takipci.cls}) onundeki iz "
                        f"#{lider_id} ile arasindaki takip suresini {ardisik} ardisik "
                        f"karede {en_kisa:.2f} sn'ye dusurdu (guvenli esik "
                        f"{guvenli:.1f} sn, hiz {v:.0f} km/s).",
                        "Mesafe dogrulanmis homografi ile olculmustur; tahmini bir "
                        "degerdir.",
                    ],
                    f_bas=bas, f_son=f,
                ))
                break
    return olaylar
