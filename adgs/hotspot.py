"""Faz 8 - Kara nokta analizi.

Belediyenin bu sistemden cikarabilecegi en degerli PLANLAMA ciktisi budur:
hangi noktada, ne siklikta ve ne agirlikta olay oluyor. Kusur veya ceza degil,
YATIRIM KARARI girdisi - hangi kavsaga sinyalizasyon, hangi yola kasis.

IKI AYRI GRUPLAMA, cunku iki ayri kaynak profili var:
    GPS varsa   -> cografi kumeleme (araca monteli kamera; yol hasari)
    GPS yoksa   -> kamera bazli gruplama (sabit CCTV; kaza/ihlal)

Ikisi birbirine KARISTIRILMAZ. Sabit kameranin olaylarini "0,0 koordinatinda
kumelenmis" gibi gostermek, haritada var olmayan bir kara nokta uretirdi.

Agirliklar bir risk MODELI degil, siralama olcegidir: bir kaza bir ihlalden
agir, ihlal bir yol hasarindan agirdir. Gercek kara nokta analizi yaralanma ve
maddi hasar verisini de kullanir - bu sistemde o veri yok.
"""

from __future__ import annotations

import math
from collections import defaultdict

# Siralama agirliklari. Mutlak bir risk skoru DEGIL, goreli onceliktir.
AGIRLIK = {"KAZA": 5.0, "IHLAL": 2.0, "ALTYAPI": 1.0}
# Yol hasarinda siddet carpani (M5'in urettigi notlardan okunur).
SIDDET_CARPANI = {"YUKSEK": 2.0, "ORTA": 1.4, "DUSUK": 1.0}

VARSAYILAN_YARICAP_M = 50.0
_DUNYA_YARICAPI_M = 6_371_000.0


def mesafe_m(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Iki (enlem, boylam) arasi haversine mesafesi (metre).

    Duz oklid kullanmak Turkiye enleminde boylam ekseninde ~%25 hata verir -
    50 m'lik bir kumeleme yaricapinda bu, komsu kavsaklari birlestirirdi.
    """
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * _DUNYA_YARICAPI_M * math.asin(min(1.0, math.sqrt(h)))


def _siddet(olay: dict) -> str | None:
    metin = " ".join(olay.get("notlar") or [])
    return next((s for s in ("YUKSEK", "ORTA", "DUSUK") if f"siddet {s}" in metin), None)


def agirlik(olay: dict) -> float:
    """Tek olayin siralama agirligi."""
    temel = AGIRLIK.get(olay.get("tip", ""), 1.0)
    s = _siddet(olay)
    return temel * SIDDET_CARPANI.get(s, 1.0) if s else temel


def _gps(olay: dict) -> tuple[float, float] | None:
    g = olay.get("gps")
    if not g or len(g) < 2 or g[0] is None or g[1] is None:
        return None
    return (float(g[0]), float(g[1]))


def kumele(olaylar: list[dict],
           yaricap_m: float = VARSAYILAN_YARICAP_M) -> list[dict]:
    """GPS'li olaylari cografi olarak kumeler, agirliga gore siralar.

    ponytail: acgozlu (greedy) kumeleme - ilk olay merkezi baslatir, yaricap
    icindekiler ona katilir. DBSCAN daha iyi sinir cizerdi ama bir bagimlilik
    ve ayarlanacak ikinci parametre getirirdi; olay sayisi yuzler mertebesinde
    kaldigi surece bu yeterli. Merkez, katilan noktalarin ortalamasidir.
    """
    kumeler: list[dict] = []

    for olay in olaylar:
        nokta = _gps(olay)
        if nokta is None:
            continue
        hedef = next((k for k in kumeler
                      if mesafe_m(k["merkez"], nokta) <= yaricap_m), None)
        if hedef is None:
            hedef = {"merkez": nokta, "noktalar": [], "olaylar": [],
                     "agirlik": 0.0, "tipler": defaultdict(int)}
            kumeler.append(hedef)
        hedef["noktalar"].append(nokta)
        hedef["olaylar"].append(olay.get("event_id"))
        hedef["agirlik"] += agirlik(olay)
        hedef["tipler"][olay.get("tip", "?")] += 1
        n = len(hedef["noktalar"])
        hedef["merkez"] = (sum(p[0] for p in hedef["noktalar"]) / n,
                           sum(p[1] for p in hedef["noktalar"]) / n)

    for k in kumeler:
        k["olay_sayisi"] = len(k["olaylar"])
        k["agirlik"] = round(k["agirlik"], 2)
        k["tipler"] = dict(k["tipler"])
        k["lat"], k["lon"] = round(k["merkez"][0], 6), round(k["merkez"][1], 6)
        del k["merkez"], k["noktalar"]
    return sorted(kumeler, key=lambda k: -k["agirlik"])


def kamera_bazli(olaylar: list[dict]) -> list[dict]:
    """GPS'siz olaylari kameraya gore gruplar (sabit CCTV).

    Sabit kamera icin "kara nokta" zaten kameranin baktigi yerdir; olmayan bir
    koordinat uretmek yerine kamera siralanir.
    """
    gruplar: dict[str, dict] = {}
    for o in olaylar:
        if _gps(o) is not None:
            continue
        ad = o.get("camera_id") or "(kamera tanimsiz)"
        g = gruplar.setdefault(ad, {"camera_id": ad, "olay_sayisi": 0,
                                    "agirlik": 0.0, "tipler": defaultdict(int),
                                    "olaylar": []})
        g["olay_sayisi"] += 1
        g["agirlik"] += agirlik(o)
        g["tipler"][o.get("tip", "?")] += 1
        g["olaylar"].append(o.get("event_id"))
    for g in gruplar.values():
        g["agirlik"] = round(g["agirlik"], 2)
        g["tipler"] = dict(g["tipler"])
    return sorted(gruplar.values(), key=lambda g: -g["agirlik"])


def analiz(olaylar: list[dict],
           yaricap_m: float = VARSAYILAN_YARICAP_M) -> dict:
    """Kara nokta analizi: cografi kumeler + kamera bazli gruplar.

    Ikisi ayri alanlarda doner; birlestirmek yaniltici olurdu.
    """
    return {
        "yaricap_m": yaricap_m,
        "toplam_olay": len(olaylar),
        "gps_li_olay": sum(1 for o in olaylar if _gps(o) is not None),
        "cografi_kumeler": kumele(olaylar, yaricap_m),
        "kamera_gruplari": kamera_bazli(olaylar),
        "uyari": (
            "Kara nokta analizi bir PLANLAMA ciktisidir; kusur veya ceza "
            "dogurmaz. Agirliklar goreli siralama olcegidir, mutlak risk skoru "
            "degildir - gercek analiz yaralanma ve maddi hasar verisi gerektirir."
        ),
    }
