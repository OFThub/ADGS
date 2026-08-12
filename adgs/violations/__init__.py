"""M4 - Trafik ihlali tespiti.

Her ihlal tipi ayri bir modul, hepsi ayni sozlesmeyi uygular:

    tespit_et(izler: list[Track], ctx: Baglam) -> list[Event]

Boylece bir tipi acmak/kapatmak digerlerine dokunmaz (plan Faz 4 kabul
kriteri). Aktiflik config/rules/violations.yaml -> aktif altindadir; CLI
--detect ile acikca istenen tip config'i gecersiz kilar ama bu bir uyari
olarak kaydedilir - sistem sessizce farkli davranmaz.

Modullerin hicbiri kendi tespitini yapmaz: hepsi M2'nin urettigi Track
listesini okur, boylece bir aracin ID'si tum modullerde ayni kalir.
"""

from __future__ import annotations

from pathlib import Path

from adgs.schema import Event, Track
from adgs.violations import lane, parking, redlight, speed, tailgating, wrongway
from adgs.violations.base import Baglam

__all__ = [
    "Baglam", "KAYIT", "FAZ_4A", "KALIBRASYONA_BAGLI",
    "kural_yukle", "kamera_yukle", "tespit_et",
]

# CLI adi -> modul. CLI adlari kisa (dosya adlariyla ayni); rapora yazilan
# ihlal kodu her modulun KOD sabitidir (KIRMIZI_ISIK, TERS_YON, ...).
KAYIT = {
    "redlight": redlight,
    "wrongway": wrongway,
    "parking": parking,
    "lane": lane,
    "tailgating": tailgating,
    "speed": speed,
}

# Plan Faz 4a - once bu uc tip teslim edilir.
FAZ_4A = ("redlight", "wrongway", "parking")

# Kalibrasyon dogrulanmadan CALISMAYAN tipler (plan Faz 4 kabul kriteri).
KALIBRASYONA_BAGLI = ("speed", "tailgating")

_VARSAYILAN_KURAL = (
    Path(__file__).resolve().parents[2] / "config" / "rules" / "violations.yaml"
)


def kural_yukle(yol: str | Path | None = None) -> dict:
    """config/rules/violations.yaml okur. Dosya yoksa bos sozluk (varsayilanlar)."""
    import yaml

    p = Path(yol) if yol else _VARSAYILAN_KURAL
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def kamera_yukle(yol: str | Path) -> dict:
    """config/cameras/<id>.yaml ham sozlugu - geometri tanimlari icin.

    calib.yukle() ayni dosyayi okur ama yalnizca homografiyi cikarir; dedektorler
    dur cizgisi, serit ve poligon tanimlarina da ihtiyac duyar.
    """
    import yaml

    p = Path(yol)
    if not p.exists():
        return {}
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def tespit_et(izler: list[Track], ctx: Baglam,
              tipler: list[str] | None = None) -> list[Event]:
    """Secili ihlal dedektorlerini calistirir.

    `tipler` None ise config'te aktif olanlar calisir. Verilirse tam olarak o
    tipler calisir; config'te kapali olan bir tip acikca istenmisse calisir ama
    bu ctx.uyarilar'a yazilir.
    """
    aktif = ctx.kural.get("aktif") or {}
    secili = list(tipler) if tipler is not None else [
        ad for ad in KAYIT if aktif.get(ad, False)
    ]

    olaylar: list[Event] = []
    for ad in secili:
        modul = KAYIT.get(ad)
        if modul is None:
            ctx.uyarilar.append(f"{ad}: bilinmeyen ihlal tipi - atlandi")
            continue
        if tipler is not None and not aktif.get(ad, True):
            ctx.uyarilar.append(
                f"{ad}: config'te kapali ama --detect ile acikca istendi - calistirildi"
            )
        olaylar += modul.tespit_et(izler, ctx)
    return olaylar
