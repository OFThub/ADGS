"""M6 - Kusur / sorumluluk analiz motoru. Sistemin hukuken EN HASSAS parcasi.

BU MODUL YUZDE URETMEZ.

Karayollari Trafik Yonetmeligi m.156/3 uyarinca tutanagi duzenleyen gorevli
bile kusur orani belirtmez; orani TRAMER belirler. Motor bu yuzden tutanak
mantigini izler: her taraf icin hangi KTK maddesinin ihlal edildigini ve bunun
KTK m.84 kapsaminda ASLI mi TALI mi oldugunu raporlar.

Uc sonuc vardir:
    ASLI              ihlal, m.84'un KAPALI listesinde eslesti
    TALI              ihlal tespit edildi ama listede yok
    TESPIT_EDILEMEDI  o tarafta hic ihlal tespit edilemedi

TESPIT_EDILEMEDI "KUSURSUZ" DEMEK DEGILDIR ve boyle okunmamalidir. Goruntude
ihlal gorulmemis olmasi ihlal olmadigini gostermez - kamera acisi, okluzyon ve
kare hizi sinirlari vardir. Bu ayrim koda gomulmustur: ilgili tarafa
kaldirilamayan bir not eklenir.

GPU gerektirmez, goruntuye dokunmaz, yalnizca Event + YAML okur. Bu yuzden
sistemin hukuken en hassas kismi ayni zamanda birim testle tam kapsanan kismi.
"""

from __future__ import annotations

from pathlib import Path

from adgs.schema import Event

_VARSAYILAN_TABLO = (
    Path(__file__).resolve().parent.parent / "config" / "rules" / "fault_ktk84.yaml"
)

ASLI = "ASLI"
TALI = "TALI"
TESPIT_EDILEMEDI = "TESPIT_EDILEMEDI"

_NOT_DOGRULANMAMIS = (
    "UYARI: kusur tablosu (KTK m.84) dogrulanmamistir - "
    "config/rules/fault_ktk84.yaml icindeki meta.dogrulama_tarihi bostur."
)


def yukle(yol: str | Path | None = None) -> dict:
    """Kusur tablosunu okur.

    Dosya yoksa SESSIZCE bos tablo donmez: tablosuz calismak her ihlali TALI
    yapar ve bu, sessizce yanlis bir hukuki sonuctur.
    """
    import yaml

    p = Path(yol) if yol else _VARSAYILAN_TABLO
    if not p.exists():
        raise FileNotFoundError(
            f"Kusur tablosu bulunamadi: {p}. Tablosuz kusur siniflandirmasi "
            "yapilamaz - her ihlal yanlislikla TALI sayilirdi."
        )
    return yaml.safe_load(p.read_text(encoding="utf-8")) or {}


def _asli_eslesme(tablo: dict) -> dict[str, dict]:
    return {
        h["ihlal_kodu"]: h
        for h in (tablo.get("asli_kusur_halleri") or [])
        if h.get("ihlal_kodu")
    }


def sinifla(ihlal_kodu: str, tablo: dict) -> tuple[str, str | None]:
    """(kusur_sinifi, ktk_madde) dondurur.

    m.84 KAPALI bir listedir: eslesme yoksa ihlal TALI'dir. "Benzer" bir maddeye
    yaklastirma YAPILMAZ - yanlis bir asli kusur atamasi, bir vatandasin adina
    yanlis bir kayit uretir.
    """
    h = _asli_eslesme(tablo).get(ihlal_kodu)
    if h is not None:
        return ASLI, h.get("madde")
    return tablo.get("varsayilan_sinif", TALI), None


def uygula(events: list[Event], tablo: dict | None = None) -> list[Event]:
    """Her Event'in taraflarindaki ihlalleri siniflandirir (yerinde doldurur).

    Yalnizca KAZA ve IHLAL olaylari islenir; ALTYAPI belediyenin kendi gorev
    alanidir ve kusur/ceza dogurmaz.
    """
    tablo = yukle() if tablo is None else tablo
    ihlal_yoksa = tablo.get("ihlal_yoksa", TESPIT_EDILEMEDI)
    dogrulanmis = bool((tablo.get("meta") or {}).get("dogrulama_tarihi"))

    for evt in events:
        if evt.tip not in ("KAZA", "IHLAL"):
            continue
        for taraf in evt.parties:
            if not taraf.violations:
                # Bos birakmak "kusursuz" gibi okunur - acikca yazilir.
                evt.notes.append(
                    f"Iz #{taraf.track_id}: {ihlal_yoksa} - goruntude ihlal "
                    "tespit edilemedi. Bu, o tarafin KUSURSUZ oldugu anlamina "
                    "GELMEZ; yalnizca bu kayittan ihlal cikarilamadigini gosterir."
                )
                continue
            for ihlal in taraf.violations:
                sinif, madde = sinifla(ihlal.ihlal_kodu, tablo)
                ihlal.kusur_sinifi = sinif
                ihlal.ktk_madde = madde
        if not dogrulanmis and _NOT_DOGRULANMAMIS not in evt.notes:
            evt.notes.append(_NOT_DOGRULANMAMIS)
    return events


def ozet(evt: Event) -> list[str]:
    """Insan okunur kusur ozeti - `adgs explain` ve rapor icin.

    Bilerek YUZDE ICERMEZ: "taraf A %70 kusurlu" turu bir cumle bu sistemin
    uretebilecegi bir cikti degildir.
    """
    satirlar: list[str] = []
    for taraf in evt.parties:
        if not taraf.violations:
            satirlar.append(f"iz #{taraf.track_id}: {TESPIT_EDILEMEDI} (kusursuz DEGIL)")
            continue
        for ihlal in taraf.violations:
            madde = f"KTK {ihlal.ktk_madde}" if ihlal.ktk_madde else "KTK m.84 disinda"
            satirlar.append(
                f"iz #{taraf.track_id}: {ihlal.ihlal_kodu} -> {madde} "
                f"[{ihlal.kusur_sinifi or '?'}]"
            )
    return satirlar
