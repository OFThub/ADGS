"""M7 - Ceza ve ceza puani hesaplama.

BU DOSYADA TEK BIR TUTAR YOKTUR ve olmamalidir. Tutarlar tarih damgali YAML
tablolarindan gelir (config/penalties/<gecerlilik-tarihi>.yaml). Kural
tests/test_penalty.py icindeki grep testiyle zorlanir.

Neden: ceza tutarlari sik degisir - yeniden degerleme ve kanun degisikligi ayni
yil icinde iki kez olabilir. Kodda gomulu bir tutar, guncellenmedigi anda
sessizce yanlis cikti uretmeye baslar ve bunu kimse fark etmez.

Tablo secimi OLAYIN tarihine gore yapilir, bugune gore degil: gecmis tarihli
bir video, cekildigi tarihte gecerli olan tabloyla hesaplanir.

Uygun tablo bulunamazsa HESAPLAMA YAPILMAZ. Eski bir tabloya sessizce dusmek
yasaktir - yanlis bir tutar, eksik bir tutardan cok daha zararlidir.

Erken odeme indirimi hesaplanmaz: indirim teblig tarihine baglidir ve video
bunu bilemez. Yalnizca not olarak belirtilir.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from adgs.schema import Event

_VARSAYILAN_DIZIN = Path(__file__).resolve().parent.parent / "config" / "penalties"

_NOT_TARIH_YOK = (
    "Ceza hesaplanmadi: olayin tarihi bilinmiyor. Tarih olmadan hangi ceza "
    "tablosunun gecerli oldugu belirlenemez (--tarih ile verin)."
)
_NOT_INDIRIM = (
    "Erken odeme indirimi hesaplanmamistir; indirim teblig tarihine baglidir."
)


def tablolari_listele(dizin: str | Path | None = None) -> list[tuple[date, Path]]:
    """(gecerlilik_baslangici, dosya) ciftlerini tarihe gore sirali dondurur.

    Gecerlilik tarihi DOSYA ADINDAN okunur; adi ISO tarih olmayan dosyalar
    tablo sayilmaz - boylece not/README dosyalari yanlislikla tablo olarak
    yuklenmez.
    """
    d = Path(dizin) if dizin else _VARSAYILAN_DIZIN
    if not d.exists():
        return []
    bulunan: list[tuple[date, Path]] = []
    for p in d.glob("*.yaml"):
        try:
            bulunan.append((date.fromisoformat(p.stem), p))
        except ValueError:
            continue
    return sorted(bulunan)


def tablo_sec(olay_tarihi: date, dizin: str | Path | None = None) -> dict | None:
    """Olay tarihinde GECERLI OLAN tabloyu dondurur; yoksa None.

    None donmesi "ceza yok" degil "hesaplanamaz" demektir - cagiran taraf bunu
    sifirla doldurmamalidir.
    """
    import yaml

    uygun = [(t, p) for t, p in tablolari_listele(dizin) if t <= olay_tarihi]
    if not uygun:
        return None
    gecerlilik, yol = uygun[-1]
    cfg = yaml.safe_load(yol.read_text(encoding="utf-8")) or {}
    meta = cfg.setdefault("meta", {})
    beyan = meta.get("gecerlilik_baslangic")
    if beyan and beyan != gecerlilik.isoformat():
        # Dosya adi ile ic beyan celisirse hangisinin dogru oldugu bilinemez;
        # tahmin etmek yanlis tarihli bir ceza hesabi uretir.
        raise ValueError(
            f"Ceza tablosu tutarsiz: dosya adi {gecerlilik.isoformat()} ama "
            f"meta.gecerlilik_baslangic {beyan} ({yol.name}). Ikisi ayni olmali."
        )
    meta["gecerlilik_baslangic"] = gecerlilik.isoformat()
    return cfg


def uygula(events: list[Event], olay_tarihi: date | None = None,
           dizin: str | Path | None = None) -> list[Event]:
    """Ihlallere tahmini tutar ve ceza puani yazar (yerinde doldurur).

    Her ihlal hangi tablodan hesaplandigini `ceza_tablosu_tarihi` alaninda
    tasir - hesap sonradan denetlenebilir olmalidir.
    """
    ilgili = [e for e in events if e.tip in ("KAZA", "IHLAL")]
    if not ilgili:
        return events

    if olay_tarihi is None:
        for evt in ilgili:
            evt.notes.append(_NOT_TARIH_YOK)
        return events

    tablo = tablo_sec(olay_tarihi, dizin)
    if tablo is None:
        for evt in ilgili:
            evt.notes.append(
                f"Ceza hesaplanmadi: {olay_tarihi.isoformat()} tarihinde gecerli "
                "bir ceza tablosu bulunamadi. Eski tabloya DUSULMEMISTIR."
            )
        return events

    meta = tablo.get("meta") or {}
    tarih = meta["gecerlilik_baslangic"]
    dogrulanmis = bool(meta.get("dogrulama_tarihi"))
    kayitlar = {
        c["ihlal_kodu"]: c for c in (tablo.get("cezalar") or []) if c.get("ihlal_kodu")
    }

    for evt in ilgili:
        eksik: list[str] = []
        hesaplandi = False
        for taraf in evt.parties:
            for ihlal in taraf.violations:
                kayit = kayitlar.get(ihlal.ihlal_kodu)
                if kayit is None:
                    eksik.append(ihlal.ihlal_kodu)
                    continue
                ihlal.ceza_tutari_try = int(kayit["tutar_try"])
                puan = kayit.get("ceza_puani")
                ihlal.ceza_puani = None if puan is None else int(puan)
                ihlal.ceza_tablosu_tarihi = tarih
                hesaplandi = True
        if eksik:
            evt.notes.append(
                f"Ceza hesaplanmadi ({', '.join(sorted(set(eksik)))}): "
                f"{tarih} tarihli tabloda bu ihlal icin kayit yok."
            )
        if hesaplandi:
            evt.notes.append(f"Ceza tutarlari {tarih} tarihli tablodan alinmistir.")
            evt.notes.append(_NOT_INDIRIM)
            if not dogrulanmis:
                evt.notes.append(
                    f"UYARI: {tarih} tarihli ceza tablosu DOGRULANMAMISTIR "
                    "(meta.dogrulama_tarihi bos). Tutarlar resmi rehberle "
                    "karsilastirilmadan kullanilmamalidir."
                )
    return events
