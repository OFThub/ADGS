"""M7 ceza motoru - ZORUNLU test dosyasi (plan Faz 5 kabul kriteri).

Uc sey kilitlenir:
    1. penalty.py icinde TUTAR YOKTUR (plan: grep -nE '[0-9]{4,}' bos donmeli)
    2. Tablo OLAYIN tarihine gore secilir, bugune gore degil
    3. Uygun tablo yoksa hesaplama YAPILMAZ - eski tabloya sessiz dusus yasak

Tablolar sentetik uretilir; gercek tutarlar teste kopyalanmaz - kopya, guncel
tablodan sapip sessizce eskiyen ikinci bir gercek kaynagi olurdu.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest

from adgs import penalty
from adgs.schema import Event, Party, Violation

# Sentetik tutarlar: gercek ceza tablosuyla ilgisi yok, bilerek kucuk.
_TUTAR_ESKI, _PUAN_ESKI = 111, 5
_TUTAR_YENI, _PUAN_YENI = 222, 7


def _tablo_yaz(dizin: Path, tarih: str, tutar: int, puan: int | None,
               dogrulanmis: bool = False, beyan: str | None = None) -> Path:
    dogrulama = '"2026-01-01"' if dogrulanmis else "null"
    p = dizin / f"{tarih}.yaml"
    p.write_text(
        "meta:\n"
        f'  gecerlilik_baslangic: "{beyan or tarih}"\n'
        f"  dogrulama_tarihi: {dogrulama}\n"
        "cezalar:\n"
        "  - ihlal_kodu: KIRMIZI_ISIK\n"
        '    ktk_madde: "x"\n'
        f"    tutar_try: {tutar}\n"
        f"    ceza_puani: {'null' if puan is None else puan}\n",
        encoding="utf-8",
    )
    return p


@pytest.fixture
def dizin(tmp_path: Path) -> Path:
    _tablo_yaz(tmp_path, "2020-01-01", _TUTAR_ESKI, _PUAN_ESKI)
    _tablo_yaz(tmp_path, "2021-06-15", _TUTAR_YENI, _PUAN_YENI)
    return tmp_path


def _evt(kodlar: list[str] | None = None, tip: str = "IHLAL") -> Event:
    kodlar = ["KIRMIZI_ISIK"] if kodlar is None else kodlar
    return Event(
        event_id="t", tip=tip, alt_tip=kodlar[0] if kodlar else "-",
        t_start=0.0, t_end=1.0, frame_start=0, frame_end=25,
        source_video="-", source_profile="cctv_fixed", conf=0.9,
        evidence={"keyframes": [0]},
        parties=[Party(track_id=1,
                       violations=[Violation(ihlal_kodu=k, conf=0.9) for k in kodlar])],
    )


def _v(evt: Event) -> Violation:
    return evt.parties[0].violations[0]


# --- 1. Kodda tutar yok -----------------------------------------------------


def test_penalty_kodunda_hicbir_tutar_yok():
    """Plan Faz 5 kabul kriteri: grep -nE '[0-9]{4,}' adgs/penalty.py BOS donmeli.

    Kodda gomulu bir tutar, guncellenmedigi anda sessizce yanlis cikti uretir.
    """
    kaynak = Path(penalty.__file__).read_text(encoding="utf-8")
    bulunan = re.findall(r"\d{4,}", kaynak)
    assert bulunan == [], f"penalty.py icinde sayi bulundu: {bulunan}"


# --- 2. Tablo secimi olayin tarihine gore -----------------------------------


def test_tablolar_dosya_adindan_siralanir(dizin: Path):
    tarihler = [t for t, _p in penalty.tablolari_listele(dizin)]
    assert tarihler == [date(2020, 1, 1), date(2021, 6, 15)]


def test_gecmis_tarihli_olay_o_gunun_tablosuyla_hesaplanir(dizin: Path):
    """Gecmis bir video BUGUNUN tablosuyla hesaplanmaz."""
    evt = _evt()
    penalty.uygula([evt], date(2020, 5, 10), dizin)
    assert _v(evt).ceza_tutari_try == _TUTAR_ESKI
    assert _v(evt).ceza_tablosu_tarihi == "2020-01-01"


def test_yeni_tarihli_olay_yeni_tabloyu_kullanir(dizin: Path):
    evt = _evt()
    penalty.uygula([evt], date(2026, 8, 12), dizin)
    assert _v(evt).ceza_tutari_try == _TUTAR_YENI
    assert _v(evt).ceza_puani == _PUAN_YENI
    assert _v(evt).ceza_tablosu_tarihi == "2021-06-15"


def test_gecerlilik_gunu_dahildir(dizin: Path):
    evt = _evt()
    penalty.uygula([evt], date(2021, 6, 15), dizin)
    assert _v(evt).ceza_tablosu_tarihi == "2021-06-15"


def test_tarih_adi_olmayan_dosya_tablo_sayilmaz(tmp_path: Path):
    (tmp_path / "NOTLAR.yaml").write_text("meta: {}\n", encoding="utf-8")
    _tablo_yaz(tmp_path, "2020-01-01", _TUTAR_ESKI, _PUAN_ESKI)
    assert [t for t, _p in penalty.tablolari_listele(tmp_path)] == [date(2020, 1, 1)]


# --- 3. Hesaplanamayan durumlarda REDDETME ----------------------------------


def test_tablo_oncesi_tarihte_hesaplama_yapilmaz(dizin: Path):
    """Eski tabloya sessizce dusmek yasak - hicbiri gecerli degilse hesap yok."""
    evt = _evt()
    penalty.uygula([evt], date(2019, 12, 31), dizin)
    assert _v(evt).ceza_tutari_try is None
    assert _v(evt).ceza_tablosu_tarihi is None
    assert any("gecerli bir ceza tablosu bulunamadi" in n for n in evt.notes)


def test_tarih_verilmezse_hesaplama_yapilmaz(dizin: Path):
    evt = _evt()
    penalty.uygula([evt], None, dizin)
    assert _v(evt).ceza_tutari_try is None
    assert any("olayin tarihi bilinmiyor" in n for n in evt.notes)


def test_tabloda_olmayan_ihlal_icin_tutar_uydurulmaz(dizin: Path):
    evt = _evt(["HIZ_IHLALI"])
    penalty.uygula([evt], date(2026, 8, 12), dizin)
    assert _v(evt).ceza_tutari_try is None
    assert any("HIZ_IHLALI" in n and "kayit yok" in n for n in evt.notes)


def test_dosya_adi_ile_ic_beyan_celisirse_hata(tmp_path: Path):
    """Hangisinin dogru oldugu bilinemez; tahmin yanlis tarihli hesap uretir."""
    _tablo_yaz(tmp_path, "2020-01-01", _TUTAR_ESKI, _PUAN_ESKI, beyan="2019-01-01")
    with pytest.raises(ValueError, match="tutarsiz"):
        penalty.tablo_sec(date(2026, 8, 12), tmp_path)


def test_hic_tablo_yoksa_none(tmp_path: Path):
    assert penalty.tablo_sec(date(2026, 8, 12), tmp_path) is None


# --- Cikti sozlesmesi -------------------------------------------------------


def test_her_hesap_hangi_tablodan_geldigini_tasir(dizin: Path):
    evt = _evt()
    penalty.uygula([evt], date(2026, 8, 12), dizin)
    assert any("2021-06-15 tarihli tablodan alinmistir" in n for n in evt.notes)


def test_erken_odeme_indirimi_hesaplanmaz(dizin: Path):
    """Indirim teblig tarihine baglidir; video bunu bilemez."""
    evt = _evt()
    penalty.uygula([evt], date(2026, 8, 12), dizin)
    assert any("Erken odeme indirimi hesaplanmamistir" in n for n in evt.notes)


def test_dogrulanmamis_tablo_uyarisi_eklenir(dizin: Path):
    evt = _evt()
    penalty.uygula([evt], date(2026, 8, 12), dizin)
    assert any("DOGRULANMAMISTIR" in n for n in evt.notes)


def test_dogrulanmis_tabloda_uyari_yok(tmp_path: Path):
    _tablo_yaz(tmp_path, "2020-01-01", _TUTAR_ESKI, _PUAN_ESKI, dogrulanmis=True)
    evt = _evt()
    penalty.uygula([evt], date(2026, 8, 12), tmp_path)
    assert not any("DOGRULANMAMISTIR" in n for n in evt.notes)


def test_altyapi_olayi_ceza_almaz(dizin: Path):
    evt = _evt(tip="ALTYAPI")
    penalty.uygula([evt], date(2026, 8, 12), dizin)
    assert _v(evt).ceza_tutari_try is None


def test_ceza_puani_null_ise_none_kalir(tmp_path: Path):
    _tablo_yaz(tmp_path, "2020-01-01", _TUTAR_ESKI, None)
    evt = _evt()
    penalty.uygula([evt], date(2026, 8, 12), tmp_path)
    assert _v(evt).ceza_tutari_try == _TUTAR_ESKI
    assert _v(evt).ceza_puani is None


# --- Gercek tablo -----------------------------------------------------------


def test_repodaki_tablo_yuklenebiliyor():
    tablolar = penalty.tablolari_listele()
    assert tablolar, "config/penalties altinda tablo yok"
    for tarih, yol in tablolar:
        assert penalty.tablo_sec(tarih) is not None, yol
