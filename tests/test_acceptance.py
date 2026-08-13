"""Kabul kriteri tablosu (adgs.acceptance).

Bu tablonun degeri tek bir seye bagli: OLCULMEDI ile GECTI'yi karistirmamasi.
Karistirirsa staj raporunda "kriter saglandi" yazar ve aslinda hicbir sey
olculmemis olur. Testlerin cogu bu ayrimi koruyor.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from adgs import acceptance as ka


def _kabul_yaz(kok: Path, dizin: str, map50: float) -> None:
    p = kok / "runs" / dizin
    p.mkdir(parents=True, exist_ok=True)
    (p / "kabul.json").write_text(
        json.dumps({"map50": map50, "zaman": "2026-08-13T09:00:00+00:00"}),
        encoding="utf-8")


# --- olculmedi / gecti / kaldi ayrimi ---------------------------------------


def test_olcum_yoksa_olculmedi(tmp_path: Path, monkeypatch):
    """Model olculmemisse GECTI de KALDI da degildir."""
    monkeypatch.setattr(ka, "KOK", tmp_path)
    s = ka.Sonuc()
    ka._model_kriteri(s, "1", "x", "rdd2022/yolo26s", 0.45, "adgs eval ...")
    assert s.kriterler[0].durum == ka.OLCULMEDI
    assert "adgs eval" in s.kriterler[0].not_


def test_hedefin_altinda_kalan_olcum_kaldi(tmp_path: Path, monkeypatch):
    """Olculup hedefi tutturamamak OLCULMEDI degil KALDI'dir."""
    monkeypatch.setattr(ka, "KOK", tmp_path)
    _kabul_yaz(tmp_path, "rdd2022/yolo26s", 0.1916)
    s = ka.Sonuc()
    ka._model_kriteri(s, "1", "x", "rdd2022/yolo26s", 0.45, "adgs eval ...")
    assert s.kriterler[0].durum == ka.KALDI
    assert s.kriterler[0].deger == "0.1916"


def test_hedefi_tutturan_olcum_gecti(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(ka, "KOK", tmp_path)
    _kabul_yaz(tmp_path, "cardd/yolo26s", 0.5087)
    s = ka.Sonuc()
    ka._model_kriteri(s, "6", "x", "cardd/yolo26s", 0.40, "adgs eval ...")
    assert s.kriterler[0].durum == ka.GECTI


def test_bozuk_olcum_dosyasi_gecti_uretmiyor(tmp_path: Path, monkeypatch):
    """Okunamayan olcum dosyasi sessizce basari sayilmamali."""
    monkeypatch.setattr(ka, "KOK", tmp_path)
    p = tmp_path / "runs" / "rdd2022" / "yolo26s"
    p.mkdir(parents=True)
    (p / "kabul.json").write_text("{bozuk json", encoding="utf-8")
    s = ka.Sonuc()
    ka._model_kriteri(s, "1", "x", "rdd2022/yolo26s", 0.45, "adgs eval ...")
    assert s.kriterler[0].durum == ka.OLCULMEDI


# --- davranissal kriterler ---------------------------------------------------


def test_hiz_modulu_reddi_olculuyor():
    """Faz 4'un asil kriteri gercekten kod yolu calistirilarak olculur."""
    s = ka.Sonuc()
    ka._hiz_reddi(s)
    assert s.kriterler[0].durum == ka.GECTI
    assert "TESPIT YAPILMADI" in s.kriterler[0].not_


def test_penalty_kodunda_tutar_olmadigi_olculuyor():
    s = ka.Sonuc()
    ka._ceza_kodda_sayi_yok(s)
    assert s.kriterler[0].durum == ka.GECTI, s.kriterler[0].deger


def test_ceza_tablosu_yoksa_reddediliyor():
    s = ka.Sonuc()
    ka._ceza_tablosu_yoksa_reddediyor(s)
    assert s.kriterler[0].durum == ka.GECTI


def test_dogrulanmamis_mevzuat_gecti_sayilmiyor(tmp_path: Path, monkeypatch):
    """dogrulama_tarihi null iken tablo YESIL gorunmemeli."""
    monkeypatch.setattr(ka, "KOK", tmp_path)
    (tmp_path / "config" / "rules").mkdir(parents=True)
    (tmp_path / "config" / "penalties").mkdir(parents=True)
    (tmp_path / "config" / "rules" / "fault_ktk84.yaml").write_text(
        "meta:\n  dogrulama_tarihi: null\n", encoding="utf-8")
    s = ka.Sonuc()
    ka._mevzuat_dogrulamasi(s)
    assert s.kriterler[0].durum == ka.OLCULMEDI
    assert "mevzuat.gov.tr" in s.kriterler[0].not_


def test_yukleme_sadece_dosya_istiyor():
    """Kullanicidan ek bilgi istenmedigi de bir kabul kriteridir."""
    s = ka.Sonuc()
    ka._tek_girdi_video(s)
    assert s.kriterler[0].durum == ka.GECTI, s.kriterler[0].deger


# --- tablo butunlugu ---------------------------------------------------------


def test_tum_fazlar_tabloda_var():
    s = ka.olc()
    assert {k.faz for k in s.kriterler} >= {"0", "1", "2", "3", "4", "5", "6", "7"}


def test_her_olculmedi_satirinin_sebebi_var():
    """Sebepsiz OLCULMEDI, "yapmadik" demenin kibar hali olurdu."""
    for k in ka.olc().kriterler:
        if k.durum == ka.OLCULMEDI:
            assert k.not_.strip(), f"sebepsiz OLCULMEDI: {k.ad}"


def test_tablo_biciminde_sayim_var():
    metin = ka.bicimle(ka.olc())
    assert "GECTI:" in metin and "KALDI:" in metin and "OLCULMEDI:" in metin


@pytest.mark.parametrize("durum", [ka.GECTI, ka.KALDI, ka.OLCULMEDI])
def test_sayim_durumlari_kapsiyor(durum):
    s = ka.Sonuc()
    s.ekle("9", "x", "y", durum)
    assert s.sayim()[durum] == 1
