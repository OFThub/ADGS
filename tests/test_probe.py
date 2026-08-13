"""Videodan parametre turetme (adgs.probe).

Test edilen iddia: kullanici hicbir ek bilgi vermeden, turetilen degerler ya
bir OLCUME ya da dosya adindaki acik bir isarete dayanir - ve turetilemeyen
alan BOS kalir. Bos kalmasi eksiklik degil, kontrol edilen davranistir:
tahmin edilen bir cekim tarihi yanlis ceza tablosu, tahmin edilen bir kamera
yanlis dur cizgisi demektir.

Kamera hareketi sentetik videoyla test edilir: sabit dokulu arka plan uzerinde
gecen kutu (sabit kamera) ile kayan arka plan (araca monteli).
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import cv2
import numpy as np
import pytest

from adgs import probe


def _desenli_arkaplan(w: int = 320, h: int = 240) -> np.ndarray:
    """Faz korelasyonu icin dokulu sahne - duz renkli kare kayma uretmez."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


def _video_yaz(yol: Path, kareler: list[np.ndarray], fps: float = 20.0) -> Path:
    h, w = kareler[0].shape[:2]
    vw = cv2.VideoWriter(str(yol), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for k in kareler:
        vw.write(k)
    vw.release()
    return yol


@pytest.fixture
def sabit_video(tmp_path: Path) -> Path:
    """Sabit kamera: arka plan duruyor, sadece bir kutu geciyor."""
    arka = _desenli_arkaplan()
    kareler = []
    for i in range(60):
        k = arka.copy()
        x = 10 + i * 4
        cv2.rectangle(k, (x, 100), (x + 40, 140), (0, 0, 0), -1)
        kareler.append(k)
    return _video_yaz(tmp_path / "sabit.mp4", kareler)


@pytest.fixture
def hareketli_video(tmp_path: Path) -> Path:
    """Araca monteli kamera: tum sahne kayiyor."""
    genis = _desenli_arkaplan(w=1000)
    kareler = [genis[:, i * 8:i * 8 + 320].copy() for i in range(60)]
    return _video_yaz(tmp_path / "hareketli.mp4", kareler)


# --- Olcum ------------------------------------------------------------------


def test_sabit_kamera_olculuyor(sabit_video: Path):
    kayma = probe.kamera_kaymasi(sabit_video)
    assert kayma is not None
    assert kayma < probe.SABIT_UST, f"sabit sahne hareketli olculdu: {kayma}"


def test_hareketli_kamera_olculuyor(hareketli_video: Path):
    kayma = probe.kamera_kaymasi(hareketli_video)
    assert kayma is not None
    assert kayma > probe.HAREKETLI_ALT, f"kayan sahne sabit olculdu: {kayma}"


def test_sabit_video_kaza_ihlal_tarar(sabit_video: Path):
    d = probe.incele(sabit_video)
    assert d["profile"] == "cctv_fixed"
    assert "accident" in d["detect"] and "ihlal" in d["detect"]


def test_hareketli_video_yol_hasari_tarar(hareketli_video: Path):
    d = probe.incele(hareketli_video)
    assert d["profile"] == "vehicle_mounted"
    assert d["detect"] == "roaddamage"


def test_okunamayan_video_tahmin_etmiyor(tmp_path: Path):
    """Olcum yapilamayinca profile KARAR VERILMEZ; iki dedektor ailesi de calisir."""
    bozuk = tmp_path / "bozuk.mp4"
    bozuk.write_bytes(b"bu bir video degil")
    assert probe.kamera_kaymasi(bozuk) is None
    d = probe.incele(bozuk)
    assert "roaddamage" in d["detect"] and "accident" in d["detect"]
    assert "OLCULEMEDI" in d["gerekce"][0]


def test_her_karara_gerekce_yaziliyor(sabit_video: Path):
    """Turetilmis parametre gerekcesiz gosterilmez - kullanici neden'i gorur."""
    d = probe.incele(sabit_video)
    assert len(d["gerekce"]) == 3
    assert all(g.strip() for g in d["gerekce"])


# --- Tarih ------------------------------------------------------------------


@pytest.mark.parametrize("ad,beklenen", [
    ("kavsak_2026-07-15.mp4", "2026-07-15"),
    ("kavsak_2026_07_15.mp4", "2026-07-15"),
    ("20260715_kavsak.mp4", "2026-07-15"),
    ("15.07.2026_kayit.mp4", "2026-07-15"),
])
def test_tarih_dosya_adindan_okunuyor(ad: str, beklenen: str):
    assert probe.tarih_bul(ad, bugun=date(2026, 8, 13))[0] == beklenen


@pytest.mark.parametrize("ad", [
    "kavsak.mp4",              # tarih yok
    "2026-13-45_x.mp4",        # boyle bir tarih yok
    "2099-01-01_x.mp4",        # gelecekte cekilmis olamaz
    "12345678_x.mp4",          # rastgele sayi dizisi
])
def test_tarih_uydurulmuyor(ad: str):
    """Cozulemeyen tarih BUGUNE DUSMEZ - yanlis ceza tablosu secilirdi."""
    tarih, gerekce = probe.tarih_bul(ad, bugun=date(2026, 8, 13))
    assert tarih is None
    assert "BOS" in gerekce


# --- Kamera -----------------------------------------------------------------


@pytest.fixture
def kamera_dizini(tmp_path: Path) -> Path:
    d = tmp_path / "cameras"
    d.mkdir()
    (d / "arnavutkoy_kavsak_01.yaml").write_text("camera_id: x", encoding="utf-8")
    (d / "demo_sentetik.yaml").write_text("camera_id: y", encoding="utf-8")
    return d


def test_kamera_dosya_adindan_eslesiyor(kamera_dizini: Path):
    k, _ = probe.kamera_bul("20260715_arnavutkoy_kavsak_01.mp4", kamera_dizini)
    assert k == "arnavutkoy_kavsak_01"


def test_tanimsiz_kamera_eslesmiyor(kamera_dizini: Path):
    """Cozunurluk/benzerlik TAHMINI YOK: yanlis kamera = yanlis dur cizgisi."""
    k, gerekce = probe.kamera_bul("rastgele_kavsak.mp4", kamera_dizini)
    assert k is None
    assert "BOS" in gerekce


def test_zararli_dosya_adi_kamera_uretmiyor(kamera_dizini: Path):
    k, _ = probe.kamera_bul("<svg onload=alert(1)>.mp4", kamera_dizini)
    assert k is None


def test_kamera_kimligi_bicim_suzgecinden_geciyor(tmp_path: Path):
    """Kimlik veritabanina yazilip arayuzde gosteriliyor - bicim sinirli."""
    d = tmp_path / "cameras"
    d.mkdir()
    (d / "adgs_test_bicimsiz name.yaml").write_text("x: 1", encoding="utf-8")
    k, _ = probe.kamera_bul("adgs_test_bicimsiz name.mp4", d)
    assert k is None
