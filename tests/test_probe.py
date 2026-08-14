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
    # profil, tarih, kamera, guven esigi
    assert len(d["gerekce"]) == 4
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


# --- Tarih verilmemisse bugune dusme (kullanici istegi) ----------------------


def test_tarih_okunamazsa_bugun_varsayiliyor():
    """Kullanici istegi: tarih yoksa SIMDI baz alinsin."""
    t, gerekce = probe.tarih_bul("kavsak.mp4", bugun=date(2026, 8, 13),
                                 bugune_dus=True)
    assert t == "2026-08-13"


def test_varsayilan_tarih_varsayim_oldugunu_soyluyor():
    """Olculmus tarihle varsayilan tarih ciktida ayirt edilebilmeli.

    Arsiv videosu bugunun ceza tablosuyla hesaplanirsa tutar yanlis cikar;
    bu ihtimalin gorunur kalmasi icin gerekce VARSAYILDI diye isaretlenir.
    """
    _, gerekce = probe.tarih_bul("kavsak.mp4", bugun=date(2026, 8, 13),
                                 bugune_dus=True)
    assert "VARSAYILDI" in gerekce
    assert "yanlis cikabilir" in gerekce


def test_dosya_adindaki_tarih_varsayima_tercih_ediliyor():
    """Gercek tarih varsa bugune DUSULMEZ."""
    t, gerekce = probe.tarih_bul("kavsak_2026-07-15.mp4", bugun=date(2026, 8, 13),
                                 bugune_dus=True)
    assert t == "2026-07-15"
    assert "VARSAYILDI" not in gerekce


def test_incele_varsayilan_olarak_bugune_dusuyor(sabit_video):
    """Yukleme yolunda tarih artik bos kalmiyor."""
    d = probe.incele(sabit_video, bugun=date(2026, 8, 13))
    assert d["tarih"] == "2026-08-13"
    assert any("VARSAYILDI" in g for g in d["gerekce"])


# --- Sahne kesmesi ----------------------------------------------------------


def test_sahne_kesmesi_bulunuyor(tmp_path: Path):
    """Iki farkli sahnenin birlestigi kare kesme olarak isaretlenmeli."""
    rng = np.random.default_rng(1)
    mavi = np.zeros((90, 160, 3), np.uint8); mavi[:, :, 0] = 220
    kirmizi = np.zeros((90, 160, 3), np.uint8); kirmizi[:, :, 2] = 220
    kareler = [mavi.copy() for _ in range(15)] + [kirmizi.copy() for _ in range(15)]
    yol = _video_yaz(tmp_path / "kesmeli.mp4", kareler)
    kesmeler = probe.sahne_kesmeleri(yol)
    assert kesmeler, "kesme bulunamadi"
    assert any(13 <= k <= 17 for k in kesmeler), f"kesme yeri yanlis: {kesmeler}"


def test_tek_sahnede_kesme_yok(tmp_path: Path):
    """Sabit sahne kesme uretmemeli - yoksa gecerli kazalar elenirdi."""
    arka = _desenli_arkaplan()
    kareler = []
    for i in range(40):
        k = arka.copy()
        cv2.rectangle(k, (10 + i * 3, 100), (50 + i * 3, 140), (0, 0, 0), -1)
        kareler.append(k)
    yol = _video_yaz(tmp_path / "tek.mp4", kareler)
    assert probe.sahne_kesmeleri(yol) == []


def test_okunamayan_video_kesme_uretmiyor(tmp_path: Path):
    bozuk = tmp_path / "bozuk.mp4"
    bozuk.write_bytes(b"video degil")
    assert probe.sahne_kesmeleri(bozuk) == []


# --- Gece sahnesinde guven esigi --------------------------------------------
#
# OLCULDU: gece sahnesinde varsayilan esik (0.35) 17 karede 0 arac buldu,
# 0.15 ile 9 arac. Gece kaydinda arac isik lekesine donuyor.


def _duz_video(yol: Path, deger: int, n: int = 20) -> Path:
    kare = np.full((90, 160, 3), deger, np.uint8)
    return _video_yaz(yol, [kare.copy() for _ in range(n)])


def test_gece_sahnesinde_esik_dusuruluyor(tmp_path: Path):
    yol = _duz_video(tmp_path / "gece.mp4", 40)
    isik = probe.isik_seviyesi(yol)
    assert isik is not None and isik < probe.GECE_ESIGI
    conf, gerekce = probe._conf_coz(isik)
    assert conf == probe.GECE_CONF
    assert "gece" in gerekce.lower()


def test_gunduz_sahnesinde_varsayilan_esik(tmp_path: Path):
    yol = _duz_video(tmp_path / "gunduz.mp4", 200)
    conf, _ = probe._conf_coz(probe.isik_seviyesi(yol))
    assert conf == probe.VARSAYILAN_CONF


def test_dusuk_esik_yanlis_pozitif_riskini_soyluyor(tmp_path: Path):
    """Esik dusurmek bedava degil - gerekce bunu yaziyor."""
    _, gerekce = probe._conf_coz(50.0)
    assert "YANLIS POZITIF" in gerekce


def test_isik_olculemezse_varsayilan(tmp_path: Path):
    bozuk = tmp_path / "bozuk.mp4"
    bozuk.write_bytes(b"video degil")
    assert probe.isik_seviyesi(bozuk) is None
    assert probe._conf_coz(None)[0] == probe.VARSAYILAN_CONF


def test_incele_conf_donduruyor(sabit_video):
    d = probe.incele(sabit_video)
    assert "conf" in d and 0 < d["conf"] <= 1
    assert len(d["gerekce"]) == 4
