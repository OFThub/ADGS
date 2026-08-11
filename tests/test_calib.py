"""Kalibrasyon - hiza bagli her ciktinin guvenlik kapisi.

Buradaki en onemli test dogru mesafe hesabi degil, YANLIS kalibrasyonun
REDDEDILMESIDIR: gecersiz homografi sessizce gecerse yanlis hiz, yanlis hiz
yanlis kusur/ceza on degerlendirmesi uretir.
"""

from __future__ import annotations

from pathlib import Path

from adgs import calib

# 1000 piksel = 10 metre olan basit, tam olculebilir bir yer duzlemi.
_PIKSEL = [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]
_DUNYA = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]]


def _cfg(gercek_m: float, maks: float = 10.0) -> dict:
    return {
        "camera_id": "test_kamera",
        "maks_hata_yuzde": maks,
        "homografi": {"piksel": _PIKSEL, "dunya": _DUNYA},
        # Fit noktalarindan bagimsiz olcum: 500 piksel = 5.0 m olmali.
        "dogrulama": [{"piksel": [[0, 0], [500, 0]], "gercek_m": gercek_m}],
    }


def test_dogru_kalibrasyon_kabul_edilir():
    k = calib.kalibre_et(_cfg(5.0))
    assert k.gecerli is True
    assert k.hata_yuzde < 0.01


def test_yanlis_kalibrasyon_reddedilir():
    # Gercekte 5 m olan mesafe 10 m diye beyan edilirse hata %50 -> red.
    k = calib.kalibre_et(_cfg(10.0))
    assert k.gecerli is False
    assert k.hata_yuzde > 10.0


def test_dogrulama_yoksa_reddedilir():
    cfg = _cfg(5.0)
    cfg["dogrulama"] = []
    k = calib.kalibre_et(cfg)
    assert k.gecerli is False
    assert any("Dogrulama" in n for n in k.notlar)


def test_esik_gecmeyen_hata_reddedilir():
    # olculen 5.0 m, beyan 5.4 m -> ~%7.4 hata: %10 esikte gecer, %5'te gecmez.
    assert calib.kalibre_et(_cfg(5.4)).gecerli is True
    assert calib.kalibre_et(_cfg(5.4, maks=5.0)).gecerli is False


def test_gecersiz_kalibrasyonda_mesafe_none_doner():
    # Kritik davranis: olculemedi -> None. Asla 0 veya tahmin donmez.
    k = calib.kalibre_et(_cfg(10.0))
    assert calib.mesafe_m(k, (0, 0), (500, 0)) is None
    assert calib.hiz_kmh(k, (0, 0), (500, 0), dt_s=1.0) is None


def test_gecerli_kalibrasyonda_mesafe_ve_hiz_hesaplanir():
    k = calib.kalibre_et(_cfg(5.0))
    m = calib.mesafe_m(k, (0, 0), (1000, 0))
    assert m is not None and abs(m - 10.0) < 0.01
    # 10 m / 1 s = 36 km/s
    v = calib.hiz_kmh(k, (0, 0), (1000, 0), dt_s=1.0)
    assert v is not None and abs(v - 36.0) < 0.1


def test_sifir_sure_hiz_uretmez():
    k = calib.kalibre_et(_cfg(5.0))
    assert calib.hiz_kmh(k, (0, 0), (500, 0), dt_s=0.0) is None


def test_eksik_dosya_cokmez_reddeder():
    k = calib.yukle(Path("config/cameras/olmayan_kamera.yaml"))
    assert k.gecerli is False
    assert any("bulunamadi" in n for n in k.notlar)


def test_saha_sablonu_asla_gecerli_sayilmaz():
    """Olculmemis sablon dosyasi kalibre gorunmemelidir - regresyon testi."""
    yol = Path("config/cameras/arnavutkoy_kavsak_01.yaml")
    if not yol.exists():
        return
    assert calib.yukle(yol).gecerli is False
