"""KVKK bulaniklastirma (Faz 7).

Bayrak "calisiyor" demek yetmez - goruntunun GERCEKTEN degistigi olculur.
Bulaniklastirma sessizce devre disi kalirsa kisisel veri (plaka, yuz) ham
olarak arsive ve sunuma girer.
"""

from __future__ import annotations

import numpy as np

from adgs import render
from adgs.schema import Detection, Track


def _iz(cls: str, bbox=(100, 100, 200, 180), f: int = 0) -> Track:
    t = Track(track_id=1, cls=cls)
    t.frames[f] = Detection(bbox=bbox, cls=cls, conf=0.9)
    return t


def _gurultu(w: int = 320, h: int = 240):
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (h, w, 3), dtype=np.uint8)


# --- Bolge secimi -----------------------------------------------------------


def test_yaya_icin_bas_bolgesi():
    (x1, y1, x2, y2), = render.kvkk_bolgeleri(_iz("yaya"), 0)
    assert (x1, x2) == (100, 200)
    assert y1 == 100 and y2 < 180  # kutunun ust kismi


def test_arac_icin_plaka_bolgesi():
    (x1, y1, x2, y2), = render.kvkk_bolgeleri(_iz("otomobil"), 0)
    assert y2 == 180 and y1 > 100          # alt kisim
    assert x1 > 100 and x2 < 200           # orta bant


def test_bayrak_kapaliysa_bolge_uretilmez():
    assert render.kvkk_bolgeleri(_iz("otomobil"), 0, plaka=False) == []
    assert render.kvkk_bolgeleri(_iz("yaya"), 0, yuz=False) == []


def test_ilgisiz_sinif_bulaniklastirilmaz():
    assert render.kvkk_bolgeleri(_iz("trafik_isigi"), 0) == []


def test_karede_olmayan_iz_bolge_uretmez():
    assert render.kvkk_bolgeleri(_iz("otomobil"), 99) == []


def test_dejenere_kutu_cokme_yapmaz():
    assert render.kvkk_bolgeleri(_iz("otomobil", bbox=(10, 10, 10, 10)), 0) == []


# --- Gercekten bulaniyor mu -------------------------------------------------


def test_bolge_gercekten_bulaniklasiyor():
    img = _gurultu()
    once = img[120:170, 130:170].copy()
    render.bulaniklastir(img, [(120, 110, 180, 175)])
    sonra = img[120:170, 130:170]
    assert not np.array_equal(once, sonra)
    # "Degisti" yetmez - varyans dusmeli, yani gercekten BULANIKLASMALI.
    assert sonra.var() < once.var()


def test_bolge_disi_dokunulmuyor():
    img = _gurultu()
    disari = img[0:30, 0:30].copy()
    render.bulaniklastir(img, [(120, 110, 180, 175)])
    assert np.array_equal(disari, img[0:30, 0:30])


def test_kare_disina_tasan_bolge_kirpilir():
    img = _gurultu(80, 60)
    render.bulaniklastir(img, [(60, 40, 500, 500)])  # cokmemeli
    assert img.shape == (60, 80, 3)


def test_kvkk_kapaliysa_goruntu_degismez():
    img = _gurultu()
    kopya = img.copy()
    render._kvkk_uygula(img, [_iz("otomobil")], 0, None)
    assert np.array_equal(kopya, img)


def test_kvkk_acikken_goruntu_degisir():
    img = _gurultu()
    kopya = img.copy()
    render._kvkk_uygula(img, [_iz("otomobil")], 0, render.KVKK_VARSAYILAN)
    assert not np.array_equal(kopya, img)


def test_varsayilan_acik():
    """Video kisisel veri icerir; guvenli taraf bulaniklastirmaktir."""
    assert render.KVKK_VARSAYILAN == {"plaka": True, "yuz": True}
