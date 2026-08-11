"""TR plaka dogrulama - yanlis plaka uretmemenin testi.

Odak "dogru plakayi okuyabiliyor mu" degil, "yanlis plakayi reddediyor mu".
"""

from __future__ import annotations

import numpy as np

from adgs import plate

_GORUNTU = np.zeros((32, 128, 3), dtype=np.uint8)


def _sahte_ocr(metin: str, conf: float = 0.9):
    return lambda _img: (metin, conf)


def test_gecerli_formatlar_kabul_edilir():
    for p in ("34ABC123", "06A1234", "35AB123", "81ZZZ99", "01A12345"):
        assert plate.gecerli_mi(p), p


def test_gecersiz_formatlar_reddedilir():
    for p in ("", "ABC123", "99ABC123", "00A1234", "34ABCD123", "34A1", "3-4ABC123"):
        assert not plate.gecerli_mi(p), p


def test_harf_rakam_sayisi_uyumu_zorunlu():
    # 3 harf en fazla 3 rakam alir; 4 rakam gecersiz olmali.
    assert plate.gecerli_mi("34ABC123")
    assert not plate.gecerli_mi("34ABC1234")
    # 1 harf en az 4 rakam ister.
    assert not plate.gecerli_mi("34A123")


def test_normalize_bosluk_ve_tr_harf_temizler():
    assert plate.normalize(" 34 abc 123 ") == "34ABC123"
    assert plate.normalize("34-ŞBC-123") == "34SBC123"


def test_harf_bolgesindeki_rakam_harfe_donusur():
    assert plate.duzelt("340BC123") == "34OBC123"


def test_gecerli_plaka_duzeltmeyle_bozulmaz():
    assert plate.duzelt("34ABC123") == "34ABC123"
    assert plate.duzelt("06A1234") == "06A1234"


def test_ocr_yoksa_none_doner():
    # ANPR kapali - hata degil, yapilandirma durumu.
    assert plate.oku(_GORUNTU, ocr=None) == (None, 0.0)


def test_dusuk_guvenli_okuma_reddedilir():
    assert plate.oku(_GORUNTU, ocr=_sahte_ocr("34ABC123", conf=0.2)) == (None, 0.0)


def test_yapisal_kurali_gecemeyen_okuma_reddedilir():
    # OCR bir sey uretti ama plaka olamaz -> None. Sessizce kabul edilmemeli.
    assert plate.oku(_GORUNTU, ocr=_sahte_ocr("XQ7", conf=0.99)) == (None, 0.0)
    assert plate.oku(_GORUNTU, ocr=_sahte_ocr("99ZZZ99", conf=0.99)) == (None, 0.0)


def test_gecerli_okuma_dondurulur():
    p, c = plate.oku(_GORUNTU, ocr=_sahte_ocr("34 ABC 123", conf=0.88))
    assert p == "34ABC123"
    assert c == 0.88


def test_ocr_patlarsa_cokmez():
    def patlayan(_img):
        raise RuntimeError("ocr motoru coktu")

    assert plate.oku(_GORUNTU, ocr=patlayan) == (None, 0.0)


def test_bicimle_goruntuleme():
    assert plate.bicimle("34ABC123") == "34 ABC 123"
