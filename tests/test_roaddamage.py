"""M5 siddet/oncelik mantigi - modelsiz, deterministik."""

from pathlib import Path

import pytest

from adgs.roaddamage import RDD_CLASSES, detect, oncelik, siddet


def test_buyuk_alan_her_zaman_yuksek():
    for c in RDD_CLASSES:
        assert siddet(c, 0.10) == "YUKSEK"


def test_yapisal_hasar_kucuk_olsa_da_orta():
    # Cukur ve timsah sirti catlak yapisal bozulma gostergesi - DUSUK'e dusemez.
    assert siddet("D40", 0.0001) == "ORTA"
    assert siddet("D20", 0.0001) == "ORTA"


def test_yuzeysel_catlak_kucukse_dusuk():
    assert siddet("D00", 0.0001) == "DUSUK"
    assert siddet("D10", 0.0001) == "DUSUK"


def test_alan_esigi_sinifi_yukseltir():
    assert siddet("D00", 0.02) == "ORTA"


def test_oncelik_cukuru_her_zaman_acil_isaretler():
    assert oncelik("DUSUK", "D40") == 1
    assert oncelik("YUKSEK", "D00") == 1
    assert oncelik("ORTA", "D00") == 2
    assert oncelik("DUSUK", "D00") == 3


def test_yanlis_model_reddedilir():
    """COCO agirligi verilirse sessizce yanlis etiketlemek yerine hata verir."""
    coco = Path("yolo26s.pt")
    if not coco.exists():
        pytest.skip("yolo26s.pt yok")
    with pytest.raises(ValueError, match="yol hasari modeli degil"):
        detect("olmayan_video.mp4", coco)
