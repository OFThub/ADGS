"""M2 tespit/takip yardimcilari - modelsiz, deterministik."""

from __future__ import annotations

from adgs import detect
from adgs.schema import Detection, Track


def _iz(tid: int, cls: str, n_kare: int) -> Track:
    t = Track(track_id=tid, cls=cls)
    for f in range(n_kare):
        t.frames[f] = Detection(bbox=(10, 20, 30, 40), cls=cls, conf=0.8)
    return t


def test_zemin_noktasi_kutunun_alt_ortasi():
    # Arac yer duzlemine tekerlekleriyle degdigi icin homografi alt orta
    # noktaya uygulanmalidir; merkez kullanilirsa arac boyu kadar hata girer.
    assert detect._zemin_noktasi((100, 200, 180, 260)) == (140.0, 260.0)


def test_zemin_noktasi_merkezden_farklidir():
    bbox = (0, 0, 100, 400)
    zemin = detect._zemin_noktasi(bbox)
    merkez = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    assert zemin != merkez
    assert zemin[1] > merkez[1]


def test_kisa_izler_elenir():
    izler = [_iz(1, "otomobil", 10), _iz(2, "yaya", 2), _iz(3, "otomobil", 3)]
    kalan = detect.kisa_izleri_ele(izler, min_kare=3)
    assert [t.track_id for t in kalan] == [1, 3]


def test_esik_kadar_kare_kalir():
    # min_kare tam esik degeri kabul edilmeli (>=), kaybedilmemeli.
    assert len(detect.kisa_izleri_ele([_iz(1, "otomobil", 3)], min_kare=3)) == 1
    assert len(detect.kisa_izleri_ele([_iz(1, "otomobil", 2)], min_kare=3)) == 0


def test_ozet_sinif_basina_sayar():
    izler = [_iz(1, "otomobil", 5), _iz(2, "otomobil", 5), _iz(3, "yaya", 5)]
    assert detect.ozet(izler) == {"otomobil": 2, "yaya": 1}


def test_coco_eslemesi_trafik_siniflarini_kapsar():
    for coco in ("car", "truck", "bus", "motorcycle", "bicycle", "person"):
        assert coco in detect.COCO_ESLEME
    # COCO'nun trafikle ilgisiz siniflari eslemede olmamali.
    assert "banana" not in detect.COCO_ESLEME


def test_zemin_siniflari_sadece_hareketli_nesneler():
    # Trafik isigi/tabela zemine degmez; dunya koordinati uretilmemeli.
    assert "trafik_isigi" not in detect._ZEMINDE
    assert "trafik_isareti" not in detect._ZEMINDE
    assert "otomobil" in detect._ZEMINDE
