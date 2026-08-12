"""M3 kaza tespiti - modelsiz, deterministik.

En onemli testler POZITIF degil NEGATIF olanlar: okluzyonun, normal trafigin ve
kirmizi isikta duran araclarin kaza sayilmamasi. Plan geregi precision recall'dan
oncedir - yanlis kaza kaydi, kacirilan kazadan pahalidir.
"""

from __future__ import annotations

from adgs import accident
from adgs.accident import KazaParam
from adgs.schema import Detection, Track

FPS = 10.0


def _iz(tid: int, kareler: dict[int, tuple[int, int, int, int]], cls: str = "otomobil") -> Track:
    t = Track(track_id=tid, cls=cls)
    for f, bbox in kareler.items():
        t.frames[f] = Detection(bbox=bbox, cls=cls, conf=0.9)
    return t


def _kutu(x: int, y: int, b: int = 40) -> tuple[int, int, int, int]:
    return (x, y, x + b, y + b)


def _carpisan_cift(dur_kare: int = 60):
    """A hizla yaklasir, kare 20'de B'ye carpar, sonra durur."""
    a_kareler = {f: _kutu(100 - (20 - f) * 8, 100) for f in range(0, 20)}
    a_kareler.update({f: _kutu(100, 100) for f in range(20, dur_kare)})
    b_kareler = {f: _kutu(120, 100) for f in range(0, dur_kare)}
    return _iz(1, a_kareler), _iz(2, b_kareler)


# --- yardimci fonksiyonlar --------------------------------------------------


def test_iou_cakismayan_kutular_sifir():
    assert accident.iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_ayni_kutu_bir():
    assert accident.iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_yarim_cakisma():
    # 10x10 ve 10x10, 5 piksel kayik -> kesisim 50, birlesim 150
    assert abs(accident.iou((0, 0, 10, 10), (5, 0, 15, 10)) - 50 / 150) < 1e-6


def test_yon_farki_ters_yon_180():
    assert abs(accident.yon_farki_derece((1, 0), (-1, 0)) - 180.0) < 1e-6


def test_yon_farki_duran_nesne_sifir():
    # Duran nesnenin yonu tanimsizdir; 0 donmeli ki yanlis tetikleme olmasin.
    assert accident.yon_farki_derece((0, 0), (1, 0)) == 0.0


def test_hiz_olculemezse_none():
    # Tek kare varsa hiz yok - 0 DEGIL None donmeli.
    assert accident.hiz_piksel(_iz(1, {5: _kutu(0, 0)}), 0, 10) is None


# --- negatif senaryolar (asil deger burada) ---------------------------------


def test_okluzyon_kaza_sayilmaz():
    """Iki arac ust uste gecer ama ikisi de hizini korur -> kaza degil."""
    a = _iz(1, {f: _kutu(f * 8, 100) for f in range(0, 40)})
    b = _iz(2, {f: _kutu(f * 8 + 10, 100) for f in range(0, 40)})
    assert accident.tespit_et([a, b], FPS, "test.mp4") == []


def test_cakisma_yoksa_kaza_yok():
    a = _iz(1, {f: _kutu(0, 0) for f in range(40)})
    b = _iz(2, {f: _kutu(500, 500) for f in range(40)})
    assert accident.tespit_et([a, b], FPS, "test.mp4") == []


def test_kirmizi_isikta_duran_araclar_kaza_sayilmaz():
    """Yan yana duran iki arac: hareketsizlik var ama ani degisim yok."""
    a = _iz(1, {f: _kutu(100, 100) for f in range(40)})
    b = _iz(2, {f: _kutu(120, 100) for f in range(40)})
    assert accident.tespit_et([a, b], FPS, "test.mp4") == []


def test_carpisip_devam_eden_kaza_sayilmaz():
    """Cakisma + yavaslama var ama arac durmuyor -> sinyal 3 eksik."""
    a_kareler = {f: _kutu(100 - (20 - f) * 8, 100) for f in range(20)}
    a_kareler.update({f: _kutu(100 + (f - 20) * 2, 100) for f in range(20, 60)})
    b = _iz(2, {f: _kutu(120, 100) for f in range(60)})
    assert accident.tespit_et([_iz(1, a_kareler), b], FPS, "test.mp4") == []


def test_cok_kisa_izler_degerlendirilmez():
    a, b = _carpisan_cift()
    kisa_a = _iz(1, dict(list(a.frames.items())[:3]))
    assert accident.tespit_et([kisa_a, b], FPS, "test.mp4") == []


# --- pozitif senaryo --------------------------------------------------------


def test_gercek_carpisma_tespit_edilir():
    a, b = _carpisan_cift()
    olaylar = accident.tespit_et([a, b], FPS, "kavsak.mp4")
    assert len(olaylar) == 1
    e = olaylar[0]
    assert e.tip == "KAZA"
    assert e.alt_tip == "CARPISMA"
    assert {p.track_id for p in e.parties} == {1, 2}


def test_kaza_olayinda_kanit_var():
    e = accident.tespit_et(list(_carpisan_cift()), FPS, "kavsak.mp4")[0]
    assert e.kanitli_mi()
    assert e.evidence["keyframes"]
    assert e.evidence["kareler"]


def test_kaza_olayinda_on_degerlendirme_notu_var():
    from adgs.schema import ON_DEGERLENDIRME_NOTU

    e = accident.tespit_et(list(_carpisan_cift()), FPS, "kavsak.mp4")[0]
    assert ON_DEGERLENDIRME_NOTU in e.notes


def test_zaman_damgasi_fps_ile_hesaplanir():
    e = accident.tespit_et(list(_carpisan_cift()), FPS, "kavsak.mp4")[0]
    assert abs(e.t_start - e.frame_start / FPS) < 1e-9
    assert abs(e.t_end - e.frame_end / FPS) < 1e-9


# --- kalibrasyon filtresi ---------------------------------------------------


def _kalib_bir_piksel_bir_metre():
    from adgs import calib

    return calib.kalibre_et({
        "camera_id": "t",
        "homografi": {"piksel": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
                      "dunya": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]]},
        "dogrulama": [{"piksel": [[0, 0], [500, 0]], "gercek_m": 500.0}],
    })


def test_yer_duzleminde_uzak_cift_elenir():
    """Goruntude cakisan ama yerde onlarca metre uzakta olan cift okluzyondur."""
    k = _kalib_bir_piksel_bir_metre()
    assert k.gecerli
    a, b = _carpisan_cift()
    assert len(accident.tespit_et([a, b], FPS, "t.mp4")) == 1
    assert accident.tespit_et([a, b], FPS, "t.mp4", kalib=k) == []


def test_gecersiz_kalibrasyon_filtreyi_devre_disi_birakir():
    """Kalibrasyon gecersizse okluzyon filtresi uygulanmaz, tespit yine calisir."""
    from adgs.calib import Kalibrasyon

    a, b = _carpisan_cift()
    assert len(accident.tespit_et([a, b], FPS, "t.mp4", kalib=Kalibrasyon("t"))) == 1


def test_esikler_ayarlanabilir():
    a, b = _carpisan_cift()
    assert accident.tespit_et([a, b], FPS, "t.mp4", p=KazaParam(iou_esik=0.99)) == []
