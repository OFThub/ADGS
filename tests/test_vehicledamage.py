"""M8 arac hasar degerlendirmesi.

Bu modulun degeri kadar SINIRI da test edilir. CarDD yakin cekim verisidir;
CCTV'nin 60x40 piksellik arac kutusunda sinif uretmek uydurmaktir. Cozunurluk
kapisinin gercekten kapandigi burada kilitlenir.

Model, video ve GPU gerekmez: dedektor ve kare getirici enjekte edilir.
"""

from __future__ import annotations

import numpy as np

from adgs import vehicledamage as m8
from adgs.schema import Detection, Event, Party, Track

VIDEO = "yok.mp4"


def _kare(w: int = 640, h: int = 480):
    return np.zeros((h, w, 3), dtype=np.uint8)


def _iz(track_id: int, bbox_per_kare: dict[int, tuple[int, int, int, int]]) -> Track:
    t = Track(track_id=track_id, cls="otomobil")
    for f, b in bbox_per_kare.items():
        t.frames[f] = Detection(bbox=b, cls="otomobil", conf=0.9)
    return t


def _kaza(track_ids: list[int], keyframe: int = 10) -> Event:
    return Event(
        event_id="kaza_0001", tip="KAZA", alt_tip="CARPISMA",
        t_start=0.0, t_end=1.0, frame_start=keyframe - 5, frame_end=keyframe + 5,
        source_video=VIDEO, source_profile="cctv_fixed", conf=0.5,
        evidence={"keyframes": [keyframe]},
        parties=[Party(track_id=t) for t in track_ids],
    )


def _sabit_dedektor(tespitler):
    return lambda crop: list(tespitler)


# --- Siddet -----------------------------------------------------------------


def test_kucuk_cizik_hafif():
    assert m8.siddet(["scratch"], 0.01) == "HAFIF"


def test_genis_hasar_agir():
    assert m8.siddet(["scratch"], 0.20) == "AGIR"


def test_orta_alan_orta():
    assert m8.siddet(["scratch"], 0.07) == "ORTA"


def test_yapisal_hasar_kucuk_olsa_da_orta():
    """Bir cam kirigi, genis bir cizikten daha agir bir sonuctur."""
    for t in ("crack", "glass shatter", "tire flat"):
        assert m8.siddet([t], 0.001) == "ORTA", t


def test_yapisal_hasar_agir_esigini_gecebilir():
    assert m8.siddet(["crack"], 0.30) == "AGIR"


# --- Bolge ------------------------------------------------------------------


def test_hareket_yonundeki_hasar_on():
    # Arac saga gidiyor; hasar merkezin saginda -> on
    assert m8.bolge((150.0, 100.0), (50, 80, 150, 120), (1.0, 0.0)) == "on"


def test_hareket_yonunun_tersindeki_hasar_arka():
    assert m8.bolge((50.0, 100.0), (50, 80, 150, 120), (1.0, 0.0)) == "arka"


def test_yanal_hasar_yan():
    assert m8.bolge((100.0, 130.0), (50, 80, 150, 120), (1.0, 0.0)) == "yan"


def test_yon_bilinmiyorsa_bolge_uretilmez():
    """Duran aracin onu neresi belli degildir; "on" VARSAYILMAZ."""
    assert m8.bolge((150.0, 100.0), (50, 80, 150, 120), None) is None


def test_duran_iz_icin_yon_none():
    iz = _iz(1, {f: (100, 100, 200, 180) for f in range(20)})
    assert m8._yon(iz, 10) is None


def test_hareketli_iz_icin_yon_bulunur():
    iz = _iz(1, {f: (100 + f * 10, 100, 200 + f * 10, 180) for f in range(20)})
    yon = m8._yon(iz, 10)
    assert yon is not None
    assert yon[0] > 0.99  # saga


# --- Cozunurluk kapisi (modulun en kritik davranisi) ------------------------


def test_dusuk_cozunurlukte_sinif_URETILMEZ():
    """CCTV'nin tipik arac kutusu. Model bir sinif uretirdi; biz uretmiyoruz."""
    iz = _iz(1, {10: (100, 100, 160, 140)})  # 60x40 piksel
    evt = _kaza([1])
    uyarilar = m8.degerlendir(
        VIDEO, [evt], [iz],
        dedektor=_sabit_dedektor([("scratch", (5, 5, 20, 20), 0.9)]),
        kare_getir=lambda f: _kare(),
    )
    h = evt.parties[0].hasar
    assert h["guvenilir"] is False
    assert h["tipler"] == []
    assert h["siddet"] is None
    # Esik sabite baglanir - olcumle degisince test bayatlamasin.
    assert str(m8.MIN_KENAR_PIKSEL) in h["sebep"]
    assert any("URETILMEDI" in u for u in uyarilar)


def test_yeterli_cozunurlukte_sinif_uretilir():
    iz = _iz(1, {10: (100, 100, 400, 350), 12: (110, 100, 410, 350)})
    evt = _kaza([1])
    m8.degerlendir(
        VIDEO, [evt], [iz],
        dedektor=_sabit_dedektor([("dent", (20, 20, 60, 60), 0.9)]),
        kare_getir=lambda f: _kare(),
    )
    h = evt.parties[0].hasar
    assert h["guvenilir"] is True
    assert h["tipler"] == ["dent"]
    assert h["etiketler"] == ["Gocuk"]
    assert h["siddet"] in ("HAFIF", "ORTA", "AGIR")


def test_esik_parametreyle_gevsetilebilir():
    """Yakin cekim/olay yeri fotografi beslendiginde kapinin acilmasi gerekir."""
    iz = _iz(1, {10: (100, 100, 160, 140)})
    evt = _kaza([1])
    m8.degerlendir(
        VIDEO, [evt], [iz], min_kenar=32,
        dedektor=_sabit_dedektor([("crack", (5, 5, 20, 20), 0.9)]),
        kare_getir=lambda f: _kare(),
    )
    assert evt.parties[0].hasar["guvenilir"] is True


# --- Tespit yok / model yok / kapsam ----------------------------------------


def test_hasar_bulunamazsa_bos_ama_guvenilir():
    """"Hasar yok" ile "bakamadim" ayni sey degil - ikisi ayri isaretlenir."""
    iz = _iz(1, {10: (100, 100, 400, 350)})
    evt = _kaza([1])
    m8.degerlendir(VIDEO, [evt], [iz], dedektor=_sabit_dedektor([]),
                   kare_getir=lambda f: _kare())
    h = evt.parties[0].hasar
    assert h["guvenilir"] is True
    assert h["tipler"] == []
    assert "tespit edilmedi" in h["sebep"]


def test_model_yoksa_sessizce_gecilmez():
    iz = _iz(1, {10: (100, 100, 400, 350)})
    evt = _kaza([1])
    uyarilar = m8.degerlendir(VIDEO, [evt], [iz], model_path="yok/model.pt")
    assert evt.parties[0].hasar is None
    assert any("model bulunamadi" in u for u in uyarilar)


def test_kaza_disi_olaylar_dokunulmaz():
    evt = _kaza([1])
    evt.tip = "IHLAL"
    iz = _iz(1, {10: (100, 100, 400, 350)})
    assert m8.degerlendir(VIDEO, [evt], [iz],
                          dedektor=_sabit_dedektor([("dent", (1, 1, 5, 5), 0.9)]),
                          kare_getir=lambda f: _kare()) == []
    assert evt.parties[0].hasar is None


def test_iz_keyframede_yoksa_sebep_yazilir():
    iz = _iz(1, {50: (100, 100, 400, 350)})  # keyframe 10 yok
    evt = _kaza([1])
    m8.degerlendir(VIDEO, [evt], [iz], dedektor=_sabit_dedektor([]),
                   kare_getir=lambda f: _kare())
    assert evt.parties[0].hasar["guvenilir"] is False
    assert "bulunamadi" in evt.parties[0].hasar["sebep"]


def test_kare_okunamazsa_uyari():
    iz = _iz(1, {10: (100, 100, 400, 350)})
    evt = _kaza([1])
    uyarilar = m8.degerlendir(VIDEO, [evt], [iz], dedektor=_sabit_dedektor([]),
                              kare_getir=lambda f: None)
    assert any("okunamadi" in u for u in uyarilar)


# --- Toplama mantigi --------------------------------------------------------


def test_en_agir_tip_bolgeyi_belirler():
    """Cizik ve catlak birlikteyse bolge catlaga gore secilir."""
    iz = _iz(1, {10: (100, 100, 400, 350), 12: (110, 100, 410, 350)})
    evt = _kaza([1])
    m8.degerlendir(
        VIDEO, [evt], [iz],
        dedektor=_sabit_dedektor([
            ("scratch", (0, 0, 20, 20), 0.9),        # arka tarafta
            ("crack", (270, 100, 299, 140), 0.9),    # on tarafta
        ]),
        kare_getir=lambda f: _kare(),
    )
    h = evt.parties[0].hasar
    assert h["tipler"][0] == "crack"  # agirliga gore sirali
    assert h["bolge"] == "on"


def test_kirpma_kare_disina_tasarsa_cokmez():
    """Takip kutusu kare sinirini asabilir; numpy dilimi bos dizi uretirdi."""
    iz = _iz(1, {10: (500, 380, 900, 700)})  # 640x480 kareyi asiyor
    evt = _kaza([1])
    m8.degerlendir(VIDEO, [evt], [iz], min_kenar=32,
                   dedektor=_sabit_dedektor([("dent", (1, 1, 10, 10), 0.9)]),
                   kare_getir=lambda f: _kare())
    assert evt.parties[0].hasar is not None


def test_hasar_notu_eksper_sinirini_soyler():
    iz = _iz(1, {10: (100, 100, 400, 350)})
    evt = _kaza([1])
    m8.degerlendir(VIDEO, [evt], [iz],
                   dedektor=_sabit_dedektor([("dent", (20, 20, 60, 60), 0.9)]),
                   kare_getir=lambda f: _kare())
    assert any("eksper tespiti yerine" in n for n in evt.notes)


def test_alan_orani_biri_gecemez():
    iz = _iz(1, {10: (100, 100, 400, 350)})
    evt = _kaza([1])
    m8.degerlendir(
        VIDEO, [evt], [iz],
        dedektor=_sabit_dedektor([("dent", (0, 0, 200, 150), 0.9)] * 5),
        kare_getir=lambda f: _kare(),
    )
    assert evt.parties[0].hasar["alan_orani"] <= 1.0
