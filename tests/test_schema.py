"""Event sozlesmesinin hukuki garantilerini test eder.

Bu testler CV modeline bagli degildir; sistemin hukuken en hassas davranisi
olan zorunlu on-degerlendirme notunun her KAZA/IHLAL ciktisinda bulunmasini
garanti eder.
"""

from adgs.schema import ON_DEGERLENDIRME_NOTU, Event, Party, Violation


def _event(tip: str, **kw) -> Event:
    return Event(
        event_id="t1",
        tip=tip,
        alt_tip="TEST",
        t_start=0.0,
        t_end=1.0,
        frame_start=0,
        frame_end=30,
        source_video="test.mp4",
        source_profile="cctv_fixed",
        conf=0.9,
        **kw,
    )


def test_kaza_ve_ihlal_zorunlu_not_tasir():
    for tip in ("KAZA", "IHLAL"):
        assert ON_DEGERLENDIRME_NOTU in _event(tip).notes, tip


def test_altyapi_kusur_notu_tasimaz():
    # Yol hasari belediyenin kendi gorev alani; kusur/ceza uyarisi anlamsiz.
    assert ON_DEGERLENDIRME_NOTU not in _event("ALTYAPI").notes


def test_not_iki_kez_eklenmez():
    evt = _event("KAZA", notes=[ON_DEGERLENDIRME_NOTU])
    assert evt.notes.count(ON_DEGERLENDIRME_NOTU) == 1


def test_mevcut_notlar_korunur():
    evt = _event("IHLAL", notes=["kalibrasyon dogrulanmadi"])
    assert "kalibrasyon dogrulanmadi" in evt.notes
    assert ON_DEGERLENDIRME_NOTU in evt.notes


def test_kanitsiz_event_isaretlenir():
    assert not _event("KAZA").kanitli_mi()
    assert _event("KAZA", evidence={"keyframes": [12]}).kanitli_mi()
    assert _event("ALTYAPI", evidence={"clip": "c.mp4"}).kanitli_mi()


def test_hukuki_alanlar_varsayilan_bos():
    # M6/M7 doldurana kadar None kalmali - uydurma deger uretilmemeli.
    v = Violation(ihlal_kodu="KIRMIZI_ISIK", conf=0.8)
    assert (v.ktk_madde, v.kusur_sinifi, v.ceza_tutari_try, v.ceza_puani) == (None,) * 4
    assert v.ceza_tablosu_tarihi is None
    assert Party(track_id=1).violations == []
