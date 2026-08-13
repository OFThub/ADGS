"""M10 depolama katmani.

Iki sey ozellikle kilitlenir:
    1. KVKK saklama suresi - kayit SILINIRKEN klip dosyasi da silinmeli.
       Sadece satiri silmek diskte kisisel veri birakir.
    2. Kanitsiz olay veritabanina YAZILMAZ (rapor kuraliyla ayni).
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from adgs import store
from adgs.schema import Event, Party, Violation


def _evt(event_id: str = "kaza_0001", tip: str = "KAZA", kanit: bool = True,
         klip: str | None = None, ihlal: str | None = None) -> Event:
    evidence: dict = {}
    if kanit:
        evidence["keyframes"] = [10]
    if klip:
        evidence["clip"] = klip
    ihlaller = []
    if ihlal:
        ihlaller = [Violation(ihlal_kodu=ihlal, conf=0.8, ktk_madde="84/a",
                              kusur_sinifi="ASLI", ceza_tutari_try=111,
                              ceza_puani=5, ceza_tablosu_tarihi="2026-02-27")]
    return Event(
        event_id=event_id, tip=tip, alt_tip="CARPISMA",
        t_start=0.0, t_end=2.0, frame_start=5, frame_end=55,
        source_video="k.mp4", source_profile="cctv_fixed", conf=0.44,
        evidence=evidence, gps=(41.18, 28.74),
        parties=[Party(track_id=7, plate="34ABC123", violations=ihlaller,
                       hasar={"siddet": "ORTA", "tipler": ["dent"]})],
    )


@pytest.fixture
def conn(tmp_path: Path):
    c = store.baglan(tmp_path / "t.db")
    yield c
    c.close()


# --- Yazma / okuma ----------------------------------------------------------


def test_video_ve_olay_kaydedilir(conn):
    vid = store.video_kaydet(conn, "k.mp4", camera_id="kavsak_01",
                             cekim_tarihi="2026-07-15")
    assert store.olaylari_kaydet(conn, vid, [_evt(ihlal="KIRMIZI_ISIK")]) == 1
    olaylar = store.olaylari_getir(conn)
    assert len(olaylar) == 1
    o = olaylar[0]
    assert o["event_id"] == "kaza_0001"
    assert o["camera_id"] == "kavsak_01"
    assert o["gps"] == [41.18, 28.74]


def test_taraf_ve_ihlal_alanlari_korunur(conn):
    vid = store.video_kaydet(conn, "k.mp4")
    store.olaylari_kaydet(conn, vid, [_evt(ihlal="KIRMIZI_ISIK")])
    taraf = store.olay_getir(conn, "kaza_0001")["parties"][0]
    assert taraf["track_id"] == 7
    assert taraf["plaka"] == "34ABC123"
    assert taraf["hasar"]["siddet"] == "ORTA"
    v = taraf["violations"][0]
    assert v["ihlal_kodu"] == "KIRMIZI_ISIK"
    assert v["ktk_madde"] == "84/a"
    assert v["kusur_sinifi"] == "ASLI"
    assert v["ceza_tutari_try"] == 111
    assert v["ceza_tablosu_tarihi"] == "2026-02-27"


def test_notlar_korunur(conn):
    from adgs.schema import ON_DEGERLENDIRME_NOTU

    vid = store.video_kaydet(conn, "k.mp4")
    store.olaylari_kaydet(conn, vid, [_evt()])
    assert ON_DEGERLENDIRME_NOTU in store.olay_getir(conn, "kaza_0001")["notlar"]


def test_kanitsiz_olay_yazilmaz(conn):
    """Rapor kuraliyla ayni: kaniti olmayan olay kayda girmez."""
    vid = store.video_kaydet(conn, "k.mp4")
    assert store.olaylari_kaydet(conn, vid, [_evt(kanit=False)]) == 0
    assert store.olaylari_getir(conn) == []


def test_bilinmeyen_olay_none(conn):
    assert store.olay_getir(conn, "yok") is None


# --- Filtreleme -------------------------------------------------------------


def test_tipe_gore_filtre(conn):
    vid = store.video_kaydet(conn, "k.mp4")
    store.olaylari_kaydet(conn, vid, [
        _evt("kaza_0001", "KAZA"), _evt("ihlal_0001", "IHLAL"),
        _evt("altyapi_0001", "ALTYAPI"),
    ])
    assert len(store.olaylari_getir(conn, tip="IHLAL")) == 1
    assert len(store.olaylari_getir(conn)) == 3


def test_tarihe_ve_kameraya_gore_filtre(conn):
    v1 = store.video_kaydet(conn, "a.mp4", camera_id="k1", cekim_tarihi="2026-07-15")
    v2 = store.video_kaydet(conn, "b.mp4", camera_id="k2", cekim_tarihi="2026-07-16")
    store.olaylari_kaydet(conn, v1, [_evt("e1")])
    store.olaylari_kaydet(conn, v2, [_evt("e2")])
    assert len(store.olaylari_getir(conn, tarih="2026-07-15")) == 1
    assert len(store.olaylari_getir(conn, camera_id="k2")) == 1
    assert len(store.olaylari_getir(conn, video_id=v1)) == 1


def test_ayni_video_ayni_olay_iki_kez_yazilmaz(conn):
    vid = store.video_kaydet(conn, "k.mp4")
    store.olaylari_kaydet(conn, vid, [_evt()])
    store.olaylari_kaydet(conn, vid, [_evt()])
    assert len(store.olaylari_getir(conn)) == 1


# --- Durum ------------------------------------------------------------------


def test_video_durumu_guncellenir(conn):
    vid = store.video_kaydet(conn, "k.mp4")
    assert store.video_getir(conn, vid)["durum"] == "BEKLIYOR"
    store.durum_guncelle(conn, vid, "HATA", "model yok")
    d = store.video_getir(conn, vid)
    assert d["durum"] == "HATA" and d["hata"] == "model yok"


def test_video_olay_sayisi(conn):
    vid = store.video_kaydet(conn, "k.mp4")
    store.olaylari_kaydet(conn, vid, [_evt("e1"), _evt("e2")])
    assert store.video_getir(conn, vid)["olay_sayisi"] == 2


# --- KVKK: saklama suresi ---------------------------------------------------


def _eskit(conn, video_id: int, gun: int) -> None:
    eski = (datetime.now(timezone.utc) - timedelta(days=gun)).isoformat(timespec="seconds")
    conn.execute("UPDATE videos SET islenme_zamani = ? WHERE id = ?", (eski, video_id))
    conn.commit()


def test_suresi_dolan_kayit_silinir(conn):
    vid = store.video_kaydet(conn, "k.mp4")
    store.olaylari_kaydet(conn, vid, [_evt()])
    _eskit(conn, vid, 40)
    assert store.saklama_uygula(conn, gun=30)["silinen_video"] == 1
    assert store.olaylari_getir(conn) == []


def test_suresi_dolmayan_kayit_kalir(conn):
    vid = store.video_kaydet(conn, "k.mp4")
    store.olaylari_kaydet(conn, vid, [_evt()])
    _eskit(conn, vid, 10)
    assert store.saklama_uygula(conn, gun=30)["silinen_video"] == 0
    assert len(store.olaylari_getir(conn)) == 1


def test_klip_dosyasi_da_silinir(conn, tmp_path: Path):
    """Sadece satiri silmek diskte kisisel veri (plaka, yuz) birakirdi."""
    klip = tmp_path / "kaza_0001.mp4"
    klip.write_bytes(b"x")
    vid = store.video_kaydet(conn, "k.mp4")
    store.olaylari_kaydet(conn, vid, [_evt(klip=str(klip))])
    _eskit(conn, vid, 40)
    assert store.saklama_uygula(conn, gun=30)["silinen_klip"] == 1
    assert not klip.exists()


def test_cascade_gercekten_calisiyor(conn):
    """SQLite'ta foreign_keys PRAGMA'si varsayilan KAPALI - acilmazsa taraf ve
    ihlal satirlari oksuz kalir ve KVKK imhasi eksik olur."""
    vid = store.video_kaydet(conn, "k.mp4")
    store.olaylari_kaydet(conn, vid, [_evt(ihlal="KIRMIZI_ISIK")])
    _eskit(conn, vid, 40)
    store.saklama_uygula(conn, gun=30)
    for tablo in ("events", "parties", "violations"):
        assert conn.execute(f"SELECT COUNT(*) FROM {tablo}").fetchone()[0] == 0, tablo


def test_sifir_veya_negatif_saklama_reddedilir(conn):
    with pytest.raises(ValueError):
        store.saklama_uygula(conn, gun=0)


def test_eksik_klip_dosyasi_cokme_yapmaz(conn):
    vid = store.video_kaydet(conn, "k.mp4")
    store.olaylari_kaydet(conn, vid, [_evt(klip="yok/olan/klip.mp4")])
    _eskit(conn, vid, 40)
    assert store.saklama_uygula(conn, gun=30)["silinen_klip"] == 0
