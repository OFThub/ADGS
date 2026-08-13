"""Faz 7 - API ve arayuz.

Video isleme burada test EDILMEZ (GPU ve dakikalar surer); test edilen sey
sozlesme: uclar dogru sekli donuyor mu, filtre calisiyor mu, olmayan kayit 404
mu, KVKK imhasi ucdan tetiklenebiliyor mu ve zorunlu uyari her listede var mi.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from adgs import api, store
from adgs.schema import Event, Party, Violation


def _evt(event_id: str = "kaza_0001", tip: str = "KAZA") -> Event:
    return Event(
        event_id=event_id, tip=tip, alt_tip="CARPISMA",
        t_start=0.0, t_end=2.0, frame_start=5, frame_end=55,
        source_video="k.mp4", source_profile="cctv_fixed", conf=0.44,
        evidence={"keyframes": [10]}, gps=(41.18, 28.74),
        parties=[Party(track_id=7, plate="34ABC123", violations=[
            Violation(ihlal_kodu="KIRMIZI_ISIK", conf=0.8, ktk_madde="84/a",
                      kusur_sinifi="ASLI", ceza_tutari_try=111,
                      ceza_tablosu_tarihi="2026-02-27")])],
    )


@pytest.fixture
def istemci(tmp_path: Path, monkeypatch):
    db = tmp_path / "t.db"
    monkeypatch.setattr(store, "VARSAYILAN_DB", db)
    monkeypatch.setattr(api, "CIKTI_DIZINI", tmp_path / "runs")
    monkeypatch.setattr(api, "YUKLEME_DIZINI", tmp_path / "uploads")
    conn = store.baglan(db)
    vid = store.video_kaydet(conn, "k.mp4", camera_id="kavsak_01",
                             cekim_tarihi="2026-07-15", durum="TAMAM")
    store.olaylari_kaydet(conn, vid, [_evt("kaza_0001", "KAZA"),
                                      _evt("altyapi_0001", "ALTYAPI")])
    conn.close()
    with TestClient(api.app) as c:
        yield c


# --- Arayuz -----------------------------------------------------------------


def test_arayuz_aciliyor(istemci):
    r = istemci.get("/")
    assert r.status_code == 200
    assert "ADGS" in r.text
    # Konumlandirma arayuzde de gorunmeli - cikti baglamindan koparak dolasir.
    assert "bağlayıcı bir tespit üretmez" in r.text


def test_arayuz_harici_kaynak_kullanmiyor(istemci):
    """Belediye aginda dis bagimlilik istemiyoruz.

    "cdn" kelimesini aramak yetmez - kendi yorumumuzda da geciyor. Aranan sey
    gercek referans: protokollu src/href, @import ve protokolsuz //host.
    """
    import re

    metin = istemci.get("/").text
    assert not re.search(r'(?:src|href)\s*=\s*["\'](?:https?:)?//', metin, re.I)
    assert "@import" not in metin


# --- Olaylar ----------------------------------------------------------------


def test_olay_listesi(istemci):
    d = istemci.get("/events").json()
    assert d["sayi"] == 2
    assert {e["event_id"] for e in d["events"]} == {"kaza_0001", "altyapi_0001"}


def test_listede_zorunlu_uyari_var(istemci):
    assert "baglayici bir tespit degildir" in istemci.get("/events").json()["uyari"]


def test_tipe_gore_filtre(istemci):
    assert istemci.get("/events?tip=ALTYAPI").json()["sayi"] == 1


def test_tarihe_ve_kameraya_gore_filtre(istemci):
    assert istemci.get("/events?tarih=2026-07-15").json()["sayi"] == 2
    assert istemci.get("/events?tarih=2020-01-01").json()["sayi"] == 0
    assert istemci.get("/events?camera_id=kavsak_01").json()["sayi"] == 2


def test_tek_olay(istemci):
    d = istemci.get("/events/kaza_0001").json()
    assert d["parties"][0]["violations"][0]["kusur_sinifi"] == "ASLI"
    assert d["parties"][0]["plaka"] == "34ABC123"


def test_olmayan_olay_404(istemci):
    assert istemci.get("/events/yok").status_code == 404


# --- Videolar ---------------------------------------------------------------


def test_video_listesi_ve_durumu(istemci):
    v = istemci.get("/videos").json()["videolar"]
    assert len(v) == 1
    d = istemci.get(f"/videos/{v[0]['id']}/status").json()
    assert d["durum"] == "TAMAM"
    assert d["olay_sayisi"] == 2


def test_olmayan_video_404(istemci):
    assert istemci.get("/videos/9999/status").status_code == 404


def test_video_yukleme_kaydi_hemen_olusur(istemci, monkeypatch):
    """Analiz dakikalar surer; istek bekletilmez, durum BEKLIYOR doner."""
    cagrildi = {}
    monkeypatch.setattr(api, "_isle", lambda *a, **k: cagrildi.setdefault("evet", True))
    r = istemci.post("/videos", files={"file": ("a.mp4", b"0123", "video/mp4")})
    assert r.status_code == 200
    d = r.json()
    assert d["durum"] == "BEKLIYOR" and d["video_id"] > 0
    assert cagrildi.get("evet") is True


# --- Tek girdi videodur -----------------------------------------------------


def test_yukleme_sadece_dosya_ister(istemci, monkeypatch):
    """profile/detect/camera/tarih ARTIK SORULMUYOR - videodan turetiliyor."""
    monkeypatch.setattr(api, "_isle", lambda *a, **k: None)
    r = istemci.post("/videos", files={"file": ("a.mp4", b"0", "video/mp4")})
    assert r.status_code == 200
    t = r.json()["turetilen"]
    assert set(t) >= {"profile", "detect", "camera", "tarih", "gerekce"}
    # Her turetilen degerin bir gerekcesi olmali - sessiz varsayilan yok.
    assert len(t["gerekce"]) == 3


def test_turetilen_degerler_veritabanina_yaziliyor(istemci, monkeypatch, tmp_path: Path):
    """Turetim sonucu kayda gecmezse arayuz yanlis kamera/tarih gosterir."""
    monkeypatch.setattr(api, "_isle", lambda *a, **k: None)
    r = istemci.post(
        "/videos",
        files={"file": ("demo_sentetik_2026-07-15.mp4", b"0", "video/mp4")})
    vid = r.json()["video_id"]
    d = istemci.get(f"/videos/{vid}/status").json()
    assert d["camera_id"] == "demo_sentetik"
    assert d["cekim_tarihi"] == "2026-07-15"


@pytest.mark.parametrize("alan,kotu", [
    ("tarih", '"><script>alert(1)</script>'),
    ("camera", "<svg onload=alert(1)>"),
    ("profile", "<img src=x onerror=alert(1)>"),
    ("detect", "; rm -rf /"),
])
def test_istemcinin_gonderdigi_alanlar_yok_sayiliyor(istemci, monkeypatch, tmp_path,
                                                     alan, kotu):
    """Bu alanlar artik uctan OKUNMUYOR; gonderilse bile kayda gecmemeli.

    Eskiden 400 donuyorlardi (dogrulama). Simdi hic okunmadiklari icin sessizce
    yok sayilirlar - test ettigimiz sey degerin VERITABANINA SIZMADIGI.
    """
    monkeypatch.setattr(api, "_isle", lambda *a, **k: None)
    r = istemci.post("/videos", files={"file": ("a.mp4", b"0", "video/mp4")},
                     data={alan: kotu})
    assert r.status_code == 200
    d = istemci.get(f"/videos/{r.json()['video_id']}/status").json()
    assert kotu not in str(d)


# --- Is emri ----------------------------------------------------------------


def test_is_emri_pdf_uretiliyor(istemci):
    r = istemci.get("/events/altyapi_0001/is-emri.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"


def test_olmayan_olay_icin_is_emri_404(istemci):
    assert istemci.get("/events/yok/is-emri.pdf").status_code == 404


# --- KVKK -------------------------------------------------------------------


def test_kvkk_imha_ucu(istemci, tmp_path: Path):
    conn = store.baglan(tmp_path / "t.db")
    eski = (datetime.now(timezone.utc) - timedelta(days=99)).isoformat(timespec="seconds")
    conn.execute("UPDATE videos SET islenme_zamani = ?", (eski,))
    conn.commit()
    conn.close()
    d = istemci.post("/kvkk/purge?gun=30").json()
    assert d["silinen_video"] == 1
    assert istemci.get("/events").json()["sayi"] == 0


def test_gecersiz_saklama_suresi_400(istemci):
    assert istemci.post("/kvkk/purge?gun=0").status_code == 400


# --- Guvenlik: stored XSS ---------------------------------------------------


def test_arayuz_sunucu_verisini_innerhtml_ile_basmiyor(istemci):
    """tarih/kamera kullanicidan gelir, kaydedilir ve herkese gosterilir.

    Sablon dizesiyle innerHTML'e basmak stored XSS'tir; degerler dugum olarak
    olusturulup textContent ile yazilmali.
    """
    js = istemci.get("/").text
    # Sunucu verisi iceren sablon dizesi innerHTML'e atanmamali.
    assert "innerHTML = `" not in js
    assert ".innerHTML =" not in js.replace('tb.innerHTML = "";', "")
    assert "textContent" in js
    assert "encodeURIComponent" in js


@pytest.mark.parametrize("kotu", [
    '<img src=x onerror=alert(1)>.mp4',
    '"><script>alert(1)</script>.mp4',
])
def test_zararli_dosya_adi_kamera_olarak_kaydedilmiyor(istemci, monkeypatch, kotu):
    """Dosya adi tek kullanici girdisi kaldi - turetim oradan okuyor.

    Kamera kimligi yalnizca config/cameras altindaki TANIMLI adlarla eslesir;
    dosya adindaki rastgele metin kamera olarak kaydedilemez.
    """
    monkeypatch.setattr(api, "_isle", lambda *a, **k: None)
    r = istemci.post("/videos", files={"file": (kotu, b"0", "video/mp4")})
    assert r.status_code == 200
    assert r.json()["turetilen"]["camera"] is None


def test_yol_hasari_modeli_yoksa_kosu_iptal_olmuyor(istemci, monkeypatch, tmp_path):
    """Eksik ek dedektor yuzunden kaza/ihlal analizi de kaybedilmemeli."""
    cagri = {}
    monkeypatch.setattr("adgs.cli.run",
                        lambda *a, **k: cagri.update(k) or 1)
    monkeypatch.setattr(store, "durum_guncelle", lambda *a, **k: None)
    monkeypatch.setattr(Path, "exists", lambda self: "best.pt" not in str(self))
    api._isle(1, tmp_path / "a.mp4", "cctv_fixed",
              "roaddamage,accident,ihlal", None, None)
    assert "roaddamage" not in cagri["detect_tipleri"]
    assert "accident" in cagri["detect_tipleri"]


@pytest.mark.parametrize("ad,beklenen", [
    ("../../kacti.mp4", "kacti.mp4"),               # yol gezinmesi
    ('<img src=x>.mp4', "_img_src_x_.mp4"),         # Windows'ta yasak karakterler
    ("20260715_kavsak.mp4", "20260715_kavsak.mp4"),  # turetimin okudugu ad bozulmaz
    ("", "video.mp4"),                              # bos ad
    ("....", "video.mp4"),                          # sadece nokta
])
def test_dosya_adi_diske_yazilabilir_hale_getiriliyor(ad, beklenen):
    """Dosya adi TEK kullanici girdisi; diske yazilamayan ad 500 uretirdi."""
    assert api._guvenli_ad(ad) == beklenen


# --- Faz 8: video, klip, kara nokta, kameralar ------------------------------


def test_isaretli_video_sunuluyor(istemci, tmp_path: Path):
    """Sunumun asil gorunur ciktisi - oynatici bunu gosterir."""
    d = tmp_path / "runs" / "1"
    d.mkdir(parents=True)
    (d / "k_annotated.mp4").write_bytes(b"\x00\x00\x00\x18ftypmp42")
    r = istemci.get("/videos/1/annotated")
    assert r.status_code == 200
    assert r.headers["content-type"] == "video/mp4"


def test_isaretli_video_yoksa_404(istemci):
    assert istemci.get("/videos/999/annotated").status_code == 404


def test_isaretli_video_404u_sebebini_soyluyor(istemci, tmp_path: Path):
    """Ciplak 404, calisan analizi ariza gibi gosteriyordu."""
    conn = store.baglan(tmp_path / "t.db")
    conn.execute("UPDATE videos SET durum = 'ISLENIYOR' WHERE id = 1")
    conn.commit()
    conn.close()
    r = istemci.get("/videos/1/annotated")
    assert r.status_code == 404
    assert "ISLENIYOR" in r.json()["detail"]


def test_olmayan_video_kaydi_ayirt_ediliyor(istemci):
    """Kaydi olmayan video ile islenmekte olan video ayni mesaji vermemeli."""
    assert "kaydi yok" in istemci.get("/videos/999/annotated").json()["detail"]


def test_klip_sunuluyor(istemci, tmp_path: Path):
    klip = tmp_path / "k.mp4"
    klip.write_bytes(b"\x00\x00\x00\x18ftypmp42")
    conn = store.baglan(tmp_path / "t.db")
    conn.execute("UPDATE events SET klip = ? WHERE event_id = ?",
                 (str(klip), "kaza_0001"))
    conn.commit()
    conn.close()
    assert istemci.get("/events/kaza_0001/klip").status_code == 200


def test_klip_yoksa_404(istemci):
    """Klibi olmayan olay icin sessizce bos video donmez."""
    assert istemci.get("/events/kaza_0001/klip").status_code == 404


def test_kara_nokta_ucu(istemci):
    d = istemci.get("/kara-nokta").json()
    assert d["toplam_olay"] == 2
    assert d["gps_li_olay"] == 2
    assert len(d["cografi_kumeler"]) == 1      # ikisi de ayni koordinatta
    assert "PLANLAMA" in d["uyari"]


def test_kara_nokta_yaricap_gecersizse_422(istemci):
    assert istemci.get("/kara-nokta?yaricap_m=0").status_code == 422


def test_kameralar_ucu(istemci):
    k = istemci.get("/kameralar").json()["kameralar"]
    assert k[0]["camera_id"] == "kavsak_01"
    assert k[0]["video"] == 1 and k[0]["olay"] == 2


def test_yol_gezinmesi_engelleniyor(istemci, monkeypatch, tmp_path: Path):
    """Dosya adi kullanicidan gelir; ../ ile dizin disina yazilmamali."""
    monkeypatch.setattr(api, "_isle", lambda *a, **k: None)
    istemci.post("/videos",
                 files={"file": ("../../kacti.mp4", b"0", "video/mp4")})
    assert (api.YUKLEME_DIZINI / "kacti.mp4").exists()
    assert not (api.YUKLEME_DIZINI.parent.parent / "kacti.mp4").exists()
