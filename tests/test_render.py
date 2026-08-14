"""KVKK bulaniklastirma (Faz 7).

Bayrak "calisiyor" demek yetmez - goruntunun GERCEKTEN degistigi olculur.
Bulaniklastirma sessizce devre disi kalirsa kisisel veri (plaka, yuz) ham
olarak arsive ve sunuma girer.
"""

from __future__ import annotations

import numpy as np

from adgs import render
from adgs.schema import Detection, Event, Track


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


# --- Tarayicida oynayan cikti ve gorunur olay -------------------------------
#
# Iki hata birlikte "kaza videoda gorunmuyor" haline geliyordu:
#   1) mp4v (MPEG-4 Part 2) hicbir tarayicida oynamiyor -> oynatici siyah
#   2) olay yalnizca kanit karesinde ciziliyor -> 30 fps'te 2 kare, fark edilmez


def _olay(frame: int = 30, tip: str = "KAZA") -> Event:
    return Event(
        event_id=f"{tip.lower()}_1", tip=tip, alt_tip="CARPISMA",
        t_start=frame / 30, t_end=frame / 30, frame_start=frame, frame_end=frame,
        source_video="x.mp4", source_profile="cctv_fixed", conf=0.7,
        evidence={"keyframes": [frame], "kareler": {frame: [130, 100, 170, 140]}},
        gps=None, parties=[], notes=[],
    )


def _video_yaz(yol, kare_sayisi: int = 90, fps: float = 30.0):
    import cv2

    w = cv2.VideoWriter(str(yol), cv2.VideoWriter_fourcc(*"mp4v"), fps, (320, 240))
    for _ in range(kare_sayisi):
        w.write(np.full((240, 320, 3), 40, np.uint8))
    w.release()
    return yol


def test_cikti_tarayicida_oynayan_kodekle_yaziliyor(tmp_path):
    """avc1/H.264 olmali. mp4v gecerli dosya uretir ama tarayici oynatmaz."""
    src = _video_yaz(tmp_path / "k.mp4", 30)
    out = render.annotate_video(src, [], tmp_path / "c.mp4")
    d = out.read_bytes()
    assert b"avc1" in d and b"avcC" in d, "H.264 degil - tarayicida oynamaz"


def test_klip_de_tarayicida_oynayan_kodekle_yaziliyor(tmp_path):
    src = _video_yaz(tmp_path / "k.mp4", 60)
    out = render.save_clip(src, 10, 20, tmp_path / "klip.mp4")
    assert out is not None and b"avc1" in out.read_bytes()


def test_tek_karelik_olay_saniyelerce_gorunur():
    """Kanit karesi 1 taneyken bile olay izlenebilir sure ekranda kalmali."""
    plan, bant = render._olay_plani([_olay(frame=30)], fps=30.0)
    assert len(plan) >= 30 * render._ASGARI_SN
    assert 30 in plan and 60 in plan


def test_olay_kutusu_tum_sure_boyunca_ciziliyor():
    """Kanit kareleri arasindaki bosluklarda kutu kaybolmamali."""
    evt = _olay(frame=10)
    evt.frame_start, evt.frame_end = 10, 40
    evt.evidence["kareler"] = {10: [0, 0, 10, 10], 40: [50, 50, 60, 60]}
    plan, _ = render._olay_plani([evt], fps=30.0)
    assert all(f in plan for f in range(10, 41)), "kutu arada kayboluyor"
    assert plan[11][0][0] == [0, 0, 10, 10]      # basa yakin -> ilk kutu
    assert plan[39][0][0] == [50, 50, 60, 60]    # sona yakin -> son kutu


def test_kanit_karesi_olmayan_olay_cizilmiyor():
    """Kanitsiz olay uydurma kutu uretmemeli."""
    evt = _olay()
    evt.evidence = {"keyframes": [30]}
    plan, bant = render._olay_plani([evt], fps=30.0)
    assert not plan and not bant


def test_bant_olay_suresince_gorunuyor():
    _, bant = render._olay_plani([_olay(frame=5)], fps=30.0)
    assert bant[5][0][0], "bant etiketi bos"


def test_yol_hasari_da_videoda_gosteriliyor(tmp_path):
    """Kaza ve yol hasari ayni videoda birlikte isaretlenmeli."""
    src = _video_yaz(tmp_path / "k.mp4", 90)
    olaylar = [_olay(30, "KAZA"), _olay(60, "ALTYAPI")]
    plan, bant = render._olay_plani(olaylar, fps=30.0)
    assert any("KAZA" in e for e, _ in bant[30])
    assert any("ALTYAPI" in e or "CARPISMA" in e for e, _ in bant[60])
    render.annotate_video(src, olaylar, tmp_path / "c.mp4")


def test_ffmpeg_yolu_bulunuyor_veya_none():
    """ffmpeg ZORUNLU degil - yoksa OpenCV yazicisina dusulmeli."""
    y = render._ffmpeg_yolu()
    assert y is None or isinstance(y, str)


def test_ffmpeg_yoksa_opencv_yazicisina_dusuluyor(tmp_path, monkeypatch):
    """Bagimlilik kaybolursa cikti buyur ama URETIM DURMAZ."""
    monkeypatch.setattr(render, "_ffmpeg_yolu", lambda: None)
    src = _video_yaz(tmp_path / "k.mp4", 20)
    out = render.annotate_video(src, [], tmp_path / "c.mp4")
    assert out.exists() and out.stat().st_size > 0


def test_ffmpeg_ciktisi_aninda_oynatilabilir(tmp_path):
    """faststart: moov atomu mdat'tan ONCE - tarayici beklemeden oynatir."""
    if render._ffmpeg_yolu() is None:
        import pytest
        pytest.skip("ffmpeg yok")
    src = _video_yaz(tmp_path / "k.mp4", 30)
    out = render.annotate_video(src, [_olay(10)], tmp_path / "c.mp4")
    d = out.read_bytes()
    assert b"avc1" in d
    assert 0 <= d.find(b"moov") < d.find(b"mdat"), "faststart yok"


def test_etikette_zaman_var():
    """5 dakikalik kayitta "hangi saniye" sorusu videonun uzerinde yanitlanmali."""
    evt = _olay(frame=1800)
    evt.t_start = 60.0
    assert render._etiket(evt).startswith("01:00 ")


def test_sn_mmss():
    assert render.sn_mmss(0) == "00:00"
    assert render.sn_mmss(65.7) == "01:05"
    assert render.sn_mmss(-3) == "00:00"


def test_isaretli_video_atomik_yaziliyor(tmp_path):
    """Dosya ayni anda tarayiciya sunuluyor - yarim hali gorunmemeli.

    Yeniden analiz sirasinda uzerine yazmak, izleyene suresiz bir MP4 verir ve
    oynatici birkac saniye sonra videoyu bitmis sayip sona atlar.
    """
    src = _video_yaz(tmp_path / "k.mp4", 30)
    hedef = tmp_path / "c.mp4"
    hedef.write_bytes(b"ESKI DOSYA")
    gecici = hedef.with_name(f"{hedef.stem}.yaziliyor{hedef.suffix}")

    render.annotate_video(src, [], hedef)
    assert not gecici.exists(), "gecici dosya temizlenmemis"
    assert hedef.read_bytes()[:10] != b"ESKI DOSYA"
    assert b"avc1" in hedef.read_bytes()


def test_yazim_sirasinda_hedef_bozulmuyor(tmp_path, monkeypatch):
    """Yazim yarida kalirsa ESKI dosya yerinde kalmali."""
    src = _video_yaz(tmp_path / "k.mp4", 30)
    hedef = tmp_path / "c.mp4"
    hedef.write_bytes(b"ESKI AMA GECERLI")

    def patla(*a, **k):
        raise RuntimeError("kodlayici coktu")

    monkeypatch.setattr(render, "_yazici", patla)
    try:
        render.annotate_video(src, [], hedef)
    except RuntimeError:
        pass
    assert hedef.read_bytes() == b"ESKI AMA GECERLI"


def test_klip_olayin_basindan_once_basliyor(tmp_path):
    """Kanit klibi carpisma ANINI icermeli, yalnizca sonrasini degil."""
    src = _video_yaz(tmp_path / "k.mp4", 300, fps=30.0)
    out = render.save_clip(src, 150, 200, tmp_path / "klip.mp4")
    assert out is not None
    import cv2
    cap = cv2.VideoCapture(str(out))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.release()
    # 3 sn on + olay (50 kare) + 2 sn arka = 90 + 51 + 60
    assert n >= 90 + 51, f"klip cok kisa: {n} kare"


def test_klip_on_tamponu_arkadan_uzun():
    """Yaklasma gorunmeli; olay sonrasi bu kadar uzun olmasi gerekmiyor."""
    assert render.ON_TAMPON_SN > render.ARKA_TAMPON_SN


def test_klip_video_basinda_kirpiliyor(tmp_path):
    """Olay 1. saniyedeyse negatif kareye gidilmemeli."""
    src = _video_yaz(tmp_path / "k.mp4", 120, fps=30.0)
    out = render.save_clip(src, 5, 20, tmp_path / "klip.mp4")
    assert out is not None and out.stat().st_size > 0


def test_eski_simetrik_tampon_hala_calisiyor(tmp_path):
    src = _video_yaz(tmp_path / "k.mp4", 200, fps=30.0)
    out = render.save_clip(src, 100, 120, tmp_path / "klip.mp4", tampon=5)
    assert out is not None
