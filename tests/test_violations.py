"""M4 ihlal dedektorleri.

Testlerin agirligi bilerek NEGATIF taraftadir: bu modullerin tehlikesi
kacirmak degil, uydurmaktir. Yanlis bir ihlal kaydi bir vatandasin adina
yanlis bir kusur kaydi uretir; kacirilan ihlal maliyetsizdir. Plan da
precision'i recall'in onune koyar.

En kritik test: hiz modulu kalibrasyon dogrulanmadan CALISMAYI REDDEDER
(plan Faz 4 kabul kriteri, "test edilmis davranis").

Testler video/GPU gerektirmez: izler sentetik kurulur, boylece esik
davranislari deterministik dogrulanir.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from adgs import violations as m4
from adgs.calib import Kalibrasyon
from adgs.schema import ON_DEGERLENDIRME_NOTU, Detection, Track
from adgs.violations import lane, parking, redlight, speed, tailgating, wrongway
from adgs.violations.base import Baglam, kesisiyor_mu, nokta_poligonda, zemin

VIDEO = Path("data/testset/yok.mp4")
FPS = 25.0


# --- Yardimcilar ------------------------------------------------------------


def _iz(track_id: int, zeminler: dict[int, tuple[float, float]],
        cls: str = "otomobil", w: int = 40, h: int = 30) -> Track:
    """Zemin noktasi (alt-orta) verilen kutulardan Track uretir."""
    t = Track(track_id=track_id, cls=cls)
    for f, (x, y) in zeminler.items():
        t.frames[f] = Detection(
            bbox=(int(x - w / 2), int(y - h), int(x + w / 2), int(y)),
            cls=cls, conf=0.9,
        )
    return t


def _gecerli_kalib() -> Kalibrasyon:
    return Kalibrasyon(camera_id="t", H=np.eye(3), hata_yuzde=1.5, gecerli=True)


def _ctx(kamera: dict | None = None, kural: dict | None = None,
         kalib=None) -> Baglam:
    return Baglam(video=VIDEO, fps=FPS, kalib=kalib,
                  kamera=kamera or {}, kural=kural or {})


# --- Geometri ---------------------------------------------------------------


def test_zemin_alt_orta_nokta():
    assert zemin((100, 200, 140, 260)) == (120.0, 260.0)


def test_nokta_poligonda():
    kare = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert nokta_poligonda((5, 5), kare)
    assert not nokta_poligonda((15, 5), kare)


def test_bos_poligon_hep_disarida():
    assert not nokta_poligonda((5, 5), [])
    assert not nokta_poligonda((5, 5), [[0, 0], [1, 1]])  # 3 noktadan az


def test_kesisim_sonsuz_dogru_degil_parca():
    """Cizgi parcasinin UZANTISINDA kalan hareket "gecti" sayilmamali - yoksa
    kavsaga hic girmemis araclar kirmizi isik ihlali uretir."""
    assert kesisiyor_mu((5, -5), (5, 5), (0, 0), (10, 0))
    assert not kesisiyor_mu((50, -5), (50, 5), (0, 0), (10, 0))


# --- HIZ: Faz 4 kabul kriteri -----------------------------------------------


def test_hiz_kalibrasyonsuz_calismayi_REDDEDER():
    """Plan Faz 4 kabul kriteri. Sessizce bos donmek YETMEZ - sebep raporlanir."""
    iz = _iz(1, {f: (f * 20.0, 100.0) for f in range(30)})
    iz.world_xy = {f: (f * 0.8, 0.0) for f in range(30)}  # 72 km/s
    ctx = _ctx(kamera={"hiz_limiti_kmh": 50}, kalib=None)

    assert speed.tespit_et([iz], ctx) == []
    assert any("HIZ_IHLALI" in u and "TESPIT YAPILMADI" in u for u in ctx.uyarilar)


def test_hiz_dogrulanmamis_kalibrasyonla_calismaz():
    iz = _iz(1, {f: (f * 20.0, 100.0) for f in range(30)})
    iz.world_xy = {f: (f * 0.8, 0.0) for f in range(30)}
    # gecerli=False: dogrulama hatasi esigi asmis kalibrasyon
    reddedilmis = Kalibrasyon(camera_id="t", H=np.eye(3), hata_yuzde=42.0)
    ctx = _ctx(kamera={"hiz_limiti_kmh": 50}, kalib=reddedilmis)

    assert speed.tespit_et([iz], ctx) == []
    assert any("kalibrasyon dogrulamayi gecmedi" in u for u in ctx.uyarilar)


def test_hiz_limitsiz_kamerada_calismaz():
    iz = _iz(1, {f: (f * 20.0, 100.0) for f in range(30)})
    iz.world_xy = {f: (f * 0.8, 0.0) for f in range(30)}
    ctx = _ctx(kamera={}, kalib=_gecerli_kalib())

    assert speed.tespit_et([iz], ctx) == []
    assert any("hiz_limiti_kmh" in u for u in ctx.uyarilar)


def test_hiz_asimi_tespit_edilir():
    # 0.8 m/kare * 25 fps = 20 m/s = 72 km/s; limit 50 -> toleransli esik 55
    iz = _iz(1, {f: (f * 20.0, 100.0) for f in range(30)})
    iz.world_xy = {f: (f * 0.8, 0.0) for f in range(30)}
    ctx = _ctx(kamera={"hiz_limiti_kmh": 50}, kalib=_gecerli_kalib())

    olaylar = speed.tespit_et([iz], ctx)
    assert len(olaylar) == 1
    assert olaylar[0].alt_tip == "HIZ_IHLALI"
    assert olaylar[0].parties[0].violations[0].ihlal_kodu == "HIZ_IHLALI"
    assert "72 km/s" in " ".join(olaylar[0].notes)


def test_limit_icinde_hiz_ihlal_degil():
    # 0.5 m/kare -> 45 km/s, toleransli esik 55'in altinda
    iz = _iz(1, {f: (f * 12.0, 100.0) for f in range(30)})
    iz.world_xy = {f: (f * 0.5, 0.0) for f in range(30)}
    ctx = _ctx(kamera={"hiz_limiti_kmh": 50}, kalib=_gecerli_kalib())
    assert speed.tespit_et([iz], ctx) == []


def test_tolerans_musamaha_degil_olcum_payi():
    """52 km/s: limitin uzerinde ama %10 tolerans bandinda -> ihlal DEGIL.
    Kalibrasyon hatasi %10'a kadar kabul edildigi icin esik de o kadar tasinir."""
    iz = _iz(1, {f: (f * 15.0, 100.0) for f in range(30)})
    iz.world_xy = {f: (f * 0.5777, 0.0) for f in range(30)}  # ~52 km/s
    ctx = _ctx(kamera={"hiz_limiti_kmh": 50}, kalib=_gecerli_kalib())
    assert speed.tespit_et([iz], ctx) == []


# --- TAKIP MESAFESI ---------------------------------------------------------


def _takip_ciftesi(bosluk_m: float, adim_m: float = 0.8) -> list[Track]:
    takipci = _iz(1, {f: (f * 10.0, 300.0) for f in range(30)})
    takipci.world_xy = {f: (f * adim_m, 0.0) for f in range(30)}
    lider = _iz(2, {f: (f * 10.0 + 50, 300.0) for f in range(30)})
    lider.world_xy = {f: (f * adim_m + bosluk_m, 0.0) for f in range(30)}
    return [takipci, lider]


def test_takip_mesafesi_kalibrasyonsuz_calismaz():
    ctx = _ctx(kalib=None)
    assert tailgating.tespit_et(_takip_ciftesi(10.0), ctx) == []
    assert any("TAKIP_MESAFESI" in u for u in ctx.uyarilar)


def test_takip_mesafesi_ihlali_tespit_edilir():
    # 20 m/s, 10 m bosluk -> 0.5 sn < 2 sn guvenli esik
    ctx = _ctx(kalib=_gecerli_kalib())
    olaylar = tailgating.tespit_et(_takip_ciftesi(10.0), ctx)
    assert len(olaylar) == 1
    assert olaylar[0].parties[0].track_id == 1  # ihlal takipcide, liderde degil


def test_guvenli_mesafe_ihlal_degil():
    # 20 m/s, 45 m bosluk -> 2.25 sn > 2 sn
    ctx = _ctx(kalib=_gecerli_kalib())
    assert tailgating.tespit_et(_takip_ciftesi(45.0), ctx) == []


def test_duran_trafikte_takip_mesafesi_ihlali_uretilmez():
    """Kirmizi isikta tampon tampona beklemek ihlal degildir.

    Mesafe/hiz orani duran trafikte sifira gider; min_hiz filtresi olmadan bu
    modul her kavsakta yanlis pozitif fabrikasina donusur.
    """
    ctx = _ctx(kalib=_gecerli_kalib())
    assert tailgating.tespit_et(_takip_ciftesi(3.0, adim_m=0.0), ctx) == []


def test_yan_seritteki_arac_lider_sayilmaz():
    ctx = _ctx(kalib=_gecerli_kalib())
    takipci, lider = _takip_ciftesi(10.0)
    # Lideri 6 m yana kaydir; yanal esik 2.5 m
    lider.world_xy = {f: (x, 6.0) for f, (x, _y) in lider.world_xy.items()}
    assert tailgating.tespit_et([takipci, lider], ctx) == []


# --- KIRMIZI ISIK -----------------------------------------------------------

_KAMERA_ISIK = {"dur_cizgisi": [[0, 400], [640, 400]], "isik_roi": [0, 0, 20, 20]}


def _gecen_arac(dur_sonrasi: bool = False) -> Track:
    """Kare 6'da y=400 cizgisini gecer; sonra devam eder veya durur."""
    z = {f: (100.0, 360.0 + f * 4) for f in range(6)}  # 360 -> 380, yaklasma
    z[6] = (100.0, 420.0)
    for f in range(7, 20):
        z[f] = (100.0, 420.0 if dur_sonrasi else 420.0 + (f - 6) * 10)
    return _iz(1, z)


def test_kirmizi_isik_dur_cizgisi_yoksa_calismaz():
    ctx = _ctx(kamera={"isik_roi": [0, 0, 20, 20]})
    assert redlight.tespit_et([_gecen_arac()], ctx, durum={0: "KIRMIZI"}) == []
    assert any("dur_cizgisi" in u for u in ctx.uyarilar)


def test_kirmizi_isik_isik_roi_yoksa_calismaz():
    ctx = _ctx(kamera={"dur_cizgisi": [[0, 400], [640, 400]]})
    assert redlight.tespit_et([_gecen_arac()], ctx) == []
    assert any("isik_roi" in u for u in ctx.uyarilar)


def test_kirmizi_isik_ihlali_tespit_edilir():
    ctx = _ctx(kamera=_KAMERA_ISIK)
    durum = {f: "KIRMIZI" for f in range(20)}
    olaylar = redlight.tespit_et([_gecen_arac()], ctx, durum=durum)
    assert len(olaylar) == 1
    assert olaylar[0].alt_tip == "KIRMIZI_ISIK"


def test_yesil_isikta_gecis_ihlal_degil():
    ctx = _ctx(kamera=_KAMERA_ISIK)
    durum = {f: "YESIL" for f in range(20)}
    assert redlight.tespit_et([_gecen_arac()], ctx, durum=durum) == []


def test_belirsiz_isikta_ihlal_uretilmez():
    """Isik okunamiyorsa KIRMIZI VARSAYILMAZ - okunamayan sey ihlal uretmez."""
    ctx = _ctx(kamera=_KAMERA_ISIK)
    durum = {f: "BELIRSIZ" for f in range(20)}
    assert redlight.tespit_et([_gecen_arac()], ctx, durum=durum) == []
    assert any("hic KIRMIZI kare okunmadi" in u for u in ctx.uyarilar)


def test_cizgiyi_asip_duran_arac_ihlal_degil():
    """Ihlal kirmizida kavsaga GIRMEKtir; cizgiyi bir tekerlek boyu asip
    durmak degil."""
    ctx = _ctx(kamera=_KAMERA_ISIK)
    durum = {f: "KIRMIZI" for f in range(20)}
    assert redlight.tespit_et([_gecen_arac(dur_sonrasi=True)], ctx, durum=durum) == []


# --- TERS YON ---------------------------------------------------------------

_KAMERA_SERIT = {
    "seritler": [
        {"ad": "asagi", "poligon": [[0, 0], [640, 0], [640, 640], [0, 640]],
         "yon": [0, 1]},
    ]
}


def test_ters_yon_serit_tanimsizsa_calismaz():
    ctx = _ctx(kamera={})
    iz = _iz(1, {f: (100.0, 600.0 - f * 10) for f in range(20)})
    assert wrongway.tespit_et([iz], ctx) == []
    assert any("serit" in u for u in ctx.uyarilar)


def test_ters_yon_tespit_edilir():
    ctx = _ctx(kamera=_KAMERA_SERIT)
    iz = _iz(1, {f: (100.0, 600.0 - f * 10) for f in range(20)})  # yukari
    olaylar = wrongway.tespit_et([iz], ctx)
    assert len(olaylar) == 1
    assert olaylar[0].alt_tip == "TERS_YON"


def test_dogru_yonde_giden_arac_ihlal_degil():
    ctx = _ctx(kamera=_KAMERA_SERIT)
    iz = _iz(1, {f: (100.0, 100.0 + f * 10) for f in range(20)})  # asagi
    assert wrongway.tespit_et([iz], ctx) == []


def test_duran_arac_ters_yon_sayilmaz():
    """Duran aracin yonu yoktur; gurultuden dogan aci ihlal uretmemeli."""
    ctx = _ctx(kamera=_KAMERA_SERIT)
    iz = _iz(1, {f: (100.0, 300.0) for f in range(20)})
    assert wrongway.tespit_et([iz], ctx) == []


def test_kisa_sureli_sapma_ters_yon_sayilmaz():
    """Manevra/park cikisi birkac karede ters gorunur - min_ardisik_kare eler."""
    ctx = _ctx(kamera=_KAMERA_SERIT, kural={"TERS_YON": {"min_ardisik_kare": 8}})
    z = {f: (100.0, 300.0 - f * 10) for f in range(4)}                   # kisa geri
    z.update({f: (100.0, 260.0 + (f - 4) * 10) for f in range(4, 25)})   # sonra ileri
    assert wrongway.tespit_et([_iz(1, z)], ctx) == []


# --- HATALI PARK ------------------------------------------------------------

_KAMERA_PARK = {
    "yasak_park": [
        {"ad": "durak_onu", "poligon": [[0, 0], [200, 0], [200, 200], [0, 200]]},
    ]
}
_KURAL_PARK = {"HATALI_PARK": {"sure_sn": 2.0, "kayma_piksel": 8.0}}


def test_hatali_park_poligon_yoksa_calismaz():
    ctx = _ctx(kamera={}, kural=_KURAL_PARK)
    iz = _iz(1, {f: (100.0, 100.0) for f in range(80)})
    assert parking.tespit_et([iz], ctx) == []
    assert any("yasak_park" in u for u in ctx.uyarilar)


def test_hatali_park_tespit_edilir():
    ctx = _ctx(kamera=_KAMERA_PARK, kural=_KURAL_PARK)
    iz = _iz(1, {f: (100.0, 100.0) for f in range(80)})  # 2 sn = 50 kare
    olaylar = parking.tespit_et([iz], ctx)
    assert len(olaylar) == 1
    assert olaylar[0].alt_tip == "HATALI_PARK"


def test_kisa_duraklama_park_sayilmaz():
    """Kirmizi isikta bekleyen arac park etmis degildir - fark suredir."""
    ctx = _ctx(kamera=_KAMERA_PARK, kural=_KURAL_PARK)
    iz = _iz(1, {f: (100.0, 100.0) for f in range(20)})  # 0.8 sn
    assert parking.tespit_et([iz], ctx) == []


def test_yasak_alan_disinda_park_ihlal_degil():
    ctx = _ctx(kamera=_KAMERA_PARK, kural=_KURAL_PARK)
    iz = _iz(1, {f: (500.0, 500.0) for f in range(80)})  # poligon disi
    assert parking.tespit_et([iz], ctx) == []


# --- Kayit / acma-kapama ----------------------------------------------------


def test_serit_ihlali_varsayilan_kapali():
    """KTK m.55 sinyalsiz serit degisimini yasaklar; sinyal CCTV'de okunamaz.
    Bu yuzden modul config'te KAPALI gelir."""
    assert m4.kural_yukle()["aktif"]["lane"] is False


def test_faz_4a_uc_tipi_varsayilan_acik():
    kural = m4.kural_yukle()
    assert all(kural["aktif"][ad] for ad in m4.FAZ_4A)


def test_kapali_tip_config_ile_calistirilmaz():
    ctx = _ctx(kamera=_KAMERA_SERIT, kural={"aktif": {"wrongway": False}})
    iz = _iz(1, {f: (100.0, 600.0 - f * 10) for f in range(20)})
    assert m4.tespit_et([iz], ctx) == []  # tipler=None -> config karar verir


def test_acik_tip_config_ile_calistirilir():
    ctx = _ctx(kamera=_KAMERA_SERIT, kural={"aktif": {"wrongway": True}})
    iz = _iz(1, {f: (100.0, 600.0 - f * 10) for f in range(20)})
    assert len(m4.tespit_et([iz], ctx)) == 1


def test_acikca_istenen_tip_config_kapali_olsa_da_calisir_ama_uyarir():
    ctx = _ctx(kamera=_KAMERA_SERIT, kural={"aktif": {"wrongway": False}})
    iz = _iz(1, {f: (100.0, 600.0 - f * 10) for f in range(20)})
    assert len(m4.tespit_et([iz], ctx, ["wrongway"])) == 1
    assert any("acikca istendi" in u for u in ctx.uyarilar)


def test_bilinmeyen_tip_sessizce_yutulmaz():
    ctx = _ctx()
    assert m4.tespit_et([], ctx, ["yokboyle"]) == []
    assert any("bilinmeyen ihlal tipi" in u for u in ctx.uyarilar)


def test_her_dedektor_ayni_sozlesmeyi_uygular():
    for ad, modul in m4.KAYIT.items():
        assert callable(getattr(modul, "tespit_et", None)), ad
        assert isinstance(getattr(modul, "KOD", None), str), ad


def test_kalibrasyona_bagli_tipler_kayitta():
    assert set(m4.KALIBRASYONA_BAGLI) <= set(m4.KAYIT)


# --- Cikti sozlesmesi -------------------------------------------------------


def _ornek_ihlal():
    ctx = _ctx(kamera=_KAMERA_SERIT)
    iz = _iz(1, {f: (100.0, 600.0 - f * 10) for f in range(20)})
    return wrongway.tespit_et([iz], ctx)[0]


def test_ihlal_olayi_on_degerlendirme_notu_tasir():
    assert ON_DEGERLENDIRME_NOTU in _ornek_ihlal().notes


def test_ihlal_olayi_kanit_tasir():
    evt = _ornek_ihlal()
    assert evt.kanitli_mi()
    assert evt.evidence["keyframes"]
    assert evt.evidence["kareler"]


def test_hukuki_alanlari_m4_doldurmaz():
    """M4 'ne oldu'yu soyler. KTK maddesi ve ceza Faz 5'te M6/M7 isidir;
    M4'un buralari doldurmasi yetki asimidir."""
    ihlal = _ornek_ihlal().parties[0].violations[0]
    assert ihlal.ktk_madde is None
    assert ihlal.kusur_sinifi is None
    assert ihlal.ceza_tutari_try is None
    assert ihlal.ceza_tablosu_tarihi is None


def test_serit_ihlali_guveni_dusuk_ve_sinirini_soyler():
    ctx = _ctx(
        kamera={"seritler": [
            {"ad": "sol", "poligon": [[0, 0], [320, 0], [320, 640], [0, 640]]},
            {"ad": "sag", "poligon": [[320, 0], [640, 0], [640, 640], [320, 640]]},
        ]},
        kural={"SERIT_IHLALI": {"pencere_kare": 40, "min_gecis": 2}},
    )
    # sol -> sag -> sol
    z = {f: (100.0 + f * 20, 300.0) for f in range(12)}
    z.update({f: (540.0 - (f - 12) * 20, 300.0) for f in range(12, 26)})
    olaylar = lane.tespit_et([_iz(1, z)], ctx)
    assert len(olaylar) == 1
    assert olaylar[0].conf <= 0.5
    assert any("SINYAL DOGRULANAMADI" in n for n in olaylar[0].notes)
