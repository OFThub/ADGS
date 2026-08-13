"""Faz 8 - kara nokta analizi.

En kritik davranis: GPS'li ve GPS'siz olaylarin BIRLESTIRILMEMESI. Sabit
kameranin olaylarina uydurma koordinat verip haritaya basmak, var olmayan bir
kara nokta uretirdi.
"""

from __future__ import annotations

from adgs import hotspot

# Arnavutkoy civari; ~35 m ve ~1 km ayrik noktalar.
A = [41.184000, 28.742000]
A_YAKIN = [41.184300, 28.742100]
B = [41.193000, 28.742000]


def _o(event_id: str, tip: str = "KAZA", gps=None, camera_id=None, notlar=None) -> dict:
    return {"event_id": event_id, "tip": tip, "gps": gps,
            "camera_id": camera_id, "notlar": notlar or []}


# --- Mesafe -----------------------------------------------------------------


def test_haversine_makul_mesafe_veriyor():
    d = hotspot.mesafe_m(tuple(A), tuple(A_YAKIN))
    assert 20 < d < 60, d


def test_uzak_noktalar_ayirt_ediliyor():
    assert hotspot.mesafe_m(tuple(A), tuple(B)) > 900


def test_ayni_nokta_sifir():
    assert hotspot.mesafe_m(tuple(A), tuple(A)) == 0.0


# --- Agirlik ----------------------------------------------------------------


def test_kaza_ihlalden_agir():
    assert hotspot.agirlik(_o("1", "KAZA")) > hotspot.agirlik(_o("2", "IHLAL"))


def test_ihlal_altyapidan_agir():
    assert hotspot.agirlik(_o("1", "IHLAL")) > hotspot.agirlik(_o("2", "ALTYAPI"))


def test_siddet_agirligi_yukseltiyor():
    hafif = _o("1", "ALTYAPI", notlar=["Cukur - siddet DUSUK, oncelik 3"])
    agir = _o("2", "ALTYAPI", notlar=["Cukur - siddet YUKSEK, oncelik 1"])
    assert hotspot.agirlik(agir) > hotspot.agirlik(hafif)


def test_bilinmeyen_tip_cokme_yapmaz():
    assert hotspot.agirlik(_o("1", "BILINMEYEN")) > 0


# --- Cografi kumeleme -------------------------------------------------------


def test_yakin_olaylar_ayni_kumede():
    k = hotspot.kumele([_o("1", gps=A), _o("2", gps=A_YAKIN)])
    assert len(k) == 1
    assert k[0]["olay_sayisi"] == 2


def test_uzak_olaylar_ayri_kumede():
    assert len(hotspot.kumele([_o("1", gps=A), _o("2", gps=B)])) == 2


def test_yaricap_daraltilinca_ayrisiyor():
    k = hotspot.kumele([_o("1", gps=A), _o("2", gps=A_YAKIN)], yaricap_m=10.0)
    assert len(k) == 2


def test_kumeler_agirliga_gore_sirali():
    olaylar = [_o("1", "ALTYAPI", gps=A),
               _o("2", "KAZA", gps=B), _o("3", "KAZA", gps=B)]
    k = hotspot.kumele(olaylar)
    assert k[0]["olay_sayisi"] == 2       # iki kaza once
    assert k[0]["agirlik"] > k[1]["agirlik"]


def test_kume_merkezi_noktalarin_ortasinda():
    k = hotspot.kumele([_o("1", gps=A), _o("2", gps=A_YAKIN)])[0]
    assert min(A[0], A_YAKIN[0]) <= k["lat"] <= max(A[0], A_YAKIN[0])


def test_gpssiz_olay_kumeye_girmez():
    assert hotspot.kumele([_o("1", gps=None)]) == []


def test_bozuk_gps_cokme_yapmaz():
    for bozuk in (None, [], [41.0], [None, 28.0]):
        assert hotspot.kumele([_o("1", gps=bozuk)]) == []


# --- Kamera bazli grup ------------------------------------------------------


def test_gpssiz_olaylar_kameraya_gore_gruplanir():
    g = hotspot.kamera_bazli([_o("1", camera_id="k1"), _o("2", camera_id="k1"),
                              _o("3", camera_id="k2")])
    assert len(g) == 2
    assert g[0]["camera_id"] == "k1" and g[0]["olay_sayisi"] == 2


def test_gpsli_olay_kamera_grubuna_girmez():
    """Ayni olay iki kez sayilmamali."""
    assert hotspot.kamera_bazli([_o("1", gps=A, camera_id="k1")]) == []


def test_kamerasiz_olay_kaybolmaz():
    assert hotspot.kamera_bazli([_o("1")])[0]["camera_id"] == "(kamera tanimsiz)"


# --- Butun ------------------------------------------------------------------


def test_gpsli_ve_gpssiz_KARISTIRILMAZ():
    """Sabit kameranin olayina uydurma koordinat verilirse haritada olmayan bir
    kara nokta cikar - iki grup ayri alanlarda kalmali."""
    d = hotspot.analiz([_o("1", gps=A), _o("2", camera_id="k1")])
    assert d["gps_li_olay"] == 1
    assert len(d["cografi_kumeler"]) == 1
    assert len(d["kamera_gruplari"]) == 1
    assert d["toplam_olay"] == 2


def test_analiz_planlama_uyarisi_tasir():
    d = hotspot.analiz([])
    assert "PLANLAMA" in d["uyari"]
    assert "kusur veya ceza dogurmaz" in d["uyari"]


def test_bos_girdi_cokme_yapmaz():
    d = hotspot.analiz([])
    assert d["toplam_olay"] == 0
    assert d["cografi_kumeler"] == [] and d["kamera_gruplari"] == []
