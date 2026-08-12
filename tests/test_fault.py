"""M6 kusur motoru - ZORUNLU test dosyasi (plan Faz 5 kabul kriteri).

Bu modul GPU'suz ve deterministiktir; sistemin hukuken en hassas kismi ayni
zamanda en kolay test edilen kismidir - kapsamsiz birakmak icin mazeret yok.

Iki sey burada kilitlenir:
    1. m.84 KAPALI listedir - listede olmayan hicbir ihlal ASLI olamaz
    2. Cikti hicbir kosulda YUZDE icermez
"""

from __future__ import annotations

import pytest

from adgs import fault
from adgs.schema import Event, Party, Violation

TABLO = fault.yukle()


def _evt(taraflar: dict[int, list[str]], tip: str = "KAZA") -> Event:
    return Event(
        event_id="t",
        tip=tip,
        alt_tip="CARPISMA",
        t_start=0.0, t_end=1.0, frame_start=0, frame_end=25,
        source_video="-", source_profile="cctv_fixed", conf=0.9,
        evidence={"keyframes": [0]},
        parties=[
            Party(track_id=tid,
                  violations=[Violation(ihlal_kodu=k, conf=0.9) for k in kodlar])
            for tid, kodlar in taraflar.items()
        ],
    )


def _ihlal(evt: Event, track_id: int, sira: int = 0) -> Violation:
    return next(p for p in evt.parties if p.track_id == track_id).violations[sira]


# --- Carpisma geometrisi senaryolari ----------------------------------------
# (senaryo adi, ihlal kodu, beklenen madde, beklenen sinif)
# madde None ise ihlal m.84'un kapali listesinde YOKTUR.
SENARYOLAR = [
    ("arkadan_carpma",                    "ARKADAN_CARPMA",             "84/d", "ASLI"),
    ("kavsakta_yandan_kirmizi_isik",      "KIRMIZI_ISIK",               "84/a", "ASLI"),
    ("kavsakta_yandan_gecis_onceligi",    "KAVSAK_GECIS_ONCELIGI",      "84/h", "ASLI"),
    ("serit_degisimi_sonrasi_carpisma",   "SERIDE_TECAVUZ",             "84/g", "ASLI"),
    ("ters_yon_karsilikli_carpisma",      "TERS_YON",                   "84/b", "ASLI"),
    ("sola_donuste_karsiya_carpma",       "HATALI_DONUS",               "84/f", "ASLI"),
    ("park_halindeki_araca_carpma",       "PARK_HALINDEKI_ARACA_CARPMA", "84/l", "ASLI"),
    ("geri_manevrada_carpma",             "HATALI_MANEVRA",             "84/j", "ASLI"),
    ("donel_kavsakta_carpma",             "KAVSAK_GECIS_ONCELIGI",      "84/h", "ASLI"),
    ("karsi_seride_gecerek_carpma",       "KARSI_SERIDE_GECME",         "84/c", "ASLI"),
    ("gecme_yasaginda_gecerek_carpma",    "GECME_YASAGI_IHLALI",        "84/e", "ASLI"),
    ("dar_kaplamada_gecis_onceligi",      "DAR_KAPLAMA_GECIS_ONCELIGI", "84/i", "ASLI"),
    ("yerlesim_disi_parkta_carpma",       "YERLESIM_DISI_PARK",         "84/k", "ASLI"),
    # m.84'un kapali listesinde OLMAYANLAR - hepsi TALI kalmali
    ("yaya_gecidinde_yayaya_carpma",      "YAYA_GECIDI_IHLALI",         None,   "TALI"),
    ("yaya_gecidi_disinda_yayaya_carpma", "DIKKATSIZ_SURUS",            None,   "TALI"),
    ("hiz_asimiyla_carpisma",             "HIZ_IHLALI",                 None,   "TALI"),
    ("takip_mesafesi_ihlaliyle_carpisma", "TAKIP_MESAFESI",             None,   "TALI"),
    ("sehir_ici_hatali_park",             "HATALI_PARK",                None,   "TALI"),
    ("serit_degisimi_sinyalsiz",          "SERIT_IHLALI",               None,   "TALI"),
]


@pytest.mark.parametrize("ad,kod,madde,sinif", SENARYOLAR,
                         ids=[s[0] for s in SENARYOLAR])
def test_senaryo(ad: str, kod: str, madde: str | None, sinif: str):
    evt = _evt({1: [kod]})
    fault.uygula([evt], TABLO)
    v = _ihlal(evt, 1)
    assert v.ktk_madde == madde, ad
    assert v.kusur_sinifi == sinif, ad


def test_senaryo_sayisi_plan_esigini_karsilar():
    assert len(SENARYOLAR) >= 15


# --- Kapali liste disiplini -------------------------------------------------


def test_sehir_ici_hatali_park_84k_ile_karistirilmaz():
    """84/k YALNIZCA yerlesim birimleri DISINDAKI karayolu icindir.

    M4'un tespit ettigi sehir ici hatali park bu kapsamda degildir. "Benzer
    goruyor" diye baglamak, bir vatandasa yanlis asli kusur atamaktir.
    """
    evt = _evt({1: ["HATALI_PARK"], 2: ["YERLESIM_DISI_PARK"]})
    fault.uygula([evt], TABLO)
    assert _ihlal(evt, 1).kusur_sinifi == "TALI"
    assert _ihlal(evt, 1).ktk_madde is None
    assert _ihlal(evt, 2).kusur_sinifi == "ASLI"
    assert _ihlal(evt, 2).ktk_madde == "84/k"


def test_serit_degisimi_ile_seride_tecavuz_ayrilir():
    """84/g "serit tecavuzu"dur; M4'un tespit ettigi sinyalsiz serit
    DEGISIMIDIR. Ikisi ayni sey degildir."""
    evt = _evt({1: ["SERIT_IHLALI"], 2: ["SERIDE_TECAVUZ"]})
    fault.uygula([evt], TABLO)
    assert _ihlal(evt, 1).kusur_sinifi == "TALI"
    assert _ihlal(evt, 2).kusur_sinifi == "ASLI"


def test_bilinmeyen_kod_asliye_yaklastirilmaz():
    evt = _evt({1: ["HIC_BOYLE_BIR_KOD_YOK"]})
    fault.uygula([evt], TABLO)
    assert _ihlal(evt, 1).kusur_sinifi == "TALI"
    assert _ihlal(evt, 1).ktk_madde is None


def test_tablo_12_bendin_tamamini_icerir():
    maddeler = {h["madde"] for h in TABLO["asli_kusur_halleri"]}
    beklenen = {f"84/{h}" for h in "abcdefghijkl"}
    assert maddeler == beklenen
    assert TABLO["meta"]["bent_sayisi"] == len(beklenen)


# --- Cok tarafli senaryolar -------------------------------------------------


def test_takip_mesafesi_tali_arkadan_carpma_asli():
    """Ayni tarafta iki ihlal: takip mesafesi m.84'te yoktur (TALI), ama
    carpisma gerceklestiyse asli kusuru doguran ARKADAN_CARPMA'dir."""
    evt = _evt({1: ["TAKIP_MESAFESI", "ARKADAN_CARPMA"]})
    fault.uygula([evt], TABLO)
    assert _ihlal(evt, 1, 0).kusur_sinifi == "TALI"
    assert _ihlal(evt, 1, 1).kusur_sinifi == "ASLI"
    assert _ihlal(evt, 1, 1).ktk_madde == "84/d"


def test_zincirleme_carpmada_her_taraf_ayri_degerlendirilir():
    evt = _evt({1: ["ARKADAN_CARPMA"], 2: ["ARKADAN_CARPMA"], 3: []})
    fault.uygula([evt], TABLO)
    assert _ihlal(evt, 1).kusur_sinifi == "ASLI"
    assert _ihlal(evt, 2).kusur_sinifi == "ASLI"
    assert not next(p for p in evt.parties if p.track_id == 3).violations


def test_sag_donuste_bisiklete_carpma_bisiklet_tarafi_bos_kalir():
    evt = _evt({1: ["HATALI_DONUS"], 2: []})
    fault.uygula([evt], TABLO)
    assert _ihlal(evt, 1).ktk_madde == "84/f"
    assert any("Iz #2" in n and "TESPIT_EDILEMEDI" in n for n in evt.notes)


# --- TESPIT_EDILEMEDI != KUSURSUZ -------------------------------------------


def test_ihlalsiz_taraf_kusursuz_ilan_edilmez():
    """Sistemin uretebilecegi en tehlikeli yanlis: sessizce "kusursuz"."""
    evt = _evt({7: []})
    fault.uygula([evt], TABLO)
    not_metni = " ".join(evt.notes)
    assert "TESPIT_EDILEMEDI" in not_metni
    assert "KUSURSUZ oldugu anlamina" in not_metni


def test_ozet_kusursuz_kelimesini_olumlu_kullanmaz():
    evt = _evt({7: []})
    fault.uygula([evt], TABLO)
    assert fault.ozet(evt) == ["iz #7: TESPIT_EDILEMEDI (kusursuz DEGIL)"]


# --- Yuzde uretilmez --------------------------------------------------------


def test_hicbir_ciktida_yuzde_yok():
    """KTY m.156/3: tutanagi duzenleyen bile kusur orani belirtmez."""
    evt = _evt({1: ["KIRMIZI_ISIK"], 2: []})
    fault.uygula([evt], TABLO)
    metin = " ".join(evt.notes + fault.ozet(evt))
    assert "%" not in metin
    assert "yuzde" not in metin.lower()


def test_violation_semasinda_oran_alani_yok():
    alanlar = set(Violation.__dataclass_fields__)
    assert not {a for a in alanlar if "oran" in a or "yuzde" in a}


# --- Kapsam ve guvenlik -----------------------------------------------------


def test_altyapi_olayi_kusur_almaz():
    """Yol hasari belediyenin kendi gorev alanidir; kusur/ceza dogurmaz."""
    evt = _evt({1: ["KIRMIZI_ISIK"]}, tip="ALTYAPI")
    fault.uygula([evt], TABLO)
    assert _ihlal(evt, 1).kusur_sinifi is None


def test_tablo_yoksa_sessizce_calismaz(tmp_path):
    """Tablosuz calismak her ihlali TALI yapardi - sessiz ve yanlis."""
    with pytest.raises(FileNotFoundError, match="Kusur tablosu bulunamadi"):
        fault.yukle(tmp_path / "yok.yaml")


def test_dogrulanmamis_tablo_uyarisi_eklenir():
    """meta.dogrulama_tarihi bos oldugu surece cikti bunu tasimalidir."""
    evt = _evt({1: ["KIRMIZI_ISIK"]})
    fault.uygula([evt], TABLO)
    assert any("kusur tablosu" in n and "dogrulanmamistir" in n for n in evt.notes)


def test_on_degerlendirme_notu_korunur():
    from adgs.schema import ON_DEGERLENDIRME_NOTU

    evt = _evt({1: ["KIRMIZI_ISIK"]})
    fault.uygula([evt], TABLO)
    assert ON_DEGERLENDIRME_NOTU in evt.notes
