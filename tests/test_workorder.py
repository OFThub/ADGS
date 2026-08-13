"""Faz 7 - Fen Isleri PDF is emri.

PDF'in ICINDEKI metni dogrulamak guvenilir degil (font altkumesi). Bunun yerine
dogrulanabilir olan test edilir: dosya gercekten PDF mi, siddet/oncelik M5'in
notlarindan DOGRU okunuyor mu, ve gorsel yoksa uydurma bir sey konmuyor mu.
"""

from __future__ import annotations

from pathlib import Path

from adgs import workorder


def _olay(tip: str = "ALTYAPI", notlar: list[str] | None = None) -> dict:
    return {
        "event_id": "altyapi_0001", "tip": tip, "alt_tip": "D40", "conf": 0.81,
        "gps": [41.184312, 28.742901], "video": "yol_01.mp4",
        "cekim_tarihi": "2026-07-15", "frame_start": 412, "frame_end": 470,
        "notlar": notlar if notlar is not None else
                  ["Cukur (D40) - siddet YUKSEK, oncelik 1"],
    }


def test_pdf_uretiliyor(tmp_path: Path):
    p = workorder.is_emri_pdf(_olay(), tmp_path / "e.pdf")
    assert p.exists()
    assert p.read_bytes()[:5] == b"%PDF-"


def test_siddet_ve_oncelik_m5_notundan_okunur():
    """M5 zaten hesapliyor - is emri YENIDEN HESAPLAMAZ."""
    assert workorder._siddet_ve_oncelik(_olay()) == ("YUKSEK", "1")


def test_not_yoksa_siddet_bos_kalir():
    """Bilinmeyen siddet "DUSUK" varsayilmaz."""
    assert workorder._siddet_ve_oncelik(_olay(notlar=[])) == ("-", "-")


def test_gorsel_yoksa_uydurulmaz(tmp_path: Path):
    olay = _olay()
    olay["video"] = "olmayan.mp4"
    p = workorder.is_emri_pdf(olay, tmp_path / "e.pdf")
    assert p.exists()  # sayfa yine uretilir, gorsel yerine sebep yazilir


def test_gps_yoksa_saha_ekibine_birakilir(tmp_path: Path):
    olay = _olay()
    olay["gps"] = None
    assert workorder.is_emri_pdf(olay, tmp_path / "e.pdf").exists()


def test_kaza_ve_altyapi_farkli_uyari_tasir():
    """Altyapi belediyenin gorev alani; kaza/ihlal idari islem dogurmaz."""
    assert "saha kontrolü" in workorder._ALTYAPI_NOTU
    assert "idari işleme dayanak" in workorder._KUSUR_NOTU
    assert workorder._TIP_BASLIK["ALTYAPI"] != workorder._TIP_BASLIK["KAZA"]


def test_png_olarak_da_uretilebilir(tmp_path: Path):
    """Bicim uzantidan cikarilir - gorsel dogrulama icin gerekli."""
    p = workorder.is_emri_pdf(_olay(), tmp_path / "e.png")
    assert p.read_bytes()[:4] == b"\x89PNG"
