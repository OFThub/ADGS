"""Kamera kalibrasyonu - piksel <-> yer duzlemi (metre) donusumu.

Hiza bagli her cikti buradan gecer. Kalibrasyon dogrulamayi GECMEZSE hiz/mesafe
hesabi yapilmaz: yanlis homografi yanlis hiz, yanlis hiz yanlis ceza demektir.
Bu yuzden `gecerli` alani varsayilan olarak False'tur ve yalnizca olculen hata
esigin altinda kalirsa True olur.

Dogrulama noktalari homografiyi FIT ETMEK icin kullanilan noktalardan ayridir;
fit noktalariyla dogrulama yapmak sifira yakin hata verir ve hicbir sey kanitlamaz.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

VARSAYILAN_MAKS_HATA = 10.0  # yuzde - plan Faz 2 kabul kriteri


@dataclass
class Kalibrasyon:
    """Bir kameranin yer duzlemi kalibrasyonu.

    `gecerli` False ise hiz/mesafe modulleri CALISMAMALIDIR. Bunu cagiranin
    kontrol etmesi gerekir; `mesafe_m` zaten kendisi de reddeder.
    """

    camera_id: str
    H: np.ndarray | None = None  # 3x3, piksel -> metre
    hata_yuzde: float | None = None
    maks_hata_yuzde: float = VARSAYILAN_MAKS_HATA
    gecerli: bool = False
    notlar: list[str] = field(default_factory=list)


def homografi(piksel: list, dunya: list) -> np.ndarray:
    """4+ nokta eslesmesinden piksel->metre homografisi cikarir."""
    import cv2

    p = np.asarray(piksel, dtype=np.float32)
    d = np.asarray(dunya, dtype=np.float32)
    if len(p) < 4 or len(p) != len(d):
        raise ValueError(f"En az 4 eslesen nokta gerekli (piksel={len(p)}, dunya={len(d)})")
    H, _ = cv2.findHomography(p, d, method=0)
    if H is None:
        raise ValueError("Homografi cozulemedi - noktalar dogrusal veya cakisik olabilir")
    return H


def piksel_to_dunya(H: np.ndarray, nokta: tuple[float, float]) -> tuple[float, float]:
    """Tek bir piksel noktasini yer duzlemi metre koordinatina cevirir."""
    v = H @ np.array([nokta[0], nokta[1], 1.0], dtype=np.float64)
    if abs(v[2]) < 1e-12:
        raise ValueError(f"Nokta ufuk cizgisinde ({nokta}) - dunya koordinati tanimsiz")
    return float(v[0] / v[2]), float(v[1] / v[2])


def olcum_hatasi(H: np.ndarray, dogrulama: list[dict]) -> float:
    """Bilinen gercek mesafelere gore ortalama mutlak yuzde hata.

    Her dogrulama kaydi: {"piksel": [[x1,y1],[x2,y2]], "gercek_m": 3.5}
    """
    if not dogrulama:
        raise ValueError("Dogrulama noktasi yok - kalibrasyon dogrulanamaz")
    hatalar = []
    for kayit in dogrulama:
        p1, p2 = kayit["piksel"]
        gercek = float(kayit["gercek_m"])
        if gercek <= 0:
            raise ValueError(f"gercek_m pozitif olmali: {gercek}")
        a = piksel_to_dunya(H, p1)
        b = piksel_to_dunya(H, p2)
        olculen = math.dist(a, b)
        hatalar.append(abs(olculen - gercek) / gercek * 100.0)
    return float(np.mean(hatalar))


def kalibre_et(cfg: dict, camera_id: str = "?") -> Kalibrasyon:
    """Kamera config sozlugunden Kalibrasyon uretir. Hata durumunda gecerli=False."""
    k = Kalibrasyon(
        camera_id=cfg.get("camera_id", camera_id),
        maks_hata_yuzde=float(cfg.get("maks_hata_yuzde", VARSAYILAN_MAKS_HATA)),
    )
    h = cfg.get("homografi") or {}
    try:
        k.H = homografi(h["piksel"], h["dunya"])
    except (KeyError, ValueError) as e:
        k.notlar.append(f"Homografi kurulamadi: {e}")
        return k

    try:
        k.hata_yuzde = olcum_hatasi(k.H, cfg.get("dogrulama") or [])
    except (KeyError, ValueError) as e:
        k.notlar.append(f"Dogrulama yapilamadi: {e}")
        return k

    k.gecerli = k.hata_yuzde <= k.maks_hata_yuzde
    k.notlar.append(
        f"Geri donusum hatasi %{k.hata_yuzde:.2f} (esik %{k.maks_hata_yuzde:.1f}) - "
        + ("kalibrasyon GECERLI" if k.gecerli else "kalibrasyon REDDEDILDI, hiz modulleri kapali")
    )
    return k


def yukle(yol: str | Path) -> Kalibrasyon:
    """config/cameras/<id>.yaml dosyasindan kalibrasyon yukler."""
    import yaml

    p = Path(yol)
    if not p.exists():
        return Kalibrasyon(camera_id=p.stem, notlar=[f"Kamera config bulunamadi: {p}"])
    cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    return kalibre_et(cfg, camera_id=p.stem)


def mesafe_m(k: Kalibrasyon, p1: tuple[float, float], p2: tuple[float, float]) -> float | None:
    """Iki piksel nokta arasi gercek mesafe (m). Kalibrasyon gecersizse None.

    None donmesi 'olculemedi' demektir - cagiran taraf bunu 0 veya tahminle
    doldurmamalidir.
    """
    if not k.gecerli or k.H is None:
        return None
    return math.dist(piksel_to_dunya(k.H, p1), piksel_to_dunya(k.H, p2))


def hiz_kmh(k: Kalibrasyon, p1: tuple[float, float], p2: tuple[float, float],
            dt_s: float) -> float | None:
    """Iki kare arasi hiz (km/s). Kalibrasyon gecersizse veya dt<=0 ise None."""
    if dt_s <= 0:
        return None
    m = mesafe_m(k, p1, p2)
    return None if m is None else m / dt_s * 3.6
