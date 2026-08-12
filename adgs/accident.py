"""M3 - Kaza tespiti (Asama A: kural tabanli).

Uc sinyalin UCU BIRDEN gerekir:
    1. Iki izin bounding box'lari cakisir (IoU esigi asilir)
    2. Cakismadan hemen sonra ani hiz DUSUSU veya yon DEGISIMI olur
    3. Sonrasinda en az N saniye beklenmeyen hareketsizlik surer

Neden ucu birden: tek basina cakisma kavsakta surekli olur - kamera acisindan
one gecen bir arac arkadakini ortuyor (okluzyon). Okluzyonda araclar hizini
korur ve yoluna devam eder; kazada etmez. Bu tasarim recall'i dusurur ama
precision'i yukseltir - plan geregi bilincli bir tercih (yanlis kaza kaydi,
kacirilan kazadan pahalidir).

Kalibrasyon gecerliyse dorduncu bir filtre daha devreye girer: goruntude
cakisan ama yer duzleminde metrelerce uzakta olan iki nesne elenir.

Asama B (model tabanli siniflandirici) bu modulun urettigi aday kliplerle
egitilir; Asama A onsuz da calisir.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from adgs.schema import Event, Party, Track


@dataclass
class KazaParam:
    """Tetikleme esikleri. Hepsi config'ten gelebilir; varsayilanlar CCTV icin."""

    iou_esik: float = 0.10
    # Cakismanin oncesi/sonrasi kac karelik pencerede hiz olculur
    pencere_kare: int = 5
    # Cakisma sonrasi hiz bu oranin altina duserse "ani yavaslama" sayilir
    hiz_dusus_orani: float = 0.5
    # Hareket yonu bu aciyi asarsa "ani yon degisimi" sayilir (derece)
    yon_degisim_derece: float = 40.0
    # Cakisma sonrasi bu kadar saniye hareketsizlik aranir
    hareketsizlik_sn: float = 2.0
    # Hareketsiz sayilmak icin merkez kaymasi bu pikselin altinda kalmali
    hareketsizlik_piksel: float = 4.0
    # Kalibrasyon varsa: yer duzleminde bu mesafeden uzak cift okluzyondur (m)
    maks_yer_mesafesi_m: float = 6.0
    # Cok kisa izler degerlendirmeye girmez
    min_iz_kare: int = 5


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    """Iki bounding box'in kesisim/birlesim orani."""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    kesisim = iw * ih
    if kesisim == 0:
        return 0.0
    alan_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    alan_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    birlesim = alan_a + alan_b - kesisim
    return kesisim / birlesim if birlesim > 0 else 0.0


def merkez(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def _yer_noktasi(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    x1, _y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def hareket_vektoru(iz: Track, f_bas: int, f_son: int) -> tuple[float, float] | None:
    """Iki kare arasindaki merkez kaymasi. Yeterli kare yoksa None."""
    kareler = [f for f in iz.frames if f_bas <= f <= f_son]
    if len(kareler) < 2:
        return None
    ilk, son = min(kareler), max(kareler)
    p1, p2 = merkez(iz.frames[ilk].bbox), merkez(iz.frames[son].bbox)
    return (p2[0] - p1[0], p2[1] - p1[1])


def hiz_piksel(iz: Track, f_bas: int, f_son: int) -> float | None:
    """Ortalama piksel/kare hizi. Olculemezse None (0 DEGIL - fark onemli)."""
    v = hareket_vektoru(iz, f_bas, f_son)
    if v is None:
        return None
    kareler = [f for f in iz.frames if f_bas <= f <= f_son]
    n = max(1, max(kareler) - min(kareler))
    return math.hypot(*v) / n


def yon_farki_derece(v1: tuple[float, float], v2: tuple[float, float]) -> float:
    """Iki hareket vektoru arasindaki aci (derece). Duran nesne icin 0."""
    n1, n2 = math.hypot(*v1), math.hypot(*v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return 0.0
    kos = (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)
    return math.degrees(math.acos(max(-1.0, min(1.0, kos))))


def ani_degisim_var_mi(iz: Track, f_carpisma: int, p: KazaParam) -> bool:
    """Sinyal 2: cakisma sonrasi ani yavaslama VEYA yon degisimi."""
    once_bas, once_son = f_carpisma - p.pencere_kare, f_carpisma
    sonra_bas, sonra_son = f_carpisma, f_carpisma + p.pencere_kare

    v_once = hiz_piksel(iz, once_bas, once_son)
    v_sonra = hiz_piksel(iz, sonra_bas, sonra_son)
    if v_once is None or v_sonra is None:
        return False

    # Zaten duran bir aracin "yavaslamasi" anlamsiz - once gercekten hareket etmeli.
    if v_once <= 1.0:
        return False
    if v_sonra < v_once * p.hiz_dusus_orani:
        return True

    d_once = hareket_vektoru(iz, once_bas, once_son)
    d_sonra = hareket_vektoru(iz, sonra_bas, sonra_son)
    if d_once and d_sonra:
        return yon_farki_derece(d_once, d_sonra) > p.yon_degisim_derece
    return False


def hareketsiz_mi(iz: Track, f_bas: int, fps: float, p: KazaParam) -> bool:
    """Sinyal 3: f_bas'tan itibaren N saniye boyunca merkez sabit kaldi mi."""
    gerekli = max(2, int(p.hareketsizlik_sn * fps))
    kareler = sorted(f for f in iz.frames if f >= f_bas)
    if len(kareler) < gerekli:
        return False  # yeterince veri yok - "hareketsiz" VARSAYILMAZ
    noktalar = [merkez(iz.frames[f].bbox) for f in kareler[:gerekli]]
    ref = noktalar[0]
    return all(math.dist(ref, n) <= p.hareketsizlik_piksel for n in noktalar)


def kaza_imzasi(iz: Track, f_carpisma: int, fps: float, p: KazaParam) -> bool:
    """Sinyal 2 ve 3'un AYNI izde birlikte gorulmesi.

    Ikisini ayri taraflarda aramak yanlis pozitif uretir: kavsakta park halinde
    duran bir arac "hareketsizlik" sinyalini bedavaya saglar ve yanindan gecen
    her aracla kaza uretirdi. Beklenmeyen hareketsizlik, ancak once hareket eden
    bir nesne icin anlamlidir.

    Hareketsizlik carpisma karesinden DEGIL, bir pencere sonrasindan olculur:
    arac carpma aninda henuz durmus olmaz, durmasi birkac kare surer.
    """
    return (ani_degisim_var_mi(iz, f_carpisma, p)
            and hareketsiz_mi(iz, f_carpisma + p.pencere_kare, fps, p))


def _okluzyon_mu(a: Track, b: Track, f: int, kalib, p: KazaParam) -> bool:
    """Kalibrasyon varsa: yer duzleminde uzak olan cift kaza degil, okluzyondur."""
    if kalib is None or not getattr(kalib, "gecerli", False):
        return False
    from adgs import calib as _calib

    mesafe = _calib.mesafe_m(kalib, _yer_noktasi(a.frames[f].bbox),
                             _yer_noktasi(b.frames[f].bbox))
    return mesafe is not None and mesafe > p.maks_yer_mesafesi_m


def _carpisma_karesi(a: Track, b: Track, p: KazaParam) -> int | None:
    """Iki izin IoU'sunun esigi ilk astigi ortak kare."""
    for f in sorted(set(a.frames) & set(b.frames)):
        if iou(a.frames[f].bbox, b.frames[f].bbox) >= p.iou_esik:
            return f
    return None


def tespit_et(
    izler: list[Track],
    fps: float,
    source_video: str,
    source_profile: str = "cctv_fixed",
    kalib=None,
    p: KazaParam | None = None,
) -> list[Event]:
    """Track listesinden kaza olaylari uretir (Asama A).

    Uc sinyalin ucu birden saglanmadikca Event URETILMEZ.
    """
    p = p or KazaParam()
    uygun = [t for t in izler if len(t.frames) >= p.min_iz_kare]
    olaylar: list[Event] = []
    sayac = 0

    for i, a in enumerate(uygun):
        for b in uygun[i + 1:]:
            f = _carpisma_karesi(a, b, p)
            if f is None:
                continue  # sinyal 1 yok
            if _okluzyon_mu(a, b, f, kalib, p):
                continue  # yer duzleminde uzaklar - okluzyon
            # Sinyal 2+3 ayni tarafta olmali (bkz. kaza_imzasi)
            if not (kaza_imzasi(a, f, fps, p) or kaza_imzasi(b, f, fps, p)):
                continue

            sayac += 1
            son = max(max(a.frames), max(b.frames))
            kareler: dict[int, list[int]] = {}
            for iz in (a, b):
                for kf, det in iz.frames.items():
                    if f - p.pencere_kare <= kf <= son:
                        kareler.setdefault(kf, list(det.bbox))
            olaylar.append(
                Event(
                    event_id=f"kaza_{sayac:04d}",
                    tip="KAZA",
                    alt_tip="CARPISMA",
                    t_start=max(0, f - p.pencere_kare) / fps,
                    t_end=son / fps,
                    frame_start=max(0, f - p.pencere_kare),
                    frame_end=son,
                    source_video=source_video,
                    source_profile=source_profile,
                    conf=round(iou(a.frames[f].bbox, b.frames[f].bbox), 3),
                    evidence={"keyframes": [f], "kareler": kareler},
                    parties=[Party(track_id=a.track_id), Party(track_id=b.track_id)],
                    notes=[
                        f"Kural tabanli tespit: iz #{a.track_id} ({a.cls}) ve "
                        f"#{b.track_id} ({b.cls}) kare {f}'de cakisti, ardindan "
                        f"ani hareket degisimi ve hareketsizlik gozlendi.",
                        "Bu tespit gorsel analize dayalidir; carpisma olup olmadigi "
                        "ve taraflarin kusuru yetkili merciler tarafindan belirlenir.",
                    ],
                )
            )
    return olaylar
