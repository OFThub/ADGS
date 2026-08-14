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
    # Sahne kesmesinin bu kadar yakinindaki cakisma DEGERLENDIRILMEZ. Kesmede
    # izler kopar, farkli sahnelerin kutulari cakisir ve konum sicramasi
    # "ani hareket degisimi"ni taklit eder - hepsi sahte kaza uretir.
    kesme_tampon_kare: int = 15
    # Bu kadar kare arayla suren cakismalar TEK epizot sayilir. Ayni carpismanin
    # ardisik kareleri tek olay olsun, ayri zamanlardaki iki olay ayri kalsin.
    epizot_bosluk_kare: int = 30


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


def _carpisma_kareleri(a: Track, b: Track, p: KazaParam) -> list[int]:
    """Iki izin cakisma EPIZOTLARININ baslangic kareleri.

    Onceki hali yalnizca ILK cakisma karesini donuyordu; uzun videolarda bu
    iki ayri hata uretiyordu:

      1. Ayni iki arac once yan yana gecip DAHA SONRA carpissa yalnizca ilk
         (masum) gecis sinaniyor, gercek kaza HIC gorulmuyordu.
      2. Bir cift en fazla TEK olay uretebiliyordu; 5 dakikalik bir kayitta
         ayni araclarin iki ayri olayi tek olaya iniyordu.

    Ardisik cakisma kareleri tek epizot sayilir; arada `epizot_bosluk_kare`den
    uzun bosluk varsa yeni epizot baslar - boylece tek carpismanin onlarca
    karesi onlarca olay uretmez.
    """
    ortak = sorted(set(a.frames) & set(b.frames))
    cakisan = [f for f in ortak
               if iou(a.frames[f].bbox, b.frames[f].bbox) >= p.iou_esik]
    epizotlar: list[list[int]] = []
    for f in cakisan:
        if epizotlar and f - epizotlar[-1][1] <= p.epizot_bosluk_kare:
            epizotlar[-1][1] = f
        else:
            epizotlar.append([f, f])
    return [e[0] for e in epizotlar]


def _kesmeye_yakin(f: int, kesmeler: set[int] | None, p: KazaParam) -> bool:
    """Kare bir sahne kesmesinin tampon araliginda mi."""
    if not kesmeler:
        return False
    return any(abs(f - k) <= p.kesme_tampon_kare for k in kesmeler)


def tani(izler: list[Track], fps: float, p: KazaParam | None = None,
         kesmeler: set[int] | None = None) -> dict:
    """Aday ciftlerin HANGI SINYALDE elendigini sayar.

    "1 kaza bulundu" ciktisinin sebebini tahmin etmek yerine olcmek icin.
    Esik gevsetmek yanlis pozitif uretir; hangi sinyalin ne kadar eledigi
    bilinmeden dokunulmamali.
    """
    p = p or KazaParam()
    uygun = [t for t in izler if len(t.frames) >= p.min_iz_kare]
    d = {"iz": len(izler), "uygun_iz": len(uygun), "cift": 0, "cakisan_cift": 0,
         "epizot": 0, "kesme_elendi": 0, "okluzyon_elendi": 0,
         "hareket_sinyali_yok": 0, "hareketsizlik_yok": 0, "veri_yetersiz": 0,
         "kabul": 0}
    for i, a in enumerate(uygun):
        for b in uygun[i + 1:]:
            d["cift"] += 1
            epizotlar = _carpisma_kareleri(a, b, p)
            if epizotlar:
                d["cakisan_cift"] += 1
            d["epizot"] += len(epizotlar)
            for f in epizotlar:
                if _kesmeye_yakin(f, kesmeler, p):
                    d["kesme_elendi"] += 1
                    continue
                if _okluzyon_mu(a, b, f, None, p):
                    d["okluzyon_elendi"] += 1
                    continue
                if not (ani_degisim_var_mi(a, f, p) or ani_degisim_var_mi(b, f, p)):
                    d["hareket_sinyali_yok"] += 1
                    continue
                if kaza_imzasi(a, f, fps, p) or kaza_imzasi(b, f, fps, p):
                    d["kabul"] += 1
                    continue
                # Hareket sinyali var ama hareketsizlik yok. Veri mi bitti,
                # yoksa arac gercekten hareketine devam mi etti?
                gerekli = max(2, int(p.hareketsizlik_sn * fps))
                yeter = any(len([k for k in iz.frames if k >= f + p.pencere_kare])
                            >= gerekli for iz in (a, b))
                d["hareketsizlik_yok" if yeter else "veri_yetersiz"] += 1
    return d


def tani_metni(izler: list[Track], fps: float, p: KazaParam | None = None,
               kesmeler: set[int] | None = None) -> str:
    p = p or KazaParam()
    d = tani(izler, fps, p, kesmeler)
    return "\n".join([
        f"iz: {d['iz']}  (>= {p.min_iz_kare} kare olan: {d['uygun_iz']})",
        f"cift: {d['cift']}  cakisan cift: {d['cakisan_cift']}  "
        f"cakisma epizodu: {d['epizot']}",
        "",
        f"  sahne kesmesi (montaj)         : {d['kesme_elendi']}",
        f"  okluzyon (yer duzleminde uzak) : {d['okluzyon_elendi']}",
        f"  ani hareket degisimi YOK       : {d['hareket_sinyali_yok']}",
        f"  hareketsizlik YOK (devam etti) : {d['hareketsizlik_yok']}",
        f"  hareketsizlik OLCULEMEDI       : {d['veri_yetersiz']}"
        "   <- iz kesildi / kare yetmedi",
        f"  KABUL EDILEN                   : {d['kabul']}",
    ])


def tespit_et(
    izler: list[Track],
    fps: float,
    source_video: str,
    source_profile: str = "cctv_fixed",
    kalib=None,
    p: KazaParam | None = None,
    kesmeler: set[int] | None = None,
) -> list[Event]:
    """Track listesinden kaza olaylari uretir (Asama A).

    Uc sinyalin ucu birden saglanmadikca Event URETILMEZ.

    `kesmeler` (probe.sahne_kesmeleri) verilirse kesmeye yakin cakismalar
    degerlendirilmez - derleme videoda kesme, kaza imzasini taklit eder.
    """
    p = p or KazaParam()
    uygun = [t for t in izler if len(t.frames) >= p.min_iz_kare]
    olaylar: list[Event] = []
    sayac = 0

    for i, a in enumerate(uygun):
        for b in uygun[i + 1:]:
            for f in _carpisma_kareleri(a, b, p):
                if _kesmeye_yakin(f, kesmeler, p):
                    continue  # sahne kesmesi - kaza degil, montaj
                if _okluzyon_mu(a, b, f, kalib, p):
                    continue  # yer duzleminde uzaklar - okluzyon
                # Sinyal 2+3 ayni tarafta olmali (bkz. kaza_imzasi)
                if not (kaza_imzasi(a, f, fps, p) or kaza_imzasi(b, f, fps, p)):
                    continue

                sayac += 1
                # Olay carpismanin ETRAFIYLA sinirli. Onceki hali izlerin en son
                # karesini aliyordu: 5 dakika boyunca goruntude kalan bir arac
                # 5 dakikalik "kaza" uretiyor, klip ve kutu anlamsizlasiyordu.
                bit = f + p.pencere_kare + int(p.hareketsizlik_sn * fps)
                son = min(max(max(a.frames), max(b.frames)), bit)
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
                        parties=[Party(track_id=a.track_id),
                                 Party(track_id=b.track_id)],
                        notes=[
                            f"Kural tabanli tespit: iz #{a.track_id} ({a.cls}) ve "
                            f"#{b.track_id} ({b.cls}) kare {f}'de ({f / fps:.1f} sn) "
                            "cakisti, ardindan ani hareket degisimi ve "
                            "hareketsizlik gozlendi.",
                            "Bu tespit gorsel analize dayalidir; carpisma olup "
                            "olmadigi ve taraflarin kusuru yetkili merciler "
                            "tarafindan belirlenir.",
                        ],
                    )
                )
    # Zamana gore sirala: olaylar cift dongusunden cift sirasiyla cikiyordu,
    # 5 dakikalik bir kayitta liste zaman disi gorunuyordu.
    olaylar.sort(key=lambda e: e.frame_start)
    for n, e in enumerate(olaylar, 1):
        e.event_id = f"kaza_{n:04d}"
    return olaylar
