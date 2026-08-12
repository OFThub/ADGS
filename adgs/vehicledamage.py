"""M8 - Arac hasar degerlendirmesi (CarDD).

CIKTININ SINIRI, MODULUN KENDISI KADAR ONEMLIDIR.

CarDD yakin cekim, temiz ve yuksek cozunurluklu fotograflardan olusur. Sabit
CCTV ise araci 6-10 m yukseklikten, cogu zaman 60x40 piksellik bir kutu icinde
gorur. O boyutta "cizik mi gocuk mu" sorusunun goruntude cevabi YOKTUR - ama
model yine de bir sinif uretir ve bu uydurulmus sinif hasar raporuna girer.

Bu yuzden modul bir COZUNURLUK KAPISI uygular: aracin kirpmasi esigin altindaysa
sinif bazli cikti URETILMEZ, yalnizca "degerlendirilemedi" denir ve sebebi
yazilir. Faz 4'teki "geometri yoksa reddet" disiplininin aynisi.

Bolge (on/arka/yan) aracin YONUNU gerektirir. Yon, izin hareket vektorunden
cikarilir: arac hareket yonune dogru gider, yani onu oradadir. Duran arac icin
yon bilinemez ve bolge None kalir - "on" VARSAYILMAZ.

ALTYAPI hasarinin siddeti burada YENIDEN HESAPLANMAZ. M5 (roaddamage) onu zaten
uretir; ikinci bir uygulama, iki farkli cevap veren iki gercek kaynagi olurdu.
"""

from __future__ import annotations

import math
from pathlib import Path

from adgs.schema import Event, Track

# CarDD sinif adlari (COCO kategorileri, veri kumesindeki sirayla).
CARDD_CLASSES = ["dent", "scratch", "crack", "glass shatter", "lamp broken", "tire flat"]
CARDD_LABELS = {
    "dent": "Gocuk",
    "scratch": "Cizik",
    "crack": "Catlak",
    "glass shatter": "Cam kirilmasi",
    "lamp broken": "Far kirik",
    "tire flat": "Lastik patlak",
}

# Yapisal agirlik: catlak > gocuk > cizik (CarDD'nin kendi oncelik sirasi).
# Cam kirilmasi ve lastik patlak da yapisaldir - kucuk alan kaplasalar bile
# aracin kullanilabilirligini etkiler.
_AGIRLIK = {
    "crack": 3, "glass shatter": 3, "tire flat": 3,
    "dent": 2, "lamp broken": 2,
    "scratch": 1,
}
_YAPISAL = {"crack", "glass shatter", "tire flat"}

# Arac kutusunun uzun kenari bu pikselin altindaysa sinif bazli hasar ciktisi
# GUVENILMEZ. Deger CarDD'nin egitim olceginden degil, cozunurluk calismasindan
# gelir: bkz. training/cardd_cozunurluk_etkisi.py
MIN_KENAR_PIKSEL = 128

# Hasar alaninin arac kutusuna orani.
_ORTA_ALAN = 0.05
_AGIR_ALAN = 0.15


def siddet(tipler: list[str], alan_orani: float) -> str:
    """HAFIF | ORTA | AGIR.

    Yapisal hasar (catlak, cam, lastik) kucuk alan kaplasa da ORTA'nin altina
    dusmez - bir cam kirigi, genis bir cizikten daha agir bir sonuctur.
    """
    if alan_orani >= _AGIR_ALAN:
        return "AGIR"
    if alan_orani >= _ORTA_ALAN or any(t in _YAPISAL for t in tipler):
        return "ORTA"
    return "HAFIF"


def _yon(iz: Track, f: int, pencere: int = 5) -> tuple[float, float] | None:
    """Izin kare f civarindaki hareket yonu (birim vektor). Duruyorsa None."""
    kareler = sorted(iz.frames)
    onceki = [k for k in kareler if k <= f]
    sonraki = [k for k in kareler if k > f]
    if not onceki or not sonraki:
        return None
    a = onceki[-1]
    b = min(sonraki[-1], a + pencere)
    if b <= a:
        return None
    ax1, ay1, ax2, ay2 = iz.frames[a].bbox
    bx1, by1, bx2, by2 = iz.frames[b].bbox
    d = ((bx1 + bx2 - ax1 - ax2) / 2.0, (by1 + by2 - ay1 - ay2) / 2.0)
    n = math.hypot(*d)
    return (d[0] / n, d[1] / n) if n > 1.0 else None


def bolge(hasar_merkezi: tuple[float, float], arac_bbox: tuple[int, int, int, int],
          yon: tuple[float, float] | None) -> str | None:
    """Hasarin araca gore bolgesi. Yon bilinmiyorsa None - "on" VARSAYILMAZ.

    Hasarin arac merkezine gore kaymasi hareket yonune izdusurulur: hareket
    yonundeki bilesen buyukse on/arka, yanal bilesen buyukse yan.
    """
    if yon is None:
        return None
    x1, y1, x2, y2 = arac_bbox
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    d = (hasar_merkezi[0] - mx, hasar_merkezi[1] - my)
    ileri = d[0] * yon[0] + d[1] * yon[1]
    yanal = d[0] * -yon[1] + d[1] * yon[0]
    if abs(ileri) >= abs(yanal):
        return "on" if ileri > 0 else "arka"
    return "yan"


def _yolo_dedektor(model_path: str | Path, conf: float):
    """CarDD agirligindan (kirpma -> tespit listesi) fonksiyonu uretir."""
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    adlar: dict[int, str] = model.names
    bilinmeyen = set(adlar.values()) - set(CARDD_CLASSES)
    if bilinmeyen:
        # Yanlis model verilirse sessiz yanlis etiketleme olusurdu (M5'te ayni
        # tuzak vardi) - erken ve yuksek sesle patlat.
        raise ValueError(
            f"Model arac hasari modeli degil - beklenmeyen siniflar: "
            f"{sorted(bilinmeyen)}. Beklenen: {CARDD_CLASSES}. CarDD ile ince "
            "ayar yapilmis agirlik verin."
        )

    def dedektor(crop):
        cikti = []
        for res in model.predict(source=crop, conf=conf, verbose=False):
            boxes = res.boxes
            if boxes is None:
                continue
            for i in range(len(boxes)):
                x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i])
                cikti.append((adlar[int(boxes.cls[i])], (x1, y1, x2, y2),
                              float(boxes.conf[i])))
        return cikti

    return dedektor


def _kirp(img, bbox: tuple[int, int, int, int]):
    """Kutuyu kare sinirlarina kirpar. Tasan kutu bos dizi uretirdi."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = bbox
    x1, y1 = max(0, min(int(x1), w - 1)), max(0, min(int(y1), h - 1))
    x2, y2 = max(x1 + 1, min(int(x2), w)), max(y1 + 1, min(int(y2), h))
    return img[y1:y2, x1:x2], (x1, y1, x2, y2)


def _kareyi_al(video: str | Path, f: int):
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"video acilamadi: {video}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, f))
        ok, img = cap.read()
        return img if ok else None
    finally:
        cap.release()


def degerlendir(
    video: str | Path,
    events: list[Event],
    izler: list[Track],
    model_path: str | Path | None = None,
    conf: float = 0.35,
    min_kenar: int = MIN_KENAR_PIKSEL,
    dedektor=None,
    kare_getir=None,
) -> list[str]:
    """KAZA olaylarinin taraflarina `hasar` alanini doldurur.

    `dedektor` enjekte edilebilir (kirpma -> [(sinif, bbox, conf)]); verilmezse
    model_path'ten YOLO dedektoru kurulur. `kare_getir` de enjekte edilebilir.
    Ikisi de modulun GPU'suz ve videosuz test edilebilmesi icindir.

    Donen liste, calisilamayan durumlarin sebepleridir - CLI bunlari basar.
    """
    kazalar = [e for e in events if e.tip == "KAZA"]
    if not kazalar:
        return []

    uyarilar: list[str] = []
    if dedektor is None:
        if model_path is None or not Path(model_path).exists():
            uyarilar.append(
                f"M8 arac hasari: model bulunamadi ({model_path}) -> "
                "HASAR DEGERLENDIRMESI YAPILMADI"
            )
            return uyarilar
        dedektor = _yolo_dedektor(model_path, conf)
    kare_getir = kare_getir or (lambda f: _kareyi_al(video, f))

    iz_map = {t.track_id: t for t in izler}
    dusuk_cozunurluk = 0

    for evt in kazalar:
        keyframes = evt.evidence.get("keyframes") or [evt.frame_start]
        f = int(keyframes[0])
        img = kare_getir(f)
        if img is None:
            uyarilar.append(f"{evt.event_id}: kare {f} okunamadi - hasara bakilmadi")
            continue

        for taraf in evt.parties:
            iz = iz_map.get(taraf.track_id)
            if iz is None or f not in iz.frames:
                taraf.hasar = {
                    "guvenilir": False, "tipler": [], "bolge": None, "siddet": None,
                    "sebep": f"iz #{taraf.track_id} kare {f}'de bulunamadi",
                }
                continue

            crop, arac_bbox = _kirp(img, iz.frames[f].bbox)
            kh, kw = crop.shape[:2]

            # COZUNURLUK KAPISI - bu esigin altinda sinif uretmek uydurmaktir.
            if max(kh, kw) < min_kenar:
                dusuk_cozunurluk += 1
                taraf.hasar = {
                    "guvenilir": False, "tipler": [], "bolge": None, "siddet": None,
                    "sebep": (
                        f"arac kirpmasi {kw}x{kh} piksel; sinif bazli hasar "
                        f"degerlendirmesi icin uzun kenar >= {min_kenar} gerekir"
                    ),
                }
                continue

            tespitler = dedektor(crop)
            if not tespitler:
                taraf.hasar = {
                    "guvenilir": True, "tipler": [], "etiketler": [], "bolge": None,
                    "siddet": None, "alan_orani": 0.0, "keyframe": f,
                    "sebep": "gorulebilir hasar tespit edilmedi",
                }
                continue

            arac_alan = float(kw * kh)
            toplam = 0.0
            tipler: list[str] = []
            en_agir: tuple[str, tuple[float, float]] | None = None
            for cls_adi, (dx1, dy1, dx2, dy2), _c in tespitler:
                toplam += max(0, dx2 - dx1) * max(0, dy2 - dy1)
                if cls_adi not in tipler:
                    tipler.append(cls_adi)
                if en_agir is None or _AGIRLIK.get(cls_adi, 0) > _AGIRLIK.get(en_agir[0], 0):
                    en_agir = (cls_adi, ((dx1 + dx2) / 2.0, (dy1 + dy2) / 2.0))

            alan_orani = min(1.0, toplam / arac_alan) if arac_alan > 0 else 0.0
            sirali = sorted(tipler, key=lambda t: -_AGIRLIK.get(t, 0))
            merkez_kare = (arac_bbox[0] + en_agir[1][0], arac_bbox[1] + en_agir[1][1])
            s = siddet(tipler, alan_orani)
            taraf.hasar = {
                "guvenilir": True,
                "tipler": sirali,
                "etiketler": [CARDD_LABELS.get(t, t) for t in sirali],
                "bolge": bolge(merkez_kare, arac_bbox, _yon(iz, f)),
                "siddet": s,
                "alan_orani": round(alan_orani, 4),
                "keyframe": f,
                "sebep": None,
            }
            evt.notes.append(
                f"Iz #{taraf.track_id}: gorulebilir hasar "
                f"({', '.join(CARDD_LABELS.get(t, t) for t in sirali)}), siddet {s}. "
                "Hasar tipi ve siddeti goruntuden tahmindir; eksper tespiti yerine "
                "gecmez."
            )

    if dusuk_cozunurluk:
        uyarilar.append(
            f"M8 arac hasari: {dusuk_cozunurluk} tarafta arac kirpmasi "
            f"{min_kenar} pikselin altinda - sinif bazli hasar URETILMEDI "
            "(CarDD yakin cekim verisidir, bkz. vehicledamage modul basligi)"
        )
    return uyarilar
