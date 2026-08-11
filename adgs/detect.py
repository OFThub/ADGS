"""M2 - Tespit, siniflandirma, takip, (opsiyonel) plaka.

Sistemin TEK gercek kaynagi. M3 (kaza), M4 (ihlal) kendi tespitini yapmaz;
buradan cikan Track listesini okur. Bir aracin ID'si tum modullerde ayni olur -
kusur motorunun "17 numarali arac" diyebilmesi buna baglidir.

Tekillestirme icin ayri kod yazilmaz: model.track() BoT-SORT ile kalici ID verir.
Dunya koordinati YALNIZCA kalibrasyon dogrulamayi gectiyse doldurulur; gecmezse
world_xy None kalir ve hiza bagli moduller calisamaz.
"""

from __future__ import annotations

from pathlib import Path

from adgs.schema import Detection, Track

# COCO sinifi -> ADGS sinifi. COCO'da olmayan (plaka, tabela tipi) M2 disinda kalir.
COCO_ESLEME = {
    "car": "otomobil",
    "truck": "kamyon",
    "bus": "otobus",
    "motorcycle": "motosiklet",
    "bicycle": "bisiklet",
    "person": "yaya",
    "traffic light": "trafik_isigi",
    "stop sign": "trafik_isareti",
}

# Yer temas noktasi bu siniflar icin anlamlidir (tekerlek/ayak zemindedir).
_ZEMINDE = {"otomobil", "kamyon", "otobus", "motosiklet", "bisiklet", "yaya"}


def _zemin_noktasi(bbox: tuple[int, int, int, int]) -> tuple[float, float]:
    """Kutunun alt orta noktasi - aracin zeminle temas ettigi yer.

    Kutu merkezi kullanilirsa arac yuksekligi kadar hata girer; homografi yer
    duzlemi icin cozuldugunden merkez noktasi sistematik olarak uzaga duser.
    """
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2.0, float(y2))


def takip_et(
    video: str | Path,
    model_path: str | Path = "yolo26s.pt",
    kalib=None,
    conf: float = 0.35,
    tracker: str = "botsort.yaml",
    siniflar: set[str] | None = None,
) -> list[Track]:
    """Videoyu takip eder, kalici ID'li Track listesi dondurur.

    `kalib` verilir ve gecerliyse her karede dunya koordinati da doldurulur.
    Gecersiz kalibrasyon sessizce yok sayilmaz: world_xy None kalir.
    """
    from ultralytics import YOLO

    from adgs import calib as _calib

    model = YOLO(str(model_path))
    adlar = model.names
    istenen = siniflar or set(COCO_ESLEME.values())
    dunya_acik = kalib is not None and getattr(kalib, "gecerli", False)

    izler: dict[int, Track] = {}
    for f_idx, res in enumerate(
        model.track(source=str(video), persist=True, conf=conf, tracker=tracker,
                    stream=True, verbose=False)
    ):
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            continue
        for i, tid in enumerate(boxes.id.int().tolist()):
            ham = adlar[int(boxes.cls[i])]
            cls = COCO_ESLEME.get(ham)
            if cls is None or cls not in istenen:
                continue
            x1, y1, x2, y2 = (int(v) for v in boxes.xyxy[i])
            det = Detection(bbox=(x1, y1, x2, y2), cls=cls, conf=float(boxes.conf[i]))

            iz = izler.get(tid)
            if iz is None:
                iz = izler[tid] = Track(track_id=tid, cls=cls)
            iz.frames[f_idx] = det

            if dunya_acik and cls in _ZEMINDE:
                if iz.world_xy is None:
                    iz.world_xy = {}
                try:
                    iz.world_xy[f_idx] = _calib.piksel_to_dunya(
                        kalib.H, _zemin_noktasi(det.bbox)
                    )
                except ValueError:
                    pass  # ufuk cizgisindeki nokta - o kare icin dunya koordinati yok

    return sorted(izler.values(), key=lambda t: t.track_id)


def ozet(izler: list[Track]) -> dict[str, int]:
    """Sinif basina takip sayisi - CLI ciktisi ve hizli akil kontrolu icin."""
    sayim: dict[str, int] = {}
    for t in izler:
        sayim[t.cls] = sayim.get(t.cls, 0) + 1
    return dict(sorted(sayim.items()))


def kisa_izleri_ele(izler: list[Track], min_kare: int = 3) -> list[Track]:
    """Cok kisa izleri atar.

    1-2 karelik izler neredeyse her zaman yanlis pozitif veya ID sicramasidir;
    kusur/ihlal degerlendirmesine girmeleri yanlis kayit uretir.
    """
    return [t for t in izler if len(t.frames) >= min_kare]
