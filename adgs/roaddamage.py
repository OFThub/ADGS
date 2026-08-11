"""M5 - Yol/altyapi hasari tespiti.

RDD2022 bounding-box etiketlidir (PASCAL VOC), segmentasyon maskesi icermez.
Bu yuzden siddet skoru maske alanindan degil bbox alanindan hesaplanir.
"""

from __future__ import annotations

from pathlib import Path

from adgs.schema import Event

# RDD2022 sinif kodlari. Indeks eslemesi modelin kendi names alanindan gelir;
# bu liste yalnizca dogrulama icin kullanilir (bkz. detect).
RDD_CLASSES = ["D00", "D10", "D20", "D40"]
RDD_LABELS = {
    "D00": "Boyuna catlak",
    "D10": "Enine catlak",
    "D20": "Timsah sirti catlak",
    "D40": "Cukur",
}
# Yapisal bozulma gostergesi - tespit edilirse en az ORTA siddet alir.
YAPISAL = {"D20", "D40"}

# Kare alaninin yuzdesi olarak esikler. Sabit kamera yuksekligi yoksa gercek
# m2'ye cevrilemez; bunlar goreli buyukluk esikleridir.
# ponytail: goreli alan esigi; kalibrasyon eklenirse m2 esigine gecilir.
_ORTA_ALAN = 0.010
_YUKSEK_ALAN = 0.035


def siddet(cls_kodu: str, alan_orani: float) -> str:
    """Hasar siddeti. Yapisal hasarlar kucuk olsa da ORTA'nin altina dusmez."""
    if alan_orani >= _YUKSEK_ALAN:
        return "YUKSEK"
    if alan_orani >= _ORTA_ALAN or cls_kodu in YAPISAL:
        return "ORTA"
    return "DUSUK"


def oncelik(siddet_seviyesi: str, cls_kodu: str) -> int:
    """Is emri onceligi 1 (acil) - 3 (planli)."""
    if siddet_seviyesi == "YUKSEK" or cls_kodu == "D40":
        return 1
    return 2 if siddet_seviyesi == "ORTA" else 3


def detect(
    video: str | Path,
    model_path: str | Path,
    conf: float = 0.35,
    source_profile: str = "vehicle_mounted",
    gps_track: dict[int, tuple[float, float]] | None = None,
) -> list[Event]:
    """Videoda yol hasari tespit eder, track_id basina TEK Event dondurur.

    Tekillestirme icin ayri kumeleme kodu yazilmaz: model.track() BoT-SORT ile
    kalici ID verir, ayni cukur kac karede gorunurse gorunsun tek ID alir.
    """
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    # Sinif adini indeksten degil modelin kendi adlarindan coz: yanlis model
    # verilirse IndexError yerine sessiz yanlis etiketleme olusurdu.
    adlar: dict[int, str] = model.names
    bilinmeyen = set(adlar.values()) - set(RDD_CLASSES)
    if bilinmeyen:
        raise ValueError(
            f"Model yol hasari modeli degil - beklenmeyen siniflar: {sorted(bilinmeyen)}. "
            f"Beklenen: {RDD_CLASSES}. RDD2022 ile ince ayar yapilmis agirlik verin."
        )

    # Her track_id icin en yuksek guvenli kareyi kanit olarak sakla.
    best: dict[int, dict] = {}

    for f_idx, res in enumerate(
        model.track(source=str(video), persist=True, conf=conf, stream=True, verbose=False)
    ):
        boxes = res.boxes
        if boxes is None or boxes.id is None:
            continue
        fh, fw = res.orig_shape
        frame_area = fh * fw
        for i, tid in enumerate(boxes.id.int().tolist()):
            c = float(boxes.conf[i])
            x1, y1, x2, y2 = (float(v) for v in boxes.xyxy[i])
            alan_orani = ((x2 - x1) * (y2 - y1)) / frame_area
            cls_kodu = adlar[int(boxes.cls[i])]
            rec = best.get(tid)
            if rec is None:
                best[tid] = {
                    "cls": cls_kodu, "conf": c, "alan": alan_orani,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "ilk": f_idx, "son": f_idx, "key": f_idx,
                    "kareler": {f_idx: [int(x1), int(y1), int(x2), int(y2)]},
                }
            else:
                rec["son"] = f_idx
                rec["kareler"][f_idx] = [int(x1), int(y1), int(x2), int(y2)]
                if c > rec["conf"]:  # kanit karesi en guvenli tespit olsun
                    rec.update(cls=cls_kodu, conf=c, alan=alan_orani, key=f_idx,
                               bbox=[int(x1), int(y1), int(x2), int(y2)])

    events: list[Event] = []
    for tid, r in sorted(best.items()):
        s = siddet(r["cls"], r["alan"])
        events.append(
            Event(
                event_id=f"altyapi_{tid:04d}",
                tip="ALTYAPI",
                alt_tip=r["cls"],
                t_start=float(r["ilk"]),
                t_end=float(r["son"]),
                frame_start=r["ilk"],
                frame_end=r["son"],
                source_video=str(video),
                source_profile=source_profile,
                conf=round(r["conf"], 3),
                evidence={"keyframes": [r["key"]], "bbox": r["bbox"],
                          "kareler": r["kareler"]},
                gps=(gps_track or {}).get(r["key"]),
                notes=[
                    f"{RDD_LABELS[r['cls']]} ({r['cls']}) - siddet {s}, "
                    f"oncelik {oncelik(s, r['cls'])}",
                    f"Kare alaninin %{r['alan'] * 100:.2f}'si. Kamera kalibrasyonu "
                    f"olmadigi icin gercek alan (m2) hesaplanmamistir.",
                ],
            )
        )
    return events
