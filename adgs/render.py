"""M9 - Isaretlenmis video ve JSON rapor."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import cv2

from adgs.schema import Event

_RENK = {"YUKSEK": (0, 0, 220), "ORTA": (0, 165, 255), "DUSUK": (0, 200, 200)}
_VARSAYILAN_RENK = (0, 200, 0)
_FILIGRAN = "ON DEGERLENDIRME - BAGLAYICI DEGILDIR"


def _siddet_of(evt: Event) -> str:
    for n in evt.notes:
        for s in ("YUKSEK", "ORTA", "DUSUK"):
            if f"siddet {s}" in n:
                return s
    return ""


def _etiket(evt: Event) -> str:
    s = _siddet_of(evt)
    return f"{evt.event_id} {evt.alt_tip}" + (f" [{s}]" if s else "")


def annotate_video(video: str | Path, events: list[Event], out_path: str | Path) -> Path:
    """Olaylarin bbox'larini kaynak videoya cizip yeni bir MP4 yazar."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # frame_idx -> [(bbox, etiket, renk)]
    plan: dict[int, list] = {}
    watermark = False
    for evt in events:
        renk = _RENK.get(_siddet_of(evt), _VARSAYILAN_RENK)
        etiket = _etiket(evt)
        watermark |= evt.tip in ("KAZA", "IHLAL")
        for f, bbox in (evt.evidence.get("kareler") or {}).items():
            plan.setdefault(int(f), []).append((bbox, etiket, renk))

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"video acilamadi: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            for bbox, etiket, renk in plan.get(idx, []):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), renk, 2)
                cv2.putText(img, etiket, (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 1, cv2.LINE_AA)
            if watermark:
                cv2.putText(img, _FILIGRAN, (8, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(img)
            idx += 1
    finally:
        cap.release()
        writer.release()
    return out_path


def save_clip(video: str | Path, frame_start: int, frame_end: int,
              out_path: str | Path, tampon: int = 15) -> Path | None:
    """Olay araligini ayri bir MP4 olarak kesip yazar. Kare yazilamazsa None.

    ponytail: OpenCV ile yeniden kodlar (kalite/hiz kaybi). ffmpeg -ss ile
    kodlamadan kesmek daha iyi olurdu; ffmpeg zorunlu bagimlilik olmasin diye
    gecildi - klip sayisi artarsa ffmpeg'e gecilmeli.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"video acilamadi: {video}")
    yazilan = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        bas = max(0, frame_start - tampon)
        son = frame_end + tampon
        cap.set(cv2.CAP_PROP_POS_FRAMES, bas)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                 fps, (w, h))
        try:
            for _ in range(son - bas + 1):
                ok, img = cap.read()
                if not ok:
                    break
                cv2.putText(img, _FILIGRAN, (8, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(img)
                yazilan += 1
        finally:
            writer.release()
    finally:
        cap.release()
    if yazilan == 0:
        out_path.unlink(missing_ok=True)
        return None
    return out_path


# Sinif basina sabit renk - ayni arac tum videoda ayni renkte gorunur.
_SINIF_RENK = {
    "otomobil": (0, 200, 0), "kamyon": (0, 140, 255), "otobus": (0, 200, 200),
    "motosiklet": (255, 120, 0), "bisiklet": (255, 200, 0), "yaya": (200, 0, 200),
    "trafik_isigi": (255, 255, 255), "trafik_isareti": (180, 180, 180),
}


def annotate_tracks(video: str | Path, izler: list, out_path: str | Path,
                    kalib=None) -> Path:
    """M2 ciktisini (Track listesi) ID etiketleriyle videoya cizer."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plan: dict[int, list] = {}
    for iz in izler:
        renk = _SINIF_RENK.get(iz.cls, _VARSAYILAN_RENK)
        etiket = f"#{iz.track_id} {iz.cls}"
        if iz.plate:
            etiket += f" {iz.plate}"
        for f, det in iz.frames.items():
            plan.setdefault(int(f), []).append((det.bbox, etiket, renk))

    # Kalibrasyon gecersizse bunu izleyicinin gormesi gerekir: bu videodaki
    # hicbir mesafe/hiz cikarimi gecerli degildir.
    kalib_notu = ""
    if kalib is not None and not getattr(kalib, "gecerli", False):
        kalib_notu = "KALIBRASYON YOK - MESAFE/HIZ HESAPLANMADI"

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"video acilamadi: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    try:
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            for bbox, etiket, renk in plan.get(idx, []):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), renk, 2)
                cv2.putText(img, etiket, (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 1, cv2.LINE_AA)
            cv2.putText(img, _FILIGRAN, (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if kalib_notu:
                cv2.putText(img, kalib_notu, (8, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            writer.write(img)
            idx += 1
    finally:
        cap.release()
        writer.release()
    return out_path


def write_tracks(izler: list, out_path: str | Path, source: str = "",
                 kalib=None) -> Path:
    """M2 takip ciktisini JSON'a yazar (Faz 2 teslimati)."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kaynak": source,
        "kamera": getattr(kalib, "camera_id", None),
        "kalibrasyon": {
            "gecerli": bool(getattr(kalib, "gecerli", False)),
            "hata_yuzde": getattr(kalib, "hata_yuzde", None),
            "notlar": list(getattr(kalib, "notlar", [])),
        },
        "iz_sayisi": len(izler),
        "uyari": "Bu cikti bir karar destek analizidir; baglayici bir tespit degildir.",
        "izler": [
            {
                "track_id": iz.track_id,
                "cls": iz.cls,
                "kare_sayisi": len(iz.frames),
                "ilk_kare": min(iz.frames) if iz.frames else None,
                "son_kare": max(iz.frames) if iz.frames else None,
                "plaka": iz.plate,
                "plaka_conf": iz.plate_conf,
                "dunya_koordinatli_kare": len(iz.world_xy or {}),
            }
            for iz in izler
        ],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path


def write_report(events: list[Event], out_path: str | Path, source: str = "",
                 uyarilar: list[str] | None = None) -> Path:
    """Olaylari JSON'a yazar. Kanitsiz olaylar rapora GIRMEZ.

    `uyarilar` rapora yazilir cunku "0 ihlal" ile "3 dedektor calismayi
    reddetti, 0 ihlal" ayni sey degildir. Ikincisini gizlemek raporu yaniltici
    yapar - okuyan kisi bakilmayan seyleri "yok" sanir.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kanitli = [e for e in events if e.kanitli_mi()]
    payload = {
        "kaynak": source,
        "olay_sayisi": len(kanitli),
        "kanitsiz_atlanan": len(events) - len(kanitli),
        "calisamayan_moduller": list(uyarilar or []),
        "uyari": "Bu rapor bir karar destek ciktisidir; baglayici bir tespit degildir.",
        "events": [asdict(e) for e in kanitli],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
