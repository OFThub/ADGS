"""M1 - Video alim ve on isleme."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


@dataclass
class Frame:
    idx: int  # kaynak videodaki gercek kare numarasi
    ts: float  # saniye
    image: np.ndarray
    gps: tuple[float, float] | None = None


def _resize(img: np.ndarray, long_edge: int) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= long_edge:
        return img
    s = long_edge / max(h, w)
    return cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)


def _clahe(img: np.ndarray) -> np.ndarray:
    """Dusuk isikta kontrast. LAB uzayinda L kanalina uygulanir ki renk kaymasin."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def iter_frames(
    path: str | Path,
    sample_fps: float = 2.0,
    target_long_edge: int = 1280,
    on_isleme: bool = False,
) -> Iterator[Frame]:
    """Videoyu deterministik kare akisina cevirir.

    Zaman damgasi her zaman CAP_PROP_POS_MSEC'ten okunur, kare sayisindan
    hesaplanmaz - degisken FPS'li videolarda kare indeksi != zaman.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"video acilamadi: {path}")

    src_fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    # src_fps okunamazsa her kareyi al; atlamak sessizce veri kaybettirmekten iyidir.
    step = max(1, round(src_fps / sample_fps)) if src_fps > 0 and sample_fps > 0 else 1

    try:
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            if idx % step == 0:
                ts = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                img = _resize(img, target_long_edge)
                if on_isleme:
                    img = _clahe(img)
                yield Frame(idx=idx, ts=ts, image=img)
            idx += 1
    finally:
        cap.release()


def video_meta(path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"video acilamadi: {path}")
    try:
        return {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        }
    finally:
        cap.release()
