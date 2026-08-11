"""RDD2022 (PASCAL VOC) -> YOLO formati donusturucu + train/val ayrimi.

RDD2022 bbox etiketlidir (segmentasyon yok). Norvec alt kumesi 9.9 GB ve
ultra-genis (3650x2044) karelerden olusur - dashcam dagilimindan uzak oldugu
icin varsayilan olarak haric tutulur.

Indirme: https://figshare.com/articles/dataset/21431547  (CC BY-SA 4.0)
Kullanim: python training/prep_rdd2022.py --src data/raw/RDD2022 --out data/rdd_yolo
"""

from __future__ import annotations

import argparse
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

CLASSES = ["D00", "D10", "D20", "D40"]
CLS_IDX = {c: i for i, c in enumerate(CLASSES)}
# Hedef profil arac kamerasi (yol seviyesi, ileri bakis). Bu iki alt kume
# farkli goruntuleme geometrisine sahip oldugu icin varsayilan olarak haric:
#   Norway      - 10.6 GB, ultra-genis 3650x2044 kare
#   China_Drone - drone, tepeden bakis
HARIC = {"Norway", "China_Drone"}


def voc_to_yolo(xml_path: Path) -> list[str]:
    """Bir VOC XML'i YOLO satirlarina cevirir. Bilinmeyen sinif atlanir."""
    root = ET.parse(xml_path).getroot()
    size = root.find("size")
    w, h = int(size.find("width").text), int(size.find("height").text)
    if w <= 0 or h <= 0:
        return []
    lines = []
    for obj in root.findall("object"):
        name = (obj.find("name").text or "").strip()
        if name not in CLS_IDX:
            continue
        b = obj.find("bndbox")
        x1, y1 = float(b.find("xmin").text), float(b.find("ymin").text)
        x2, y2 = float(b.find("xmax").text), float(b.find("ymax").text)
        x1, x2 = sorted((max(0.0, x1), min(float(w), x2)))
        y1, y2 = sorted((max(0.0, y1), min(float(h), y2)))
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        lines.append(
            f"{CLS_IDX[name]} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} "
            f"{(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}"
        )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, type=Path, help="Cikartilmis RDD2022 kok dizini")
    ap.add_argument("--out", default=Path("data/rdd_yolo"), type=Path)
    ap.add_argument("--val-split", default=0.15, type=float)
    ap.add_argument(
        "--include-all", action="store_true",
        help=f"Haric tutulan alt kumeleri de dahil et: {sorted(HARIC)}",
    )
    ap.add_argument("--seed", default=42, type=int)
    a = ap.parse_args()

    haric = set() if a.include_all else HARIC
    pairs: list[tuple[Path, Path]] = []
    for xml in a.src.rglob("annotations/xmls/*.xml"):
        if any(h in xml.parts for h in haric):
            continue
        img = xml.parent.parent.parent / "images" / f"{xml.stem}.jpg"
        if img.exists():
            pairs.append((img, xml))

    if not pairs:
        print(f"HATA: {a.src} altinda annotations/xmls/*.xml bulunamadi")
        return 1

    random.Random(a.seed).shuffle(pairs)
    n_val = int(len(pairs) * a.val_split)
    splits = {"val": pairs[:n_val], "train": pairs[n_val:]}

    bos = 0
    for split, items in splits.items():
        (a.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        for img, xml in items:
            lines = voc_to_yolo(xml)
            if not lines:
                bos += 1  # negatif ornek: etiketsiz kalir, YOLO bunu arka plan sayar
            shutil.copy2(img, a.out / "images" / split / img.name)
            (a.out / "labels" / split / f"{img.stem}.txt").write_text(
                "\n".join(lines), encoding="utf-8"
            )

    (a.out / "rdd2022.yaml").write_text(
        f"path: {a.out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
        f"names:\n" + "".join(f"  {i}: {c}\n" for i, c in enumerate(CLASSES)),
        encoding="utf-8",
    )
    print(f"train={len(splits['train'])} val={len(splits['val'])} etiketsiz={bos}")
    print(f"yaml: {a.out / 'rdd2022.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
