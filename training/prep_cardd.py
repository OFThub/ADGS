"""CarDD (FiftyOne export) -> YOLO detection formati.

Kaynak: huggingface.co/datasets/harpreetsahota/CarDD

IKI ONEMLI SAPMA - ciktiyi yorumlarken bilinmeli:

1. Bu ayna 2816 goruntu icerir; CarDD makalesi 4000 der. Yani elimizdeki veri
   kumenin TAMAMI DEGILDIR ve olculen mAP makaleyle dogrudan karsilastirilamaz.
2. Aynada resmi train/val/test bolunmesi YOKTUR (tags alani bos). Bu betik dosya
   adinin hash'inden deterministik bir bolunme uretir. Deterministik olmasi
   sart: her calistirmada farkli bolunme, "iyilesme" gibi gorunen gurultu uretir.

Segmentasyon maskeleri kaynakta VARDIR (samples[].segmentations) ama burada
kullanilmaz: 4 GB VRAM'de seg egitimi pratik degil ve M8'in siddet skoru zaten
bbox alan oranindan hesaplaniyor (M5 ile ayni yaklasim).

LISANS: CarDD ticari olmayan arastirma/egitim kullanimiyla sinirlidir ve
Flickr/Shutterstock sartlarina tabidir. Belediyeye operasyonel teslimatta
KULLANILAMAZ - bkz. README "Lisans notu".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

from adgs.vehicledamage import CARDD_CLASSES

# Sinif sirasi M8'in model dogrulamasiyla ayni olmali; degisirse egitilen
# modelin names alani M8'in bekledigi kumeyle uyusmaz.
_SINIF_INDEKS = {ad: i for i, ad in enumerate(CARDD_CLASSES)}


def _bolum(dosya_adi: str, val_orani: float) -> str:
    """Dosya adindan deterministik train/val bolunmesi."""
    h = hashlib.sha256(dosya_adi.encode("utf-8")).hexdigest()
    return "val" if (int(h[:8], 16) % 1000) < val_orani * 1000 else "train"


def _yolo_satiri(det: dict) -> str | None:
    """FiftyOne [x,y,w,h] (sol-ust, normalize) -> YOLO [cx,cy,w,h]."""
    idx = _SINIF_INDEKS.get(det.get("label"))
    if idx is None:
        return None  # beklenmeyen sinif - sessizce 0'a atamak yanlis etiketlerdi
    x, y, w, h = det["bounding_box"]
    cx, cy = x + w / 2.0, y + h / 2.0
    # Kaynakta kutu kenari 1.0'i birkac ondalik asabiliyor; kirp.
    cx, cy = min(max(cx, 0.0), 1.0), min(max(cy, 0.0), 1.0)
    w, h = min(max(w, 0.0), 1.0), min(max(h, 0.0), 1.0)
    if w <= 0 or h <= 0:
        return None  # dejenere kutu - egitimde NaN kaynagi
    return f"{idx} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=Path("data/raw/CarDD"), type=Path)
    ap.add_argument("--out", default=Path("data/cardd_yolo"), type=Path)
    ap.add_argument("--val-orani", default=0.2, type=float)
    a = ap.parse_args()

    samples_yolu = a.src / "samples.json"
    if not samples_yolu.exists():
        print(f"HATA: {samples_yolu} yok. Once indirin (huggingface_hub snapshot_download).")
        return 1

    kayitlar = json.loads(samples_yolu.read_text(encoding="utf-8"))["samples"]
    for bolum in ("train", "val"):
        (a.out / "images" / bolum).mkdir(parents=True, exist_ok=True)
        (a.out / "labels" / bolum).mkdir(parents=True, exist_ok=True)

    sayim = {"train": 0, "val": 0}
    etiketsiz = atlanan_kutu = eksik_goruntu = 0
    sinif_sayim: dict[str, int] = {}

    for kayit in kayitlar:
        rel = kayit["filepath"]
        kaynak = a.src / rel
        if not kaynak.exists():
            eksik_goruntu += 1
            continue
        ad = Path(rel).name
        bolum = _bolum(ad, a.val_orani)

        satirlar = []
        for det in (kayit.get("detections") or {}).get("detections") or []:
            s = _yolo_satiri(det)
            if s is None:
                atlanan_kutu += 1
                continue
            satirlar.append(s)
            sinif_sayim[det["label"]] = sinif_sayim.get(det["label"], 0) + 1

        shutil.copy2(kaynak, a.out / "images" / bolum / ad)
        (a.out / "labels" / bolum / f"{Path(ad).stem}.txt").write_text(
            "\n".join(satirlar), encoding="utf-8"
        )
        sayim[bolum] += 1
        if not satirlar:
            etiketsiz += 1

    yaml_yolu = a.out / "cardd.yaml"
    adlar = "\n".join(f"  {i}: {ad}" for i, ad in enumerate(CARDD_CLASSES))
    yaml_yolu.write_text(
        "# CarDD -> YOLO. LISANS: ticari olmayan arastirma/egitim (bkz. README).\n"
        f"path: {a.out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"names:\n{adlar}\n",
        encoding="utf-8",
    )

    print(f"train={sayim['train']} val={sayim['val']} etiketsiz={etiketsiz}")
    if eksik_goruntu:
        print(f"UYARI: {eksik_goruntu} goruntu dosyasi bulunamadi (indirme eksik?)")
    if atlanan_kutu:
        print(f"UYARI: {atlanan_kutu} kutu atlandi (bilinmeyen sinif veya sifir alan)")
    print("sinif dagilimi:", dict(sorted(sinif_sayim.items())))
    print("Veri tanimi:", yaml_yolu)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
