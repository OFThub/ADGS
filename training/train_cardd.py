"""YOLO26 - CarDD ince ayari (M8 arac hasar modeli).

Ayarlar RDD2022 kosusunun olcumlerinden gelir; ayni 4 GB kart, ayni tuzaklar:
    --amp KAPALI        fp16 bu donanimda cls_loss tasmasiyla NaN uretiyor
    --mosaic KAPALI     mozaik donusumu kutulari sifir alana kirpip NaN uretiyor
    batch 2, imgsz 640  ~1.4 GB - ekran + IDE ile ayni kartta guvenli bant
                        (batch 8 / 3.2 GB denemeleri "CUDA illegal memory
                        access" ile coktu)

CarDD 2816 goruntudur (RDD'nin ~1/8'i), bu yuzden epoch sayisi daha yuksek.

Kosu, seyrek CUDA hatalarindan sonra son checkpoint'ten otomatik devam eder.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from adgs.trainguard import nan_muhafizi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Path("data/cardd_yolo/cardd.yaml"), type=Path)
    ap.add_argument("--model", default="yolo26s.pt", help="4 GB VRAM: n veya s")
    ap.add_argument("--epochs", default=40, type=int)
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--batch", default=2, type=int,
                    help="4 GB'a sigan olculmus deger - bkz. modul basligi")
    ap.add_argument("--amp", action="store_true",
                    help="fp16'yi ac. Bu donanimda NaN uretiyor, ACMA")
    ap.add_argument("--mosaic", action="store_true",
                    help="Mozaik artirimi. NaN uretiyor - bkz. modul basligi")
    ap.add_argument("--deneme", default=4, type=int,
                    help="CUDA hatasi sonrasi son checkpoint'ten kac kez devam edilsin")
    a = ap.parse_args()

    if not a.data.exists():
        print(f"HATA: {a.data} yok. Once: python training/prep_cardd.py")
        return 1

    from ultralytics import YOLO

    cikti = Path("runs/cardd").resolve()
    son_ckpt = cikti / "yolo26s" / "weights" / "last.pt"

    for deneme in range(1, a.deneme + 1):
        devam = son_ckpt.exists()
        model = YOLO(str(son_ckpt) if devam else a.model)
        model.add_callback("on_train_batch_end", nan_muhafizi)
        try:
            model.train(
                data=str(a.data),
                epochs=a.epochs,
                imgsz=a.imgsz,
                batch=a.batch,
                amp=a.amp,
                mosaic=1.0 if a.mosaic else 0.0,
                cache=False,
                workers=4,
                # Mutlak yol sart: goreli verilirse ultralytics kendi runs_dir
                # ayarinin altina yuvalar (runs/detect/runs/cardd/...).
                project=str(cikti),
                name="yolo26s",
                exist_ok=True,
                resume=devam,
                patience=8,
            )
        except Exception as e:  # noqa: BLE001 - hangi CUDA hatasi geldigi onemsiz
            if deneme >= a.deneme:
                print(f"HATA: {deneme} denemede tamamlanamadi -> {e}")
                raise
            print(f"UYARI: kosu koptu ({type(e).__name__}), "
                  f"son checkpoint'ten devam ediliyor (deneme {deneme + 1}/{a.deneme})")
            continue
        print("Agirlik:", son_ckpt.parent / "best.pt")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
