"""YOLO26 - RDD2022 ince ayari. 4 GB VRAM (RTX 3050 Laptop) icin ayarlanmistir.

OLCUM (imgsz=640, RTX 3050 Laptop 4 GB, torch 2.6.0+cu124, 23.650 goruntu):
    tam veri + amp   -> batch 8/6, workers 8/4: 3 kosunun 3'unde NaN (bir kez
                        ardindan "CUDA illegal memory access")
    47 goruntu + amp -> ayni ayarla bir kez NaN, bir kez finite (ARALIKLI)
    tam veri - amp   -> batch 4: 45/45 iterasyon finite, 4.99 it/s (~20 dk/epoch)

Elenen hipotezler: batch esigi, imgsz, workers sayisi, dejenere etiket
(34.459 kutunun tamami pozitif alanli, min w=0.0078 h=0.0067).

Kalan aciklama fp16'nin dar us araligi: cls_loss 20-30 bandinda ve ara toplamlar
ornek cesitliligiyle buyuyor, bu yuzden buyuk kumede tasma olasiligi artiyor.
Bu is yukunde AMP KAPALI kullanilir - %40 hiz, NaN riskine degmez.

Ultralytics'in kendi AMP kontrolu yalnizca cikarimi test ettigi icin bunu
yakalayamaz; asagidaki NaN muhafizi onun yerine gecer ve 5+ saatlik bir kosunun
sessizce cop uretmesini engeller.
"""

from __future__ import annotations

import argparse
from pathlib import Path

def _nan_muhafizi(trainer) -> None:
    """NaN/Inf kayip goruldugunde egitimi ANINDA durdurur.

    Iki yeri birden kontrol eder: trainer.loss toplam skalerdir, ekranda gorunen
    box/cls/l1 degerleri ise loss_items sozlugunden gelir - biri NaN iken digeri
    finite olabiliyor, tek birine bakmak muhafizi kor birakir.
    """
    import torch

    bozuk: list[str] = []
    kayip = getattr(trainer, "loss", None)
    if isinstance(kayip, torch.Tensor) and not torch.isfinite(kayip).all():
        bozuk.append("loss")
    items = getattr(trainer, "loss_items", None)
    if isinstance(items, dict):
        bozuk += [k for k, v in items.items() if not torch.isfinite(torch.as_tensor(v)).all()]
    elif isinstance(items, torch.Tensor) and not torch.isfinite(items).all():
        bozuk.append("loss_items")

    if bozuk:
        raise RuntimeError(
            f"Kayip NaN/Inf ({', '.join(bozuk)}) - egitim durduruldu "
            f"(batch={trainer.args.batch}, imgsz={trainer.args.imgsz}, "
            f"amp={trainer.args.amp}). Bu donanimda batch<=6 kullanin; "
            f"tekrarlarsa --batch 4."
        )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=Path("data/rdd_yolo/rdd2022.yaml"), type=Path)
    ap.add_argument("--model", default="yolo26s.pt", help="4 GB VRAM: n veya s. m/l/x egitilemez.")
    ap.add_argument("--epochs", default=15, type=int)
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument(
        "--batch", default=4, type=int,
        help="amp kapaliyken 4 GB'a sigan deger - bkz. modul basligi",
    )
    ap.add_argument(
        "--amp", action="store_true",
        help="fp16'yi ac. Bu donanimda NaN uretiyor, ACMA - bkz. modul basligi",
    )
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(a.model)
    model.add_callback("on_train_batch_end", _nan_muhafizi)
    model.train(
        data=str(a.data),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        amp=a.amp,  # varsayilan KAPALI - fp16 bu donanimda NaN uretiyor
        cache=False,  # RAM'e sigmaz
        workers=4,
        # Mutlak yol sart: goreli verilirse ultralytics kendi runs_dir ayarinin
        # altina yuvalar (runs/detect/runs/rdd2022/...) ve adgs run --model
        # varsayilani tutmaz. exist_ok ile isim -2/-3 diye artmaz.
        project=str(Path("runs/rdd2022").resolve()),
        name="yolo26s",
        exist_ok=True,
        patience=4,
    )
    print("Agirlik: runs/rdd2022/yolo26s/weights/best.pt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
