"""YOLO26 - RDD2022 ince ayari. 4 GB VRAM (RTX 3050 Laptop) icin ayarlanmistir.

OLCUM (imgsz=640, RTX 3050 Laptop 4 GB, torch 2.6.0+cu124, 23.650 goruntu):
    tam veri + amp   -> batch 8/6, workers 8/4: 3 kosunun 3'unde NaN (bir kez
                        ardindan "CUDA illegal memory access")
    47 goruntu + amp -> ayni ayarla bir kez NaN, bir kez finite (ARALIKLI)
    tam veri - amp   -> batch 4: 45/45 iterasyon finite, 4.99 it/s (~20 dk/epoch)

Elenen hipotezler: batch esigi, imgsz, workers sayisi, dejenere etiket
(34.459 kutunun tamami pozitif alanli, min w=0.0078 h=0.0067), fp16 (fp32'de de
oldu), surucu/CUDA surumu (cu124+cuDNN9.1 ve cu128+cuDNN9.19'da ayni).

MOZAIK BULGUSU: mosaic=1.0 -> NaN 168. iterasyonda; mosaic=0.0 -> 1500 iterasyon
temiz. Mosaic 4 goruntuyu birlestirip rastgele afin uyguladigi icin kaynak
etiketler temiz olsa da donusum sonrasi kutu sifir alana kirpilabiliyor.
Bu yuzden mozaik VARSAYILAN OLARAK KAPALI. Acmak icin --mosaic.
Not: mozaik kapaliyken de seyrek "CUDA illegal memory access" gorulebiliyor -
bu yuzden asagidaki kosu, son checkpoint'ten otomatik devam eder.

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

# Muhafiz adgs/trainguard.py'ye tasindi: CarDD egitimi ayni 4 GB kartta ayni
# riski tasiyor ve ikinci bir kopya, iki farkli davranan iki muhafiz demekti.
# Adlar burada korunuyor - mevcut testler ve cagrilar degismesin.
from adgs.trainguard import ARDISIK_SICRAMA_SINIRI as _ARDISIK_SICRAMA_SINIRI  # noqa: F401
from adgs.trainguard import MAKS_KAYIP as _MAKS_KAYIP  # noqa: F401
from adgs.trainguard import nan_muhafizi as _nan_muhafizi


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
    ap.add_argument(
        "--mosaic", action="store_true",
        help="Mozaik artirimi. Bu veri kumesinde NaN uretiyor - bkz. modul basligi",
    )
    ap.add_argument(
        "--deneme", default=4, type=int,
        help="CUDA hatasi/NaN sonrasi son checkpoint'ten kac kez devam edilsin",
    )
    a = ap.parse_args()

    from ultralytics import YOLO

    cikti = Path("runs/rdd2022").resolve()
    son_ckpt = cikti / "yolo26s" / "weights" / "last.pt"

    for deneme in range(1, a.deneme + 1):
        # Cokme sonrasi son tamamlanan epoch'tan devam - seyrek CUDA hatalarinin
        # 5 saatlik kosuyu bastan aldirmasini engeller.
        devam = son_ckpt.exists()
        model = YOLO(str(son_ckpt) if devam else a.model)
        model.add_callback("on_train_batch_end", _nan_muhafizi)
        try:
            model.train(
                data=str(a.data),
                epochs=a.epochs,
                imgsz=a.imgsz,
                batch=a.batch,
                amp=a.amp,  # varsayilan KAPALI - fp16 bu donanimda NaN uretiyor
                mosaic=1.0 if a.mosaic else 0.0,
                cache=False,  # RAM'e sigmaz
                workers=4,
                # Mutlak yol sart: goreli verilirse ultralytics kendi runs_dir
                # ayarinin altina yuvalar (runs/detect/runs/rdd2022/...) ve
                # adgs run --model varsayilani tutmaz.
                project=str(cikti),
                name="yolo26s",
                exist_ok=True,
                resume=devam,
                patience=4,
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
