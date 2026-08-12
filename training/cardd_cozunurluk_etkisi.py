"""FAZ 6 BULGUSU: CarDD modelinin cozunurluge gore performans dususu.

Bu bir basarisizlik olcumu degil, bir SINIR olcumudur ve M8'in kod icindeki
cozunurluk kapisinin (vehicledamage.MIN_KENAR_PIKSEL) dayanagidir.

Neden gerekli: CarDD yakin cekim fotograflardan olusur (tipik 1000x750, arac
kareyi doldurur). Sabit CCTV ise araci 60x40 piksellik bir kutu icinde gorur.
Model bu kutuda da bir sinif uretir - ama uretebilmesi, dogru uretmesi anlamina
gelmez. Bu betik "hangi boyutta anlamli olmaktan cikiyor" sorusunu sayiyla
cevaplar.

Yontem: val kumesinin goruntuleri N piksel uzun kenara KUCULTULUR (bilgi kaybi
gercek), sonra model normal imgsz ile degerlendirilir. Etiketler normalize
oldugu icin olcekle degismez, aynen kopyalanir.

Beklenti: ince dokulu siniflar (scratch = cizik) once cokecek, buyuk ve yapisal
olanlar (glass shatter, tire flat) daha uzun dayanacaktir. Cikan tablo bu
beklentiyi dogrular ya da curutur - ikisi de bulgudur.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# CCTV'de tipik arac kutusu boyutlarini kapsayan olcekler.
VARSAYILAN_OLCEKLER = [640, 384, 256, 192, 128, 96, 64]


def _kucult(kaynak_dizin: Path, hedef_dizin: Path, uzun_kenar: int) -> int:
    """Goruntuleri uzun kenari `uzun_kenar` olacak sekilde kucultup yazar."""
    import cv2

    hedef_dizin.mkdir(parents=True, exist_ok=True)
    n = 0
    for p in sorted(kaynak_dizin.iterdir()):
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        olcek = uzun_kenar / float(max(h, w))
        if olcek < 1.0:
            yeni = (max(1, int(round(w * olcek))), max(1, int(round(h * olcek))))
            # INTER_AREA kucultmede dogru filtredir; bilgi kaybi gercekci olur.
            img = cv2.resize(img, yeni, interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(hedef_dizin / p.name), img)
        n += 1
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="runs/cardd/yolo26s/weights/best.pt")
    ap.add_argument("--data", default=Path("data/cardd_yolo"), type=Path)
    ap.add_argument("--out", default=Path("runs/cardd_olcek"), type=Path)
    ap.add_argument("--imgsz", default=640, type=int)
    ap.add_argument("--batch", default=2, type=int)
    ap.add_argument("--olcekler", default=",".join(str(o) for o in VARSAYILAN_OLCEKLER),
                    help="Virgulle ayrilmis uzun kenar piksel degerleri")
    a = ap.parse_args()

    if not Path(a.model).exists():
        print(f"HATA: model yok: {a.model}. Once: python training/train_cardd.py")
        return 1
    val_img = a.data / "images" / "val"
    val_lbl = a.data / "labels" / "val"
    if not val_img.exists() or not val_lbl.exists():
        print(f"HATA: val kumesi yok ({val_img}). Once: python training/prep_cardd.py")
        return 1

    from ultralytics import YOLO

    model = YOLO(a.model)
    adlar: dict[int, str] = model.names
    a.out.mkdir(parents=True, exist_ok=True)
    sonuclar = []

    for px in [int(s) for s in a.olcekler.split(",") if s.strip()]:
        kok = a.out / f"{px}px"
        img_dir = kok / "images" / "val"
        lbl_dir = kok / "labels" / "val"
        n = _kucult(val_img, img_dir, px)
        # Etiketler normalize - olcekle degismez, aynen kopyalanir.
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for p in val_lbl.glob("*.txt"):
            shutil.copy2(p, lbl_dir / p.name)

        yaml_yolu = kok / "cardd.yaml"
        yaml_yolu.write_text(
            f"path: {kok.resolve().as_posix()}\n"
            "train: images/val\n"   # egitim yapilmiyor; ultralytics alani bekliyor
            "val: images/val\n"
            "names:\n"
            + "\n".join(f"  {i}: {ad}" for i, ad in sorted(adlar.items())) + "\n",
            encoding="utf-8",
        )

        r = model.val(data=str(yaml_yolu), imgsz=a.imgsz, batch=a.batch, verbose=False)
        kayit = {
            "px": px,
            "goruntu": n,
            "map50": round(float(r.box.map50), 4),
            "map50_95": round(float(r.box.map), 4),
            "sinif_map50": {adlar.get(i, str(i)): round(float(ap), 4)
                            for i, ap in enumerate(list(r.box.maps))},
        }
        sonuclar.append(kayit)
        print(f"{px:>5} px  mAP@0.5={kayit['map50']:.4f}  "
              f"mAP@0.5:0.95={kayit['map50_95']:.4f}")

    (a.out / "sonuc.json").write_text(
        json.dumps({"model": a.model, "olcekler": sonuclar}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Ozet tablo - README'ye ve rapora dogrudan yapistirilabilir.
    sirali_ad = [adlar[i] for i in sorted(adlar)]
    print("\n| uzun kenar (px) | mAP@0.5 | " + " | ".join(sirali_ad) + " |")
    print("|---|---|" + "---|" * len(sirali_ad))
    for k in sonuclar:
        hucreler = " | ".join(f"{k['sinif_map50'].get(ad, 0):.3f}" for ad in sirali_ad)
        print(f"| {k['px']} | {k['map50']:.3f} | {hucreler} |")

    if sonuclar:
        taban = sonuclar[0]["map50"]
        print(f"\nReferans (en yuksek cozunurluk) mAP@0.5 = {taban:.3f}")
        for k in sonuclar[1:]:
            kayip = (1 - k["map50"] / taban) * 100 if taban > 0 else 0.0
            print(f"  {k['px']:>4} px -> %{kayip:.1f} dusus")
        print("\nM8'in cozunurluk kapisi (vehicledamage.MIN_KENAR_PIKSEL) bu "
              "tablodan secilmelidir: dususun kabul edilemez hale geldigi olcek.")
    print(f"\nSonuc: {a.out / 'sonuc.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
