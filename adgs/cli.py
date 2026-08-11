"""ADGS komut satiri arayuzu."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from adgs import __version__

# Windows konsolu cp1254 olabilir; CLI ciktisi ASCII tutulur ki UnicodeEncodeError
# ile cokmesin. Turkce metinler rapor/JSON tarafinda (UTF-8) yer alir.

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"


def _check(label: str, fn) -> bool:
    try:
        detail = fn()
    except Exception as exc:  # noqa: BLE001 - doctor her hatayi raporlamali
        print(f"  [FAIL] {label}: {type(exc).__name__}: {exc}")
        return False
    print(f"  [ OK ] {label}: {detail}")
    return True


def _cuda() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA gorunmuyor - GPU surucusu/torch kurulumu kontrol edilmeli")
    return f"{torch.cuda.get_device_name(0)} (torch {torch.__version__})"


def _ultralytics() -> str:
    import ultralytics

    return f"ultralytics {ultralytics.__version__} (AGPL-3.0 - bkz. README)"


def _config() -> str:
    import yaml

    cfg = yaml.safe_load((CONFIG_DIR / "pipeline.yaml").read_text(encoding="utf-8"))
    profile = cfg["source_profile"]
    if profile not in ("cctv_fixed", "vehicle_mounted"):
        raise ValueError(f"gecersiz source_profile: {profile}")
    return f"source_profile={profile}, sample_fps={cfg['sample_fps']}"


def _schema() -> str:
    from adgs.schema import ON_DEGERLENDIRME_NOTU, Event

    evt = Event(
        event_id="doctor",
        tip="KAZA",
        alt_tip="SELF_TEST",
        t_start=0.0,
        t_end=1.0,
        frame_start=0,
        frame_end=1,
        source_video="-",
        source_profile="cctv_fixed",
        conf=1.0,
    )
    assert ON_DEGERLENDIRME_NOTU in evt.notes, "on-degerlendirme notu eklenmedi"
    return "Event sozlesmesi ve zorunlu not calisiyor"


def doctor() -> int:
    print("ADGS ortam kontrolu")
    results = [
        _check("GPU / CUDA", _cuda),
        _check("Detektor", _ultralytics),
        _check("Config", _config),
        _check("Sema", _schema),
    ]
    ok = all(results)
    print("\nSonuc:", "hazir" if ok else "eksik var - yukaridaki FAIL satirlarina bak")
    return 0 if ok else 1


def run(video: str, model: str, out: str, profile: str, conf: float) -> int:
    """Faz 1 boru hatti: video -> yol hasari -> isaretli MP4 + JSON rapor."""
    from adgs import render, roaddamage

    src = Path(video)
    if not src.exists():
        print(f"[FAIL] video bulunamadi: {src}")
        return 1
    if not Path(model).exists():
        print(f"[FAIL] model bulunamadi: {model}")
        print("       Once egitim: python training/train_rdd2022.py")
        return 1

    outdir = Path(out)
    print(f"Tespit calisiyor: {src.name}")
    events = roaddamage.detect(src, model, conf=conf, source_profile=profile)

    mp4 = render.annotate_video(src, events, outdir / f"{src.stem}_annotated.mp4")
    js = render.write_report(events, outdir / "rapor.json", source=str(src))

    print()
    print(f"{len(events)} olay (tekillestirilmis, track_id basina bir kayit)")
    for s in ("YUKSEK", "ORTA", "DUSUK"):
        n = sum(1 for e in events if f"siddet {s}" in " ".join(e.notes))
        if n:
            print(f"  {s}: {n}")
    print()
    print(f"Video : {mp4}")
    print(f"Rapor : {js}")
    return 0


def _kamera_yolu(camera: str) -> Path:
    return CONFIG_DIR / "cameras" / f"{camera}.yaml"


def calibrate(camera: str, verify: bool) -> int:
    """Kamera kalibrasyonunu yukler ve dogrular.

    --verify ile dogrulama gecmezse NON-ZERO doner: hiza bagli moduller bu
    cikis koduna bakarak devre disi kalir.
    """
    from adgs import calib

    k = calib.yukle(_kamera_yolu(camera))
    print(f"Kamera: {k.camera_id}")
    for n in k.notlar:
        print(f"  - {n}")
    if k.hata_yuzde is not None:
        print(f"  Geri donusum hatasi: %{k.hata_yuzde:.2f} (esik %{k.maks_hata_yuzde:.1f})")
    print("Sonuc:", "GECERLI" if k.gecerli else "REDDEDILDI (hiz/mesafe modulleri kapali)")
    return 0 if (k.gecerli or not verify) else 1


def track(video: str, model: str, out: str, camera: str | None, conf: float,
          tracker: str, min_kare: int) -> int:
    """Faz 2 boru hatti: video -> tespit+takip -> ID'li MP4 + izler.json."""
    from adgs import calib, detect, render

    src = Path(video)
    if not src.exists():
        print(f"[FAIL] video bulunamadi: {src}")
        return 1

    k = None
    if camera:
        k = calib.yukle(_kamera_yolu(camera))
        for n in k.notlar:
            print(f"  kalibrasyon: {n}")
        if not k.gecerli:
            print("  UYARI: kalibrasyon gecersiz - dunya koordinati ve hiz URETILMEYECEK")

    print(f"Takip calisiyor: {src.name} (tracker={tracker})")
    izler = detect.takip_et(src, model_path=model, kalib=k, conf=conf, tracker=tracker)
    toplam = len(izler)
    izler = detect.kisa_izleri_ele(izler, min_kare=min_kare)

    outdir = Path(out)
    mp4 = render.annotate_tracks(src, izler, outdir / f"{src.stem}_tracked.mp4", kalib=k)
    js = render.write_tracks(izler, outdir / "izler.json", source=str(src), kalib=k)

    print()
    print(f"{len(izler)} iz ({toplam - len(izler)} kisa iz elendi, <{min_kare} kare)")
    for cls, n in detect.ozet(izler).items():
        print(f"  {cls}: {n}")
    print()
    print(f"Video : {mp4}")
    print(f"Izler : {js}")
    return 0


def evaluate(model: str, data: str, imgsz: int, batch: int, hedef: float) -> int:
    """Kabul kriteri olcumu: mAP@0.5.

    Hedefin altinda kalirsa NON-ZERO doner - "egitim bitti" ile "kriter saglandi"
    ayni sey degildir, ikincisi olculur.
    """
    from ultralytics import YOLO

    if not Path(model).exists():
        print(f"[FAIL] model bulunamadi: {model}")
        return 1
    if not Path(data).exists():
        print(f"[FAIL] veri tanimi bulunamadi: {data}")
        return 1

    m = YOLO(model)
    r = m.val(data=data, imgsz=imgsz, batch=batch, verbose=False)
    map50 = float(r.box.map50)
    print(f"\nModel : {model}")
    print(f"Veri  : {data}")
    print(f"mAP@0.5      : {map50:.4f}   (hedef >= {hedef})")
    print(f"mAP@0.5:0.95 : {float(r.box.map):.4f}")
    adlar = getattr(m, "names", {}) or {}
    for i, ap in enumerate(list(r.box.maps)):
        print(f"  {adlar.get(i, i)}: {float(ap):.4f}")
    ok = map50 >= hedef
    print("\nSonuc:", "KRITER SAGLANDI" if ok else "KRITER SAGLANMADI")
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="adgs",
        description="Trafik video analiz sistemi - karar destek araci (baglayici degildir)",
    )
    parser.add_argument("--version", action="version", version=f"adgs {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("doctor", help="Ortam ve yapilandirma kontrolu")

    r = sub.add_parser("run", help="Videoyu isle (Faz 1: yol hasari)")
    r.add_argument("video")
    r.add_argument("--model", default="runs/rdd2022/yolo26s/weights/best.pt")
    r.add_argument("--out", default="runs/f1")
    r.add_argument("--profile", default="vehicle_mounted",
                   choices=["vehicle_mounted", "cctv_fixed"])
    r.add_argument("--conf", default=0.35, type=float)

    c = sub.add_parser("calibrate", help="Kamera kalibrasyonunu dogrula (Faz 2)")
    c.add_argument("--camera", required=True)
    c.add_argument("--verify", action="store_true",
                   help="Dogrulama gecmezse non-zero cikis kodu dondur")

    t = sub.add_parser("track", help="Tespit + takip (Faz 2: arac/yaya ID'leri)")
    t.add_argument("video")
    t.add_argument("--model", default="yolo26s.pt")
    t.add_argument("--out", default="runs/f2")
    t.add_argument("--camera", default=None,
                   help="config/cameras/<ad>.yaml - kalibrasyon icin")
    t.add_argument("--conf", default=0.35, type=float)
    t.add_argument("--tracker", default="botsort.yaml",
                   choices=["botsort.yaml", "bytetrack.yaml"])
    t.add_argument("--min-kare", default=3, type=int,
                   help="Bu kareden kisa izler elenir (yanlis pozitif filtresi)")

    e = sub.add_parser("eval", help="Kabul kriteri olcumu (mAP@0.5)")
    e.add_argument("--model", default="runs/rdd2022/yolo26s/weights/best.pt")
    e.add_argument("--data", default="data/rdd_yolo/rdd2022.yaml")
    e.add_argument("--imgsz", default=640, type=int)
    e.add_argument("--batch", default=4, type=int)
    e.add_argument("--hedef", default=0.45, type=float,
                   help="Faz 1 yol hasari kabul esigi")

    args = parser.parse_args(argv)
    if args.cmd == "eval":
        return evaluate(args.model, args.data, args.imgsz, args.batch, args.hedef)
    if args.cmd == "doctor":
        return doctor()
    if args.cmd == "run":
        return run(args.video, args.model, args.out, args.profile, args.conf)
    if args.cmd == "calibrate":
        return calibrate(args.camera, args.verify)
    if args.cmd == "track":
        return track(args.video, args.model, args.out, args.camera, args.conf,
                     args.tracker, args.min_kare)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
