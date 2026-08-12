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


def _video_fps(video: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        return cap.get(cv2.CAP_PROP_FPS) or 25.0
    finally:
        cap.release()


def _klip_ekle(events: list, src: Path, outdir: Path) -> None:
    """Her olaya klip kaniti ekler. Kanitsiz olay rapora zaten yazilmaz."""
    from adgs import render

    for e in events:
        klip = render.save_clip(src, e.frame_start, e.frame_end,
                                outdir / "klipler" / f"{e.event_id}.mp4")
        if klip:
            e.evidence["clip"] = str(klip)


def _tarih_coz(metin: str | None):
    """--tarih degerini date'e cevirir. Verilmezse None (ceza hesaplanmaz).

    Bilerek bugune DUSULMEZ: arsiv videosu bugunun ceza tablosuyla hesaplanirsa
    sessizce yanlis tutar uretilir.
    """
    from datetime import date

    if not metin:
        return None
    try:
        return date.fromisoformat(metin)
    except ValueError as e:
        raise SystemExit(f"[FAIL] --tarih ISO bicimde olmali (YYYY-AA-GG): {e}") from e


def run(video: str, model: str, out: str, profile: str, conf: float,
        detect_tipleri: str = "roaddamage", track_model: str = "yolo26s.pt",
        camera: str | None = None, rules: str | None = None,
        tarih_metni: str | None = None,
        damage_model: str = "runs/cardd/yolo26s/weights/best.pt") -> int:
    """Video isleme boru hatti: yol hasari (F1), kaza (F3), ihlal (F4),
    kusur + ceza (F5), arac hasari (F6)."""
    from adgs import render
    from adgs import violations as m4

    src = Path(video)
    if not src.exists():
        print(f"[FAIL] video bulunamadi: {src}")
        return 1
    tarih = _tarih_coz(tarih_metni)

    tipler = {t.strip() for t in detect_tipleri.split(",") if t.strip()}
    ihlal_adlari = set(m4.KAYIT)
    gecersiz = tipler - ({"roaddamage", "accident", "ihlal"} | ihlal_adlari)
    if gecersiz:
        print(f"[FAIL] bilinmeyen tespit tipi: {sorted(gecersiz)}")
        print(f"       gecerli: roaddamage, accident, ihlal, {', '.join(sorted(ihlal_adlari))}")
        return 1

    outdir = Path(out)
    events: list = []
    uyarilar: list[str] = []

    if "roaddamage" in tipler:
        from adgs import roaddamage

        if not Path(model).exists():
            print(f"[FAIL] yol hasari modeli bulunamadi: {model}")
            print("       Once egitim: python training/train_rdd2022.py")
            return 1
        print(f"Yol hasari taraniyor: {src.name}")
        events += roaddamage.detect(src, model, conf=conf, source_profile=profile)

    ihlal_istendi = "ihlal" in tipler or bool(tipler & ihlal_adlari)
    if "accident" in tipler or ihlal_istendi:
        from adgs import accident, calib
        from adgs import detect as m2

        k = calib.yukle(_kamera_yolu(camera)) if camera else None
        if k is not None and not k.gecerli:
            uyarilar.append(
                "kalibrasyon gecersiz - hiz/mesafe modulleri kapali, "
                "kazada okluzyon filtresi uygulanmiyor"
            )
        # M2 TEK KEZ calisir. Kaza ve ihlal ayni iz listesini okur: ikinci bir
        # takip kosusu farkli track_id uretebilir ve ayni arac iki raporda iki
        # farkli numarayla gorunurdu (plan, veri akisi kurali 1).
        print(f"Tespit + takip: {src.name}")
        izler = m2.kisa_izleri_ele(
            m2.takip_et(src, model_path=track_model, kalib=k, conf=conf)
        )
        fps = _video_fps(src)

        if "accident" in tipler:
            kazalar = accident.tespit_et(izler, fps, str(src),
                                         source_profile=profile, kalib=k)
            _klip_ekle(kazalar, src, outdir)
            # M8 - arac hasari (Faz 6). Arac kirpmasi cok kucukse sinif bazli
            # cikti URETILMEZ; sebep uyarilara yazilir.
            from adgs import vehicledamage as m8

            uyarilar += m8.degerlendir(src, kazalar, izler, model_path=damage_model,
                                       conf=conf)
            events += kazalar

        if ihlal_istendi:
            ctx = m4.Baglam(
                video=src, fps=fps, source_profile=profile, kalib=k,
                kamera=m4.kamera_yukle(_kamera_yolu(camera)) if camera else {},
                kural=m4.kural_yukle(rules),
            )
            secili = None if "ihlal" in tipler else sorted(tipler & ihlal_adlari)
            ihlaller = m4.tespit_et(izler, ctx, secili)
            _klip_ekle(ihlaller, src, outdir)
            events += ihlaller
            uyarilar += ctx.uyarilar

    # M6 + M7 - GPU'suz, saf. ALTYAPI olaylari dokunulmadan gecer (belediyenin
    # kendi gorev alani; kusur/ceza dogurmaz).
    from adgs import fault, penalty

    try:
        fault.uygula(events)
    except FileNotFoundError as e:
        # Kusur tablosu olmadan siniflandirma her ihlali TALI yapardi. Yol
        # hasari sonuclari yine de yazilir; eksik oldugu acikca soylenir.
        uyarilar.append(f"kusur motoru calismadi: {e}")
    penalty.uygula(events, olay_tarihi=tarih)

    mp4 = render.annotate_video(src, events, outdir / f"{src.stem}_annotated.mp4")
    js = render.write_report(events, outdir / "rapor.json", source=str(src),
                             uyarilar=uyarilar)

    print()
    print(f"{len(events)} olay")
    for tip in ("ALTYAPI", "KAZA", "IHLAL"):
        n = sum(1 for e in events if e.tip == tip)
        if n:
            print(f"  {tip}: {n}")
    for kod in sorted({e.alt_tip for e in events if e.tip == "IHLAL"}):
        print(f"    {kod}: {sum(1 for e in events if e.alt_tip == kod)}")
    for s in ("YUKSEK", "ORTA", "DUSUK"):
        n = sum(1 for e in events if f"siddet {s}" in " ".join(e.notes))
        if n:
            print(f"  siddet {s}: {n}")

    # "0 ihlal" ile "3 dedektor bakamadi, 0 ihlal" ayni sey degil - susmak yasak.
    if uyarilar:
        print("\nCALISAMAYAN / KISITLI MODULLER:")
        for u in uyarilar:
            print(f"  ! {u}")

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


def explain(rapor: str, event_id: str | None) -> int:
    """Bir olayin kusur/ceza gerekcesini insan okunur bicimde yazar (Faz 5).

    Amaci denetlenebilirlik: hangi maddeye neden baglandigi, hangi tablodan
    hesaplandigi ve hangi sinirlarin gecerli oldugu tek ekranda gorulmeli.
    """
    import json

    p = Path(rapor)
    if not p.exists():
        print(f"[FAIL] rapor bulunamadi: {p}")
        return 1
    veri = json.loads(p.read_text(encoding="utf-8"))
    olaylar = veri.get("events") or []
    if event_id:
        olaylar = [e for e in olaylar if e.get("event_id") == event_id]
        if not olaylar:
            print(f"[FAIL] olay bulunamadi: {event_id}")
            return 1

    for evt in olaylar:
        print("=" * 72)
        print(f"{evt['event_id']}  {evt['tip']} / {evt['alt_tip']}  "
              f"conf={evt['conf']}  kare {evt['frame_start']}-{evt['frame_end']}")
        for taraf in evt.get("parties") or []:
            plaka = taraf.get("plate") or "-"
            print(f"  iz #{taraf['track_id']} (plaka {plaka})")
            if not taraf.get("violations"):
                # Bos birakmak "kusursuz" gibi okunur; acikca yazilir.
                print("    TESPIT_EDILEMEDI - ihlal cikarilamadi (KUSURSUZ DEGIL)")
                continue
            for v in taraf["violations"]:
                madde = v.get("ktk_madde") or "m.84 disinda"
                print(f"    {v['ihlal_kodu']}  ->  KTK {madde}  "
                      f"[{v.get('kusur_sinifi') or '?'}]")
                tutar = v.get("ceza_tutari_try")
                if tutar is None:
                    print("      ceza: HESAPLANMADI (bkz. notlar)")
                else:
                    puan = v.get("ceza_puani")
                    print(f"      ceza: {tutar} TL"
                          + (f", {puan} ceza puani" if puan is not None else "")
                          + f"  (tablo {v.get('ceza_tablosu_tarihi')})")
        print("  --- notlar ---")
        for n in evt.get("notes") or []:
            print(f"  * {n}")
    if not olaylar:
        print("Raporda olay yok.")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Windows konsolu cp1254'tur; rapordaki Turkce notlar (kusur gerekcesi,
    # on-degerlendirme uyarisi) oraya basilinca bozulur. Hukuki gerekcenin
    # okunamaz cikmasi kabul edilemez - stdout UTF-8'e alinir.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass  # yonlendirilmis/eski akis - ASCII ciktilar yine calisir

    parser = argparse.ArgumentParser(
        prog="adgs",
        description="Trafik video analiz sistemi - karar destek araci (baglayici degildir)",
    )
    parser.add_argument("--version", action="version", version=f"adgs {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("doctor", help="Ortam ve yapilandirma kontrolu")

    r = sub.add_parser("run", help="Videoyu isle (yol hasari ve/veya kaza)")
    r.add_argument("video")
    r.add_argument("--model", default="runs/rdd2022/yolo26s/weights/best.pt",
                   help="Yol hasari modeli")
    r.add_argument("--out", default="runs/f1")
    r.add_argument("--profile", default="vehicle_mounted",
                   choices=["vehicle_mounted", "cctv_fixed"])
    r.add_argument("--conf", default=0.35, type=float)
    r.add_argument("--detect", default="roaddamage",
                   help="Virgulle ayrilmis: roaddamage, accident, ihlal (config'te "
                        "aktif tum ihlaller) veya tek tek redlight, wrongway, "
                        "parking, lane, tailgating, speed")
    r.add_argument("--track-model", default="yolo26s.pt",
                   help="Kaza/ihlal tespiti icin arac/yaya modeli")
    r.add_argument("--camera", default=None,
                   help="config/cameras/<ad>.yaml - kalibrasyon ve ihlal geometrisi")
    r.add_argument("--rules", default=None,
                   help="Ihlal esikleri (varsayilan config/rules/violations.yaml)")
    r.add_argument("--damage-model", default="runs/cardd/yolo26s/weights/best.pt",
                   help="CarDD arac hasari modeli (Faz 6). Yoksa hasar "
                        "degerlendirmesi yapilmaz, sebep raporlanir")
    r.add_argument("--tarih", default=None, metavar="YYYY-AA-GG",
                   help="Videonun cekildigi tarih. Ceza tablosu buna gore secilir; "
                        "verilmezse ceza HESAPLANMAZ (bugune dusulmez)")

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

    x = sub.add_parser("explain", help="Bir olayin kusur/ceza gerekcesi (Faz 5)")
    x.add_argument("rapor", help="runs/<kosu>/rapor.json")
    x.add_argument("--event", default=None, help="Tek olay; verilmezse hepsi")

    args = parser.parse_args(argv)
    if args.cmd == "eval":
        return evaluate(args.model, args.data, args.imgsz, args.batch, args.hedef)
    if args.cmd == "doctor":
        return doctor()
    if args.cmd == "run":
        return run(args.video, args.model, args.out, args.profile, args.conf,
                   args.detect, args.track_model, args.camera, args.rules,
                   args.tarih, args.damage_model)
    if args.cmd == "explain":
        return explain(args.rapor, args.event)
    if args.cmd == "calibrate":
        return calibrate(args.camera, args.verify)
    if args.cmd == "track":
        return track(args.video, args.model, args.out, args.camera, args.conf,
                     args.tracker, args.min_kare)
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
