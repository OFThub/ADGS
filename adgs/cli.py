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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="adgs",
        description="Trafik video analiz sistemi - karar destek araci (baglayici degildir)",
    )
    parser.add_argument("--version", action="version", version=f"adgs {__version__}")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("doctor", help="Ortam ve yapilandirma kontrolu")

    args = parser.parse_args(argv)
    if args.cmd == "doctor":
        return doctor()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
