"""Kabul kriterlerinin tek komutta olculmesi (plan 7. ve 8. bolum).

Cagiran: adgs.cli.kabul (`adgs kabul`). Bu modulu baska hicbir sey import
etmez; tek yonlu bir raporlama katmani.

Tek isi durustluk: plandaki her kabul kriteri burada listelenir ve ucunden
birini alir -

    GECTI       olculdu ve hedefi tutturdu
    KALDI       OLCULDU ve hedefin altinda kaldi
    OLCULMEDI   olcum kosulu yok (model egitilmemis veya etiketli veri yok)

"Kod yazildi" bir durum DEGILDIR. Bir kriterin olculememesi basarisizlik
degildir ama basari olarak da gosterilemez; sebebi ve o sebebi kaldirmak icin
gerekeni `not_` alani tasir.

Veriye bagli kriterler (etiketli Arnavutkoy goruntusu isteyenler) OLCULMEDI
olarak durur. Bunlari sentetik veriyle "olcmek" tabloyu yesile boyardi ve
hicbir sey kanitlamazdi.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent

GECTI, KALDI, OLCULMEDI = "GECTI", "KALDI", "OLCULMEDI"


@dataclass
class Kriter:
    faz: str
    ad: str
    hedef: str
    durum: str
    deger: str = "-"
    not_: str = ""
    kanit: str = ""


@dataclass
class Sonuc:
    kriterler: list[Kriter] = field(default_factory=list)

    def ekle(self, *a, **k) -> None:
        self.kriterler.append(Kriter(*a, **k))

    def sayim(self) -> dict[str, int]:
        d = {GECTI: 0, KALDI: 0, OLCULMEDI: 0}
        for k in self.kriterler:
            d[k.durum] = d.get(k.durum, 0) + 1
        return d


# --- olcum yardimcilari ------------------------------------------------------


def _kabul_json(dizin: str) -> dict | None:
    """`adgs eval` tarafindan yazilan olcum dosyasi."""
    p = KOK / "runs" / dizin / "kabul.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _model_kriteri(s: Sonuc, faz: str, ad: str, dizin: str, hedef: float,
                   komut: str) -> None:
    """mAP kriteri: olcum dosyasindan okunur, val kosusu burada TEKRARLANMAZ."""
    d = _kabul_json(dizin)
    if d is None:
        agirlik = KOK / "runs" / dizin / "weights" / "best.pt"
        neden = ("model egitilmemis" if not agirlik.exists()
                 else "model var ama olcum yapilmamis")
        s.ekle(faz, ad, f"mAP@0.5 >= {hedef}", OLCULMEDI, "-",
               f"{neden} -> `{komut}`")
        return
    m = float(d.get("map50", 0.0))
    s.ekle(faz, ad, f"mAP@0.5 >= {hedef}", GECTI if m >= hedef else KALDI,
           f"{m:.4f}", f"olcum: {d.get('zaman', '?')}",
           f"runs/{dizin}/kabul.json")


def _veri_bekleyen(s: Sonuc, faz: str, ad: str, hedef: str, gereken: str) -> None:
    s.ekle(faz, ad, hedef, OLCULMEDI, "-", f"etiketli veri gerekli: {gereken}")


def _kalibrasyon(s: Sonuc) -> None:
    """Her kamera tanimi icin geri donusum hatasi ve reddetme davranisi."""
    from adgs import calib

    dizin = KOK / "config" / "cameras"
    dosyalar = sorted(dizin.glob("*.yaml")) if dizin.exists() else []
    if not dosyalar:
        s.ekle("2", "Kalibrasyon dogrulamasi", "hata < %10", OLCULMEDI, "-",
               "config/cameras altinda kamera tanimi yok")
        return
    for p in dosyalar:
        try:
            k = calib.yukle(p)
        except Exception as e:  # noqa: BLE001 - bozuk tanim da bir sonuctur
            s.ekle("2", f"Kalibrasyon: {p.stem}", "hata < %10", KALDI, "-",
                   f"tanim okunamadi: {type(e).__name__}: {e}")
            continue
        if k.hata_yuzde is None:
            # Bagimsiz dogrulama noktasi yoksa kalibrasyon REDDEDILMELI.
            # Reddetmek DOGRU davranis oldugu icin bu GECTI'dir, KALDI degil.
            s.ekle("2", f"Kalibrasyon: {p.stem}",
                   "dogrulamasiz tanim reddedilmeli",
                   GECTI if not k.gecerli else KALDI, "dogrulama yok",
                   "bagimsiz olcum girilmemis -> hiza bagli moduller kapali"
                   if not k.gecerli else "DOGRULAMASIZ TANIM KABUL EDILDI - HATA",
                   str(p.relative_to(KOK)))
            continue
        s.ekle("2", f"Kalibrasyon: {p.stem}", f"hata < %{k.maks_hata_yuzde:.0f}",
               GECTI if k.gecerli else KALDI, f"%{k.hata_yuzde:.2f}",
               "" if k.gecerli else "hiza bagli moduller kapali",
               str(p.relative_to(KOK)))


def _hiz_reddi(s: Sonuc) -> None:
    """Faz 4'un asil kriteri: kalibrasyon yoksa hiz modulu CALISMAYI REDDEDER."""
    from adgs.violations import speed
    from adgs.violations.base import Baglam

    ctx = Baglam(video=Path("yok.mp4"), fps=25.0, kalib=None)
    olaylar = speed.tespit_et([], ctx)
    reddetti = not olaylar and any("TESPIT YAPILMADI" in u for u in ctx.uyarilar)
    s.ekle("4", "Hiz modulu kalibrasyonsuz reddediyor", "cikti uretmemeli",
           GECTI if reddetti else KALDI,
           "reddetti" if reddetti else "CIKTI URETTI",
           ctx.uyarilar[0] if ctx.uyarilar else "uyari yazilmadi",
           "adgs/violations/speed.py")


def _ceza_kodda_sayi_yok(s: Sonuc) -> None:
    """penalty.py icinde tutar gomulu olmamali (plan: grep ile dogrulanir)."""
    import re

    p = KOK / "adgs" / "penalty.py"
    if not p.exists():
        s.ekle("5", "penalty.py'de gomulu tutar yok", "4+ haneli sayi olmamali",
               OLCULMEDI, "-", "adgs/penalty.py yok")
        return
    bulunan = re.findall(r"\b\d{4,}\b", p.read_text(encoding="utf-8"))
    s.ekle("5", "penalty.py'de gomulu tutar yok", "4+ haneli sayi olmamali",
           GECTI if not bulunan else KALDI,
           "yok" if not bulunan else ", ".join(bulunan[:5]),
           "tutarlar yalnizca config/penalties/<tarih>.yaml icinde",
           "adgs/penalty.py")


def _ceza_tablosu_yoksa_reddediyor(s: Sonuc) -> None:
    """Uygun tablo yoksa hesaplama YAPILMAMALI - eski tabloya dusmek yasak."""
    from datetime import date

    from adgs import penalty

    try:
        tablo = penalty.tablo_sec(date(1990, 1, 1), KOK / "config" / "penalties")
    except Exception as e:  # noqa: BLE001
        s.ekle("5", "Tablo yoksa hesaplama reddediliyor", "None donmeli",
               OLCULMEDI, "-", f"{type(e).__name__}: {e}")
        return
    s.ekle("5", "Tablo yoksa hesaplama reddediliyor", "None donmeli",
           GECTI if tablo is None else KALDI,
           "reddetti" if tablo is None else "TABLO SECTI",
           "1990 tarihli olay icin gecerli tablo yok", "adgs/penalty.py")


def _mevzuat_dogrulamasi(s: Sonuc) -> None:
    """Hukuki tablolar bir insan tarafindan dogrulanmis mi?

    Idari bir kriter ve KOD ILE SAGLANAMAZ. Tabloda durmasinin sebebi tam
    olarak bu: yazilim yesil gorunurken tablolar dogrulanmamis olabilir.
    """
    import yaml

    dosyalar = [KOK / "config" / "rules" / "fault_ktk84.yaml"]
    dosyalar += sorted((KOK / "config" / "penalties").glob("*.yaml"))
    for p in dosyalar:
        if not p.exists():
            continue
        try:
            d = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        except (OSError, ValueError) as e:
            s.ekle("5", f"Mevzuat dogrulamasi: {p.name}", "dogrulama_tarihi dolu",
                   KALDI, "-", f"okunamadi: {e}")
            continue
        meta = d.get("meta", d)
        tarih = meta.get("dogrulama_tarihi") if isinstance(meta, dict) else None
        s.ekle("5", f"Mevzuat dogrulamasi: {p.name}", "dogrulama_tarihi dolu",
               GECTI if tarih else OLCULMEDI, str(tarih or "null"),
               "" if tarih else
               "mevzuat.gov.tr / trafik.gov.tr ile karsilastirilmali; motor "
               "her ciktiya DOGRULANMAMISTIR notu basiyor",
               str(p.relative_to(KOK)))


def _api_uclari(s: Sonuc) -> None:
    """Faz 7: uclar ayakta mi (sunucu baslatmadan, TestClient ile)."""
    try:
        from fastapi.testclient import TestClient

        from adgs import api
    except ImportError as e:
        s.ekle("7", "API uclari yanit veriyor", "200 donmeli", OLCULMEDI, "-",
               f"bagimlilik yok: {e} -> pip install -e .[api]")
        return
    uclar = ["/", "/videos", "/events", "/kara-nokta", "/kameralar"]
    try:
        with TestClient(api.app) as c:
            kodlar = {u: c.get(u).status_code for u in uclar}
    except Exception as e:  # noqa: BLE001
        s.ekle("7", "API uclari yanit veriyor", "200 donmeli", KALDI, "-",
               f"{type(e).__name__}: {e}")
        return
    kotu = {u: k for u, k in kodlar.items() if k != 200}
    s.ekle("7", "API uclari yanit veriyor", f"{len(uclar)} uc 200 donmeli",
           GECTI if not kotu else KALDI,
           f"{len(uclar) - len(kotu)}/{len(uclar)}",
           "" if not kotu else f"200 donmeyen: {kotu}", "adgs/api.py")


def _tek_girdi_video(s: Sonuc) -> None:
    """Yukleme yalnizca dosya istemeli; kalani videodan turetilmeli."""
    import inspect

    try:
        from adgs import api
    except ImportError as e:
        s.ekle("7", "Yukleme sadece video istiyor", "ek alan sorulmamali",
               OLCULMEDI, "-", str(e))
        return
    fazla = set(inspect.signature(api.video_yukle).parameters) - {"arkaplan", "file"}
    s.ekle("7", "Yukleme sadece video istiyor", "ek alan sorulmamali",
           GECTI if not fazla else KALDI,
           "sadece dosya" if not fazla else f"fazladan alan: {sorted(fazla)}",
           "profil/dedektor/kamera/tarih adgs.probe ile videodan turetiliyor",
           "adgs/probe.py")


def _testler(s: Sonuc) -> None:
    """Zorunlu birim testler (plan 8: test_fault + test_penalty)."""
    import subprocess
    import sys

    hedefler = ["tests/test_fault.py", "tests/test_penalty.py"]
    var = [h for h in hedefler if (KOK / h).exists()]
    if not var:
        s.ekle("5", "Zorunlu birim testler", "hepsi gecmeli", OLCULMEDI, "-",
               "test dosyalari yok")
        return
    r = subprocess.run([sys.executable, "-m", "pytest", *var, "-q"],
                       cwd=KOK, capture_output=True, text=True)
    son = [x for x in r.stdout.strip().splitlines() if x.strip()]
    s.ekle("5", "Zorunlu birim testler (kusur + ceza)", "hepsi gecmeli",
           GECTI if r.returncode == 0 else KALDI,
           son[-1].strip() if son else f"cikis {r.returncode}",
           " ".join(var), "pytest")


# --- tablo -------------------------------------------------------------------


def olc() -> Sonuc:
    """Tum kabul kriterlerini olcer ve tabloyu doner."""
    s = Sonuc()

    # Faz 0. CPU derlemesi sessiz bir tuzak: her sey calisir, sadece
    # yavaslar - bu yuzden torch surumu de yaziliyor.
    try:
        import torch

        ok = bool(torch.cuda.is_available())
        surum = torch.__version__
        ad = torch.cuda.get_device_name(0) if ok else (
            f"torch {surum} CPU derlemesi - GPU yok sayiliyor; CUDA'li surum: "
            "pip install torch --index-url https://download.pytorch.org/whl/cu124"
            if "+cpu" in surum else f"torch {surum}, CUDA gorunmuyor (surucu?)")
    except Exception as e:  # noqa: BLE001
        ok, ad = False, f"torch yok ({type(e).__name__})"
    s.ekle("0", "CUDA kullanilabilir", "True", GECTI if ok else KALDI, str(ok),
           "" if ok else ad, ad if ok else "")

    # Faz 1
    _model_kriteri(s, "1", "Yol hasari tespiti (RDD2022)", "rdd2022/yolo26s", 0.45,
                   "adgs eval --data data/rdd_yolo/rdd2022.yaml --hedef 0.45")
    _veri_bekleyen(s, "1", "Tekillestirme: tek cukur tek olay", "1 hasar = 1 olay",
                   "GPS'li hizmet araci guzergahi (2-3 km, etiketli)")

    # Faz 2
    _veri_bekleyen(s, "2", "Arac tespiti", "mAP@0.5 >= 0.70",
                   "etiketli Arnavutkoy kavsak kareleri (300-500)")
    _veri_bekleyen(s, "2", "Yaya tespiti", "mAP@0.5 >= 0.60",
                   "etiketli Arnavutkoy kavsak kareleri")
    _veri_bekleyen(s, "2", "Takip tutarliligi", "IDF1 >= 0.60",
                   "MOT bicimli iz dogrulugu (elle etiketli klip)")
    _veri_bekleyen(s, "2", "Plaka OCR", "karakter dogrulugu >= %85",
                   "TR plaka veri kumesi (acik kaynak pratikte yok)")
    _kalibrasyon(s)

    # Faz 3
    _veri_bekleyen(s, "3", "Kaza tespiti", "precision >= 0.80",
                   "gercek kaza goruntusu (sabit CCTV acisi)")
    _veri_bekleyen(s, "3", "Yanlis pozitif orani", "klip basina <= 1",
                   "gercek kaza goruntusu")

    # Faz 4
    _veri_bekleyen(s, "4", "Ihlal tespiti (tip basina)", "precision >= 0.85",
                   "etiketli ihlal goruntusu (tip basina >= 30 ornek)")
    _hiz_reddi(s)

    # Faz 5
    _testler(s)
    _ceza_kodda_sayi_yok(s)
    _ceza_tablosu_yoksa_reddediyor(s)
    _mevzuat_dogrulamasi(s)

    # Faz 6
    _model_kriteri(s, "6", "Arac hasari (CarDD)", "cardd/yolo26s", 0.40,
                   "adgs eval --model runs/cardd/yolo26s/weights/best.pt"
                   " --data data/cardd_yolo/cardd.yaml --hedef 0.40")

    # Faz 7
    _api_uclari(s)
    _tek_girdi_video(s)
    return s


def bicimle(s: Sonuc) -> str:
    """Tabloyu metne cevirir."""
    basliklar = ("FAZ", "KRITER", "HEDEF", "DEGER", "DURUM")
    satirlar = [(k.faz, k.ad, k.hedef, k.deger, k.durum) for k in s.kriterler]
    gen = [max(len(str(r[i])) for r in (basliklar, *satirlar)) for i in range(5)]

    def ciz(r) -> str:
        return "  ".join(str(r[i]).ljust(gen[i]) for i in range(5)).rstrip()

    out = [ciz(basliklar), "  ".join("-" * g for g in gen)]
    for k, r in zip(s.kriterler, satirlar):
        out.append(ciz(r))
        if k.not_:
            out.append(" " * (gen[0] + 2) + f"  -> {k.not_}")
    c = s.sayim()
    out += [
        "",
        f"GECTI: {c[GECTI]}   KALDI: {c[KALDI]}   OLCULMEDI: {c[OLCULMEDI]}",
        "",
        "OLCULMEDI bir basarisizlik degildir ama basari da degildir: olcum",
        "kosulu (egitilmis model veya etiketli veri) henuz yok. Her satirin",
        "sebebi ve o sebebi kaldirmak icin gerekeni yukarida yazili.",
    ]
    return "\n".join(out)
