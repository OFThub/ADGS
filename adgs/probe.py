"""Videodan analiz parametrelerini turetir - kullanicidan ek bilgi istemeden.

Cagiran: adgs.api.video_yukle (yukleme ucu). Turettigi dort alan cli.run'in
MEVCUT parametreleridir: profile, detect_tipleri, camera, tarih_metni. Yeni
bir sozlesme yok; sadece kullanicinin doldurdugu alanlar olculerek doluyor.

Kural: her deger ya bir OLCUME (kare farki) ya da dosya adindaki ACIK bir
isarete dayanir. Turetilemeyen alan BOS birakilir ve sebebi `gerekce`ye
yazilir. Bos alan ilgili modulun kendi reddetme mekanizmasini calistirir;
sessiz varsayilan kullanilmaz. Iki yerde bu ozellikle onemli:

  tarih  - bugune DUSULMEZ. Arsiv videosu bugunun ceza tablosuyla
           hesaplanirsa sessizce yanlis tutar uretir (bkz. cli._tarih_coz).
  kamera - cozunurlukten TAHMIN EDILMEZ. Yanlis kamera = yanlis dur cizgisi
           = bir vatandas adina yanlis ihlal kaydi. Yalnizca dosya adinda
           kamera kimligi aciken geciyorsa eslesir.
"""

from __future__ import annotations

import re
import statistics
from datetime import date
from pathlib import Path

import cv2
import numpy as np

# Kamera kaymasi bantlari: yarim saniyede kameranin kaydigi mesafe, kare
# genisliginin orani olarak. Sabit kamera ~0, araca monteli kamera buyuk.
#
# Olculdu (bu depodaki videolar, yarim saniyelik aralik, medyan):
#   demo_kaza.mp4     %0.004   sabit sahne
#   dashcam derleme   %0.20    duragan/yavas cekim kesitleri agirlikta
#   demo_pan.mp4      %19.3    kaydirilan kamera
# Iki sinif arasinda ~50x fark var; esigin tam yeri kritik degil.
#
# Neden piksel yogunlugu degil de kayma: bitisik karelerde 30 fps'te sahne
# birkac piksel kayar ve duz yuzeylerde (asfalt, gokyuzu) yogunluk farki
# esigin altinda kalir - dashcam goruntusu "sabit" olarak olculuyordu. Kayma
# dogrudan olculen fiziksel buyukluk, esige duyarli degil.
SABIT_UST = 0.01
HAREKETLI_ALT = 0.03
_ORNEK_SAYISI = 8
_ARALIK_SN = 0.5  # olcum araligi; fps'ten bagimsiz olmasi icin saniye cinsinden
_OLCUM_GENISLIK = 240  # olcum kucultulmus griden yapilir, ayrinti gerekmiyor

# Kamera kimligi veritabanina yazilip arayuzde gosteriliyor; bicim sinirli.
_KAMERA_DESENI = re.compile(r"^[A-Za-z0-9_.-]{1,64}$")

_TARIH_DESENLERI = (
    (re.compile(r"(20\d{2})[-_.](\d{2})[-_.](\d{2})"), (0, 1, 2)),      # 2026-07-15
    (re.compile(r"(\d{2})[-_.](\d{2})[-_.](20\d{2})"), (2, 1, 0)),      # 15.07.2026
    (re.compile(r"(?<!\d)(20\d{2})(\d{2})(\d{2})(?!\d)"), (0, 1, 2)),   # 20260715
)


def _kucult_gri(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    s = _OLCUM_GENISLIK / max(w, 1)
    if s < 1:
        img = cv2.resize(img, (_OLCUM_GENISLIK, max(1, round(h * s))),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def kamera_kaymasi(video: str | Path, ornek: int = _ORNEK_SAYISI) -> float | None:
    """Yarim saniyede kameranin kaydigi mesafe / kare genisligi. Medyan.

    Faz korelasyonu (cv2.phaseCorrelate) iki kare arasindaki BASKIN otelemeyi
    bulur. Sabit kamerada baskin bolge duragan arka plandir -> kayma ~0;
    araca monteli kamerada tum sahne kayar -> kayma buyuk. Onunden gecen tek
    bir kamyon baskin bolgeyi calabilir, bu yuzden 8 ornegin MEDYANI alinir.

    Hanning penceresi kare kenarlarinin yarattigi yapay tepeyi bastirir;
    penceresiz olcum duragan sahnelerde de sifirdan sapar.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    try:
        toplam = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        aralik = max(1, round(fps * _ARALIK_SN))
        if toplam < aralik + 2:
            return None
        son = toplam - aralik - 2
        pencere = None
        kaymalar: list[float] = []
        for i in range(ornek):
            n = round(i * son / max(ornek, 1))
            cap.set(cv2.CAP_PROP_POS_FRAMES, n)
            ok1, a = cap.read()
            cap.set(cv2.CAP_PROP_POS_FRAMES, n + aralik)
            ok2, b = cap.read()
            if not (ok1 and ok2):
                continue
            A, B = np.float32(_kucult_gri(a)), np.float32(_kucult_gri(b))
            if pencere is None:
                pencere = cv2.createHanningWindow((A.shape[1], A.shape[0]), cv2.CV_32F)
            (dx, dy), _ = cv2.phaseCorrelate(A, B, pencere)
            kaymalar.append(float(np.hypot(dx, dy)) / A.shape[1])
        return statistics.median(kaymalar) if kaymalar else None
    finally:
        cap.release()


def _profil_coz(kayma: float | None) -> tuple[str, str, str]:
    """(profil, detect, gerekce) - olcum yoksa veya kararsizsa ikisi de calisir."""
    if kayma is None:
        return ("cctv_fixed", "roaddamage,accident,ihlal",
                "profil: kamera kaymasi OLCULEMEDI (video kisa veya okunamadi)"
                " -> tahmin edilmedi, iki dedektor ailesi de calisti")
    y = f"%{kayma * 100:.2f}"
    if kayma < SABIT_UST:
        return ("cctv_fixed", "accident,ihlal",
                f"profil: kamera kaymasi {y} < %{SABIT_UST * 100:.0f}"
                " -> sabit kamera (kaza + ihlal taramasi)")
    if kayma > HAREKETLI_ALT:
        return ("vehicle_mounted", "roaddamage",
                f"profil: kamera kaymasi {y} > %{HAREKETLI_ALT * 100:.0f}"
                " -> araca monteli kamera (yol hasari taramasi)")
    return ("cctv_fixed", "roaddamage,accident,ihlal",
            f"profil: kamera kaymasi {y} KARARSIZ bantta"
            f" (%{SABIT_UST * 100:.0f}-%{HAREKETLI_ALT * 100:.0f})"
            " -> tahmin edilmedi, iki dedektor ailesi de calisti")


def tarih_bul(ad: str, bugun: date | None = None) -> tuple[str | None, str]:
    """Dosya adindaki cekim tarihi. Bulunamazsa None - bugune DUSULMEZ.

    CCTV/DVR disa aktarimlari tarihi dosya adinda tasir; bu kurulumda
    okunabilen tek kaynak o (ffprobe yok, OpenCV kayit tarihini vermiyor).
    """
    bugun = bugun or date.today()
    for desen, (y, a, g) in _TARIH_DESENLERI:
        m = desen.search(ad)
        if not m:
            continue
        try:
            d = date(int(m.group(y + 1)), int(m.group(a + 1)), int(m.group(g + 1)))
        except ValueError:
            continue  # 2026-13-45 gibi bir sayi dizisi tarih degildir
        if d > bugun:
            continue  # gelecek tarih cekim tarihi olamaz
        return d.isoformat(), f"tarih: dosya adindan okundu ({d.isoformat()})"
    return None, ("tarih: dosya adinda cekim tarihi yok -> BOS birakildi"
                  " (ceza tablosu secilemez, tutar hesaplanmaz)")


def kamera_bul(ad: str, dizin: Path) -> tuple[str | None, str]:
    """Dosya adinda gecen tanimli kamera kimligi.

    Cozunurlukten veya goruntu benzerliginden TAHMIN EDILMEZ: yanlis eslesme
    yanlis dur cizgisi ve serit yonu demektir, o da yanlis ihlal kaydi uretir.
    Bu projede precision recall'dan once gelir - eslesme kesin degilse yok.
    """
    if not dizin.exists():
        return None, "kamera: config/cameras dizini yok"
    # Kimlik veritabanina yazilip her izleyicinin sayfasinda gosteriliyor.
    # Dosya adi disaridan gelebildigi icin bicim BURADA suzuluyor - cagiran
    # tarafta degil (tek kaynak).
    kimlikler = sorted(p.stem for p in dizin.glob("*.yaml")
                       if _KAMERA_DESENI.match(p.stem))
    kucuk = ad.lower()
    for k in kimlikler:
        if k.lower() in kucuk:
            return k, f"kamera: dosya adinda '{k}' gecti -> geometri yuklendi"
    return None, ("kamera: dosya adinda tanimli kamera kimligi yok -> BOS"
                  " (geometriye bagli dedektorler kendini kapatacak; tanimli"
                  f" olanlar: {', '.join(kimlikler) or 'yok'})")


def incele(video: str | Path, kamera_dizini: Path | None = None,
           bugun: date | None = None) -> dict:
    """Videoyu inceleyip cli.run parametrelerini uretir.

    Donen sozluk dogrudan cli.run'a beslenir:
        profile -> profile, detect -> detect_tipleri,
        camera  -> camera,  tarih  -> tarih_metni
    `gerekce` her kararin nedenini tasir ve arayuzde gosterilir; turetilmis
    bir parametrenin neden oyle secildigi kullanicidan saklanmaz.
    """
    p = Path(video)
    dizin = kamera_dizini or (Path(__file__).resolve().parent.parent
                              / "config" / "cameras")
    kayma = kamera_kaymasi(p)
    profil, detect, g_profil = _profil_coz(kayma)
    tarih, g_tarih = tarih_bul(p.name, bugun=bugun)
    kamera, g_kamera = kamera_bul(p.name, dizin)
    return {
        "profile": profil,
        "detect": detect,
        "camera": kamera,
        "tarih": tarih,
        "kamera_kaymasi": kayma,
        "gerekce": [g_profil, g_tarih, g_kamera],
    }
