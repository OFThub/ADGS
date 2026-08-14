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


# Gece/dusuk isik esigi ve o durumda kullanilacak guven esigi.
#
# OLCULDU (bu depodaki 16 sahne, HSV V kanali medyani):
#   gunduz sahneler      126 - 154
#   alacakaranlik        104
#   gece (islak zemin)    81 -  87
#
# Ayni gece sahnesinde arac tespiti (17 karede toplam):
#   ham conf 0.35 ->  0     <- varsayilan ayar hicbir sey bulamiyordu
#   ham conf 0.15 ->  9
#   CLAHE     .35 ->  1
#   CLAHE     .15 ->  8     <- CLAHE kayda deger katki yapmiyor
#   CLAHE     .08 -> 32     <- yanlis pozitif riski yuksek
# Bu yuzden cozum on isleme degil, gece sahnesinde guven esigini dusurmek.
GECE_ESIGI = 100.0
GECE_CONF = 0.15
VARSAYILAN_CONF = 0.35

KESME_ESIGI = 0.35  # ardisik karelerin histogram korelasyonu bunun altina duserse kesme


def isik_seviyesi(video: str | Path, ornek: int = 6) -> float | None:
    """Sahnenin medyan parlakligi (HSV V kanali, 0-255).

    Medyan: tek bir far parlamasi veya karartma ortalamayi bozar, medyani bozmaz.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if n < 1:
            return None
        vals = []
        for i in range(ornek):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i * max(0, n - 1) / max(ornek, 1)))
            ok, img = cap.read()
            if ok:
                k = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
                vals.append(float(cv2.cvtColor(k, cv2.COLOR_BGR2HSV)[:, :, 2].mean()))
        return statistics.median(vals) if vals else None
    finally:
        cap.release()


def _conf_coz(isik: float | None) -> tuple[float, str]:
    """(guven esigi, gerekce) - gece sahnesinde esik dusurulur."""
    if isik is None:
        return VARSAYILAN_CONF, (
            f"guven esigi: isik olculemedi -> varsayilan {VARSAYILAN_CONF}")
    if isik < GECE_ESIGI:
        return GECE_CONF, (
            f"guven esigi: sahne parlakligi {isik:.0f}/255 < {GECE_ESIGI:.0f}"
            f" (gece/dusuk isik) -> esik {GECE_CONF} kullanildi."
            " Gece kayitlarinda arac isik lekesine dondugu icin varsayilan esik"
            " hic tespit uretmiyordu; dusuk esik YANLIS POZITIF riskini artirir.")
    return VARSAYILAN_CONF, (
        f"guven esigi: sahne parlakligi {isik:.0f}/255 -> varsayilan"
        f" {VARSAYILAN_CONF}")


def sahne_kesmeleri(video: str | Path, esik: float = KESME_ESIGI) -> list[int]:
    """Sahne kesmesi olan kare numaralari (montaj/derleme videolar icin).

    Neden gerekli: sistem TEK SUREKLI kamera varsayar. Derleme videoda kesme
    aninda izler aniden biter, farkli sahnelerin kutulari cakisir ve konum
    sicramasi "ani hareket degisimi" sinyalini TAKLIT eder. Olculdu: 5 dakikalik
    bir derlemede sinyal-2'yi gecen 14 adayin 10'u kesme kaynakliydi.

    Olcum HSV histogram korelasyonu: kamera degisince renk dagilimi da degisir.
    Piksel farki yerine histogram, cunku hizli pan da buyuk piksel farki uretir
    ama renk dagilimini buyuk olcude korur.
    """
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return []
    kesmeler: list[int] = []
    onceki = None
    idx = 0
    try:
        while True:
            ok, img = cap.read()
            if not ok:
                break
            k = cv2.resize(img, (160, 90), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(k, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1], None, [32, 32], [0, 180, 0, 256])
            cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
            if onceki is not None and \
                    cv2.compareHist(onceki, h, cv2.HISTCMP_CORREL) < esik:
                kesmeler.append(idx)
            onceki = h
            idx += 1
    finally:
        cap.release()
    return kesmeler


def _profil_coz(kayma: float | None) -> tuple[str, str, str]:
    """(profil, detect, gerekce) - olcum yoksa veya kararsizsa ikisi de calisir."""
    if kayma is None:
        return ("cctv_fixed", "roaddamage,accident,ihlal",
                "profil: kamera kaymasi OLCULEMEDI (video kisa veya okunamadi)"
                " -> tahmin edilmedi, iki dedektor ailesi de calisti")
    y = f"%{kayma * 100:.2f}"
    if kayma < SABIT_UST:
        # Yol hasari sabit kamerada DA taranir: kavsakta da cukur olabilir ve
        # kullanici "kaza veya yol hasari olunca goster" istedi. Ancak RDD2022
        # tamamen ARACA MONTELI, yol seviyesi goruntudur; sabit CCTV'nin egik
        # ve uzak acisinda cukur birkac piksele duser - getirisi dusuktur ve
        # buldugu seyler dogrulanmalidir (plan, varsayim duzeltmesi 2).
        return ("cctv_fixed", "accident,ihlal,roaddamage",
                f"profil: kamera kaymasi {y} < %{SABIT_UST * 100:.0f}"
                " -> sabit kamera (kaza + ihlal; yol hasari da taranir ama"
                " sabit CCTV acisi bu model icin uygun degil)")
    if kayma > HAREKETLI_ALT:
        return ("vehicle_mounted", "roaddamage",
                f"profil: kamera kaymasi {y} > %{HAREKETLI_ALT * 100:.0f}"
                " -> araca monteli kamera (yol hasari taramasi)")
    return ("cctv_fixed", "roaddamage,accident,ihlal",
            f"profil: kamera kaymasi {y} KARARSIZ bantta"
            f" (%{SABIT_UST * 100:.0f}-%{HAREKETLI_ALT * 100:.0f})"
            " -> tahmin edilmedi, iki dedektor ailesi de calisti")


def tarih_bul(ad: str, bugun: date | None = None,
              bugune_dus: bool = False) -> tuple[str | None, str]:
    """Dosya adindaki cekim tarihi.

    CCTV/DVR disa aktarimlari tarihi dosya adinda tasir; bu kurulumda
    okunabilen tek kaynak o (ffprobe yok, OpenCV kayit tarihini vermiyor).

    `bugune_dus=True` ise okunamayan tarih icin BUGUN kullanilir. Bu bir
    OLCUM DEGIL VARSAYIMDIR ve gerekcede "VARSAYILDI" diye isaretlenir:
    arsiv videosu bugunun ceza tablosuyla hesaplanirsa tutar yanlis cikar.
    Isaretin amaci, yanlis cikma ihtimalinin ciktida gorunur kalmasi.
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
    if bugune_dus:
        return bugun.isoformat(), (
            f"tarih: dosya adinda yok -> BUGUN VARSAYILDI ({bugun.isoformat()})."
            " Olculmus bir cekim tarihi DEGILDIR; video arsivden ise o gunun"
            " ceza tablosu secilmemis olur ve tutar yanlis cikabilir.")
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
           bugun: date | None = None, tarih_bugune_dus: bool = True) -> dict:
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
    tarih, g_tarih = tarih_bul(p.name, bugun=bugun, bugune_dus=tarih_bugune_dus)
    kamera, g_kamera = kamera_bul(p.name, dizin)
    isik = isik_seviyesi(p)
    conf, g_conf = _conf_coz(isik)
    return {
        "profile": profil,
        "detect": detect,
        "camera": kamera,
        "tarih": tarih,
        "conf": conf,
        "kamera_kaymasi": kayma,
        "isik": isik,
        "gerekce": [g_profil, g_tarih, g_kamera, g_conf],
    }
