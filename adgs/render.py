"""M9 - Isaretlenmis video ve JSON rapor."""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path

import cv2

from adgs.schema import Event

_RENK = {"YUKSEK": (0, 0, 220), "ORTA": (0, 165, 255), "DUSUK": (0, 200, 200)}
_VARSAYILAN_RENK = (0, 200, 0)
_FILIGRAN = "ON DEGERLENDIRME - BAGLAYICI DEGILDIR"
# Bir olayin ekranda kalacagi ASGARI sure. Kanit karesi cogu olayda birkac
# karedir; 30 fps'te 2 kare = 0.07 sn ve izlerken hic fark edilmez.
_ASGARI_SN = 2.0

# --- KVKK bulaniklastirma ---------------------------------------------------
# Video kisisel veri icerir (yuz, plaka). Sunum ve arsiv icin varsayilan ACIK.
KVKK_VARSAYILAN = {"plaka": True, "yuz": True}

# Yaya kutusunun ust bu orani bas bolgesi sayilir.
_BAS_ORANI = 0.35
# Arac kutusunun alt bu orani ve ortadaki bu genislik plaka bolgesi sayilir.
_PLAKA_ALT_ORANI = 0.30
_PLAKA_GENISLIK_ORANI = 0.55

_ARAC = {"otomobil", "kamyon", "otobus", "motosiklet"}


def kvkk_bolgeleri(iz, f: int, plaka: bool = True,
                   yuz: bool = True) -> list[tuple[int, int, int, int]]:
    """Bir izin f karesinde bulaniklastirilacak bolgeleri dondurur.

    ponytail: bolgeler GEOMETRIK YAKLASIMDIR - ayri bir plaka/yuz dedektoru
    yok. Yaya kutusunun ustu bas, arac kutusunun alt-ortasi plaka kabul edilir.
    Fazladan alan bulaniklastirmak, eksik bulaniklastirmaktan iyidir; gercek
    dedektor eklenirse yalnizca bu fonksiyon degisir.
    """
    det = iz.frames.get(f)
    if det is None:
        return []
    x1, y1, x2, y2 = det.bbox
    g, y = x2 - x1, y2 - y1
    if g <= 0 or y <= 0:
        return []
    if yuz and iz.cls in ("yaya", "bisiklet"):
        return [(x1, y1, x2, y1 + max(1, int(y * _BAS_ORANI)))]
    if plaka and iz.cls in _ARAC:
        pg = max(1, int(g * _PLAKA_GENISLIK_ORANI))
        px = x1 + (g - pg) // 2
        return [(px, y2 - max(1, int(y * _PLAKA_ALT_ORANI)), px + pg, y2)]
    return []


def bulaniklastir(img, bolgeler: list[tuple[int, int, int, int]]) -> None:
    """Verilen bolgeleri YERINDE bulaniklastirir.

    Cekirdek bolge boyutuna gore olceklenir: sabit cekirdek, buyuk bolgelerde
    plakayi okunur birakir.
    """
    h, w = img.shape[:2]
    for x1, y1, x2, y2 in bolgeler:
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < 2 or y2 - y1 < 2:
            continue
        roi = img[y1:y2, x1:x2]
        k = max(3, (min(roi.shape[:2]) // 2) | 1)  # cekirdek tek sayi olmali
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)


def _kvkk_uygula(img, izler, f: int, kvkk: dict | None) -> None:
    if not kvkk or not izler:
        return
    bolgeler: list[tuple[int, int, int, int]] = []
    for iz in izler:
        bolgeler += kvkk_bolgeleri(iz, f, plaka=kvkk.get("plaka", True),
                                   yuz=kvkk.get("yuz", True))
    bulaniklastir(img, bolgeler)


def _siddet_of(evt: Event) -> str:
    for n in evt.notes:
        for s in ("YUKSEK", "ORTA", "DUSUK"):
            if f"siddet {s}" in n:
                return s
    return ""


def sn_mmss(sn: float) -> str:
    """Saniyeyi dd:ss bicimine cevirir (1 saatten uzun video beklenmiyor)."""
    sn = max(0, int(sn))
    return f"{sn // 60:02d}:{sn % 60:02d}"


def _etiket(evt: Event) -> str:
    # TIP basta: izleyenin ilk gormesi gereken sey "kaza mi, yol hasari mi".
    # Onceki bicim (event_id + alt_tip) tipi hic yazmiyordu; videoda
    # "kaza_1 CARPISMA" goren biri bunun KAZA oldugunu etiketten cikaramiyordu.
    #
    # ZAMAN da etikette: 5 dakikalik bir kayitta "hangi saniyede" sorusu
    # kutunun kendisi kadar onemli - not alip geri sarmayi mumkun kilar.
    s = _siddet_of(evt)
    return (f"{sn_mmss(evt.t_start)} {evt.tip} - {evt.alt_tip} ({evt.event_id})"
            + (f" [{s}]" if s else ""))


def _izleri_ciz(img, izler, f: int) -> None:
    """Tum takip kutularini ince cizgiyle cizer (olay kutularinin ALTINA)."""
    if not izler:
        return
    for iz in izler:
        det = iz.frames.get(f)
        if det is None:
            continue
        x1, y1, x2, y2 = det.bbox
        renk = _SINIF_RENK.get(iz.cls, _VARSAYILAN_RENK)
        cv2.rectangle(img, (x1, y1), (x2, y2), renk, 1)
        cv2.putText(img, f"#{iz.track_id} {iz.cls}", (x1, max(11, y1 - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, renk, 1, cv2.LINE_AA)


CRF = 23  # 18 gorsel olarak kayipsiz, 28 belirgin bozulma; 23 kanit icin yeterli


class _FFmpegYazici:
    """Ham kareleri ffmpeg'e boru ile besleyen yazici.

    Neden OpenCV yetmiyor: bu OpenCV derlemesi H.264 bit hizini AYARLAMIYOR -
    OPENCV_FFMPEG_WRITER_OPTIONS (crf, video_bitrate) yok sayiliyor ve icerik
    ne olursa olsun ~24 Mbit/sn sabit yaziyor. 5 dakikalik 720p bir video 963 MB
    cikiyordu (kaynak 24 MB). CRF ile ayni goruntu ~40 MB.

    Ayrica -movflags +faststart: moov atomu dosya BASINA tasinir, boylece
    tarayici dosyanin tamami inmeden oynatmaya baslar. 5 dakikalik bir kayitta
    fark "aninda oynuyor" ile "once tamamini indir" arasindadir.

    cv2.VideoWriter arayuzunu taklit eder (write/release/isOpened) - cagiran
    taraf hangi yazicinin kullanildigini bilmez.
    """

    def __init__(self, exe: str, out_path: Path, fps: float, w: int, h: int):
        import subprocess

        self._p = subprocess.Popen(
            [exe, "-y", "-loglevel", "error",
             "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", f"{fps:.6f}", "-i", "-",
             "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p",
             "-crf", str(CRF), "-preset", "veryfast",
             "-movflags", "+faststart", str(out_path)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL)

    def write(self, img) -> None:
        try:
            self._p.stdin.write(img.tobytes())
        except (BrokenPipeError, OSError):
            pass  # ffmpeg oldu; release() cikis kodunu raporlar

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        if self._p.stdin and not self._p.stdin.closed:
            try:
                self._p.stdin.close()
            except OSError:
                pass
        if self._p.wait() != 0:
            print(f"[UYARI] ffmpeg kodlama hatasi (cikis {self._p.returncode})")


def _ffmpeg_yolu() -> str | None:
    """Varsa ffmpeg calistirilabiliri. Zorunlu bagimlilik DEGILDIR."""
    import shutil

    p = shutil.which("ffmpeg")
    if p:
        return p
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:  # noqa: BLE001 - yoksa OpenCV yazicisina dusulur
        return None


def _yazici(out_path: Path, fps: float, w: int, h: int):
    """MP4 yazici - TARAYICIDA OYNAYAN kodekle.

    mp4v (MPEG-4 Part 2) hicbir tarayicida oynamaz: .mp4 yalnizca KAPTIR,
    tarayici icindeki kodege bakar ve Chrome/Firefox/Edge/Safari H.264 (avc1),
    VP8/9 veya AV1 ister. mp4v ile uretilen dosya VLC'de acilir, sunucu 200
    doner, oynatici siyah kalir - hata hicbir yerde gorunmedigi icin en can
    sikici hali.

    Sira: ffmpeg (H.264 + makul boyut + faststart) -> OpenCV avc1 (H.264 ama
    cok buyuk) -> OpenCV mp4v (SON CARE, tarayicida oynamaz, sebebi yazilir).
    """
    exe = _ffmpeg_yolu()
    if exe:
        return _FFmpegYazici(exe, out_path, fps, w, h)
    yz = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"avc1"), fps, (w, h))
    if yz.isOpened():
        return yz
    print(f"[UYARI] H.264 kodlayici yok - {out_path.name} mp4v ile yaziliyor ve "
          "TARAYICIDA OYNAMAZ (VLC'de acilir). Cozum: pip install imageio-ffmpeg")
    return cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))


def _olay_plani(events: list[Event], fps: float) -> tuple[dict[int, list], dict[int, list]]:
    """(kare -> kutular, kare -> bant etiketleri) planini uretir.

    Kanit karesi (`evidence["kareler"]`) cogu olayda yalnizca birkac karedir;
    sadece o karelere cizmek 5 dakikalik videoda kazayi 2 kare gosterir ve
    izlerken FARK EDILMEZ. Kutu olayin TUM suresi boyunca tutulur, kanit
    kareleri arasindaki bosluklarda en yakin bilinen kutuya yapisir.

    Ayrica her olay en az _ASGARI_SN saniye ekranda kalir: bir karelik bir
    ihlal de izleyicinin gorebilecegi kadar surer.
    """
    kutular: dict[int, list] = {}
    bantlar: dict[int, list] = {}
    asgari = max(1, int(fps * _ASGARI_SN))
    for evt in events:
        renk = _RENK.get(_siddet_of(evt), _VARSAYILAN_RENK)
        etiket = _etiket(evt)
        kareler = {int(f): b for f, b in (evt.evidence.get("kareler") or {}).items()}
        if not kareler:
            continue
        bilinen = sorted(kareler)
        bas = max(0, min(evt.frame_start, bilinen[0]))
        son = max(evt.frame_end, bilinen[-1], bas + asgari - 1)
        i = 0
        for f in range(bas, son + 1):
            # En yakin bilinen kanit karesine tutun (tek yonlu tarama).
            while i + 1 < len(bilinen) and abs(bilinen[i + 1] - f) <= abs(bilinen[i] - f):
                i += 1
            kutular.setdefault(f, []).append((kareler[bilinen[i]], etiket, renk))
            bantlar.setdefault(f, []).append((etiket, renk))
    return kutular, bantlar


def _bant_ciz(img, etiketler: list, w: int) -> None:
    """Ust bant: olay suresince ekranda kalan okunakli uyari seridi.

    Kucuk veya kare kenarindaki kutu gozden kacar; bant kacmaz.
    """
    yuk = 24 * len(etiketler)
    bolge = img[0:yuk, 0:w]
    koyu = bolge.copy()
    koyu[:] = (0, 0, 0)
    cv2.addWeighted(koyu, 0.5, bolge, 0.5, 0, bolge)
    for i, (etiket, renk) in enumerate(etiketler):
        cv2.putText(img, etiket, (10, 17 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, renk, 2, cv2.LINE_AA)


def annotate_video(video: str | Path, events: list[Event], out_path: str | Path,
                   izler: list | None = None, kvkk: dict | None = None,
                   izleri_ciz: bool = False) -> Path:
    """Olaylarin bbox'larini kaynak videoya cizip yeni bir MP4 yazar.

    `kvkk` verilirse (izler ile birlikte) plaka/yuz bolgeleri CIZIMDEN ONCE
    bulaniklastirilir - isaretleme bulanik alanin uzerine biner, altina degil.

    `izleri_ciz` acikken tum takip kutulari da ince cizgiyle cizilir. Olay
    bulunmayan bir videoda cikti aksi halde ham videodan farksiz gorunur; oysa
    sistem her araci kalici ID ile izliyor ve bunu gostermemek yaptigi isi
    gizlemek olur. Olay kutulari bunun UZERINE kalin ve siddet renginde biner.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # ATOMIK YAZIM: once yan dosyaya yaz, bitince yerine tasi.
    #
    # Bu dosya ayni anda tarayiciya SUNULUYOR. Uzerine dogrudan yazmak izleyene
    # yarim bir MP4 gosterir: sure bilgisi henuz yazilmadigi icin oynatici
    # birkac saniye sonra videoyu bitmis sayip sona atlar. Yeniden analiz her
    # zaman mumkun oldugundan bu "arada bir" degil HER kosuda olurdu.
    # Uzanti KORUNUR: ffmpeg kap bicimini uzantidan secer, ".yaziliyor" ile
    # biten bir ada yazamaz. Ad "*_annotated.mp4" kalibina da uymaz, boylece
    # api._isaretli_video() yarim dosyayi bulup sunmaz.
    gecici = out_path.with_name(f"{out_path.stem}.yaziliyor{out_path.suffix}")

    watermark = any(evt.tip in ("KAZA", "IHLAL") for evt in events)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"video acilamadi: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    plan, bantlar = _olay_plani(events, fps)
    writer = _yazici(gecici, fps, w, h)
    try:
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            _kvkk_uygula(img, izler, idx, kvkk)
            if izleri_ciz:
                _izleri_ciz(img, izler, idx)
            for bbox, etiket, renk in plan.get(idx, []):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), renk, 3)
                cv2.putText(img, etiket, (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 1, cv2.LINE_AA)
            if idx in bantlar:
                _bant_ciz(img, bantlar[idx], w)
            if watermark:
                cv2.putText(img, _FILIGRAN, (8, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            writer.write(img)
            idx += 1
    finally:
        cap.release()
        writer.release()
    # os.replace atomiktir: izleyen ya ESKI ya YENI dosyayi gorur, yarisini asla.
    os.replace(gecici, out_path)
    return out_path


ON_TAMPON_SN = 3.0    # olaydan ONCE gosterilecek sure
ARKA_TAMPON_SN = 2.0  # olaydan SONRA gosterilecek sure


def save_clip(video: str | Path, frame_start: int, frame_end: int,
              out_path: str | Path, tampon: int | None = None,
              izler: list | None = None, kvkk: dict | None = None) -> Path | None:
    """Olay araligini ayri bir MP4 olarak kesip yazar. Kare yazilamazsa None.

    Tampon ASIMETRIKTIR ve saniye cinsindendir. Onceki hali iki yana da 15 kare
    (30 fps'te 0.5 sn) koyuyordu; olayin kendisi 2+ saniye surdugu icin klip
    neredeyse tamamen carpisma SONRASINI gosteriyor, izleyen "kaza nerede?"
    diye soruyordu. Kanit klibi olayin BASLADIGI ani icermeli: yaklasma ->
    carpma -> sonrasi. Bu yuzden one 3 sn, arkaya 2 sn.

    `tampon` (kare cinsinden) verilirse iki yana da o uygulanir - eski davranis.

    ponytail: OpenCV ile yeniden kodlar (kalite/hiz kaybi). ffmpeg -ss ile
    kodlamadan kesmek daha iyi olurdu; ffmpeg zorunlu bagimlilik olmasin diye
    gecildi - klip sayisi artarsa ffmpeg'e gecilmeli.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"video acilamadi: {video}")
    yazilan = 0
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        on = tampon if tampon is not None else int(fps * ON_TAMPON_SN)
        arka = tampon if tampon is not None else int(fps * ARKA_TAMPON_SN)
        bas = max(0, frame_start - on)
        son = frame_end + arka
        cap.set(cv2.CAP_PROP_POS_FRAMES, bas)
        writer = _yazici(out_path, fps, w, h)
        try:
            for i in range(son - bas + 1):
                ok, img = cap.read()
                if not ok:
                    break
                # Klip ham goruntu tasir - KVKK burada ozellikle onemli.
                _kvkk_uygula(img, izler, bas + i, kvkk)
                cv2.putText(img, _FILIGRAN, (8, h - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                writer.write(img)
                yazilan += 1
        finally:
            writer.release()
    finally:
        cap.release()
    if yazilan == 0:
        out_path.unlink(missing_ok=True)
        return None
    return out_path


# Sinif basina sabit renk - ayni arac tum videoda ayni renkte gorunur.
_SINIF_RENK = {
    "otomobil": (0, 200, 0), "kamyon": (0, 140, 255), "otobus": (0, 200, 200),
    "motosiklet": (255, 120, 0), "bisiklet": (255, 200, 0), "yaya": (200, 0, 200),
    "trafik_isigi": (255, 255, 255), "trafik_isareti": (180, 180, 180),
}


def annotate_tracks(video: str | Path, izler: list, out_path: str | Path,
                    kalib=None, kvkk: dict | None = None) -> Path:
    """M2 ciktisini (Track listesi) ID etiketleriyle videoya cizer."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    plan: dict[int, list] = {}
    for iz in izler:
        renk = _SINIF_RENK.get(iz.cls, _VARSAYILAN_RENK)
        etiket = f"#{iz.track_id} {iz.cls}"
        if iz.plate:
            etiket += f" {iz.plate}"
        for f, det in iz.frames.items():
            plan.setdefault(int(f), []).append((det.bbox, etiket, renk))

    # Kalibrasyon gecersizse bunu izleyicinin gormesi gerekir: bu videodaki
    # hicbir mesafe/hiz cikarimi gecerli degildir.
    kalib_notu = ""
    if kalib is not None and not getattr(kalib, "gecerli", False):
        kalib_notu = "KALIBRASYON YOK - MESAFE/HIZ HESAPLANMADI"

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise FileNotFoundError(f"video acilamadi: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = _yazici(out_path, fps, w, h)
    try:
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            _kvkk_uygula(img, izler, idx, kvkk)
            for bbox, etiket, renk in plan.get(idx, []):
                x1, y1, x2, y2 = bbox
                cv2.rectangle(img, (x1, y1), (x2, y2), renk, 2)
                cv2.putText(img, etiket, (x1, max(14, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, renk, 1, cv2.LINE_AA)
            cv2.putText(img, _FILIGRAN, (8, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
            if kalib_notu:
                cv2.putText(img, kalib_notu, (8, h - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            writer.write(img)
            idx += 1
    finally:
        cap.release()
        writer.release()
    return out_path


def read_tracks(path: str | Path) -> list:
    """izler.json'u Track listesine geri cevirir (kareler=True ile yazilmissa).

    Amaci tek sey: tespit esigini degistirdiginde YOLO'yu BASTAN calistirmamak.
    5 dakikalik videoda takip ~17 dakika surer; ayni izlerden kaza/ihlali
    yeniden hesaplamak saniyeler. Esik denemesi bu ikisi arasindaki farktir.
    """
    from adgs.schema import Detection, Track

    d = json.loads(Path(path).read_text(encoding="utf-8"))
    izler = []
    for t in d.get("izler") or []:
        kareler = t.get("kareler")
        if not kareler:
            continue  # ozet-only dosya: yeniden tespit icin kullanilamaz
        iz = Track(track_id=t["track_id"], cls=t.get("cls", ""),
                   plate=t.get("plaka"), plate_conf=t.get("plaka_conf", 0.0))
        for f, v in kareler.items():
            x1, y1, x2, y2 = v[:4]
            iz.frames[int(f)] = Detection(
                bbox=(int(x1), int(y1), int(x2), int(y2)), cls=iz.cls,
                conf=float(v[4]) if len(v) > 4 else 0.0)
        dunya = t.get("dunya") or {}
        if dunya:
            iz.world_xy = {int(f): (float(a), float(b))
                           for f, (a, b) in dunya.items()}
        izler.append(iz)
    return izler


def write_tracks(izler: list, out_path: str | Path, source: str = "",
                 kalib=None, kareler: bool = False) -> Path:
    """M2 takip ciktisini JSON'a yazar (Faz 2 teslimati).

    `kareler=True` ise kare bazli kutular da yazilir ve dosya read_tracks ile
    geri okunabilir - esikleri yeniden TAKIP ETMEDEN denemek icin. Ozet dosya
    kucuk kalsin diye varsayilan kapali.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "kaynak": source,
        "kamera": getattr(kalib, "camera_id", None),
        "kalibrasyon": {
            "gecerli": bool(getattr(kalib, "gecerli", False)),
            "hata_yuzde": getattr(kalib, "hata_yuzde", None),
            "notlar": list(getattr(kalib, "notlar", [])),
        },
        "iz_sayisi": len(izler),
        "uyari": "Bu çıktı bir karar destek analizidir; bağlayıcı bir tespit değildir.",
        "izler": [
            {
                "track_id": iz.track_id,
                "cls": iz.cls,
                "kare_sayisi": len(iz.frames),
                "ilk_kare": min(iz.frames) if iz.frames else None,
                "son_kare": max(iz.frames) if iz.frames else None,
                "plaka": iz.plate,
                "plaka_conf": iz.plate_conf,
                "dunya_koordinatli_kare": len(iz.world_xy or {}),
                # [x1, y1, x2, y2, conf] - read_tracks bunu geri kurar.
                **({"kareler": {str(f): [*d.bbox, round(d.conf, 3)]
                                for f, d in sorted(iz.frames.items())},
                    "dunya": {str(f): [round(x, 3), round(y, 3)]
                              for f, (x, y) in (iz.world_xy or {}).items()}}
                   if kareler else {}),
            }
            for iz in izler
        ],
    }
    # Kare bazli dokum buyuk olur; indent yalnizca ozet dosyada.
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=None if kareler else 2),
        encoding="utf-8")
    return out_path


def write_report(events: list[Event], out_path: str | Path, source: str = "",
                 uyarilar: list[str] | None = None) -> Path:
    """Olaylari JSON'a yazar. Kanitsiz olaylar rapora GIRMEZ.

    `uyarilar` rapora yazilir cunku "0 ihlal" ile "3 dedektor calismayi
    reddetti, 0 ihlal" ayni sey degildir. Ikincisini gizlemek raporu yaniltici
    yapar - okuyan kisi bakilmayan seyleri "yok" sanir.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    kanitli = [e for e in events if e.kanitli_mi()]
    payload = {
        "kaynak": source,
        "olay_sayisi": len(kanitli),
        "kanitsiz_atlanan": len(events) - len(kanitli),
        "calisamayan_moduller": list(uyarilar or []),
        "uyari": "Bu rapor bir karar destek çıktısıdır; bağlayıcı bir tespit değildir.",
        "events": [asdict(e) for e in kanitli],
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path
