"""Faz 7 - FastAPI servisi ve tek sayfalik arayuz.

    uvicorn adgs.api:app --reload      (veya: adgs serve)

Isleme SENKRON DEGILDIR: yukleme kaydi hemen olusturulur (durum=BEKLIYOR),
analiz arka planda calisir ve durum ISLENIYOR -> TAMAM/HATA olarak ilerler.
Bir videonun analizi dakikalar surdugu icin HTTP istegini bekletmek yanlis
olurdu; istemci /videos/{id}/status ile takip eder.

Boru hatti burada YENIDEN YAZILMAZ: cli.run cagrilir, urettigi rapor.json
veritabanina alinir. Ikinci bir islem yolu, iki farkli davranan iki sistem
demekti.

KVKK: /kvkk/purge saklama suresi dolan kayitlari ve klip dosyalarini imha eder.
Servis acildiginda otomatik SILME YAPILMAZ - imha bilincli bir eylemdir ve
zamanlanmis gorevle veya elle tetiklenir.
"""

from __future__ import annotations

import re
import shutil
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, HTMLResponse

from adgs import store

YUKLEME_DIZINI = Path("data/uploads")
CIKTI_DIZINI = Path("runs/api")
_WEB = Path(__file__).resolve().parent / "web"

# camera/tarih artik istemciden ALINMIYOR - adgs.probe turetiyor ve bicimi
# orada suzuyor (probe._KAMERA_DESENI, date.isoformat). Bicim dogrulamasi
# uretildigi yerde durur; burada tekrarlanmasi iki ayri gercek kaynagi olurdu.

@asynccontextmanager
async def _yasam(app: FastAPI):
    """Acilista yarida kalmis kayitlari isaretler.

    Analiz BackgroundTasks ile SUREC ICINDE calisir; sunucu yeniden
    baslatildiginda o gorevler kaybolur ama veritabanindaki kayit BEKLIYOR /
    ISLENIYOR olarak kalir. Kimse onlari bir daha ele almaz - kullanici
    "hala isleniyor" yazisina bakip bekler, video 404 dondurmeye devam eder.

    Otomatik yeniden kuyruga alinMIYOR: kullanicinin baslatmadigi dakikalarca
    surecek bir isi acilista kendiliginden baslatmak surpriz olurdu. Kayit
    HATA olarak isaretlenir, sebebi yazilir, yeniden yukleme kullaniciya kalir.
    """
    try:
        conn = store.baglan()
        try:
            n = store.yarida_kalanlari_isaretle(conn)
        finally:
            conn.close()
        if n:
            print(f"[ADGS] yarida kalmis {n} analiz HATA olarak isaretlendi "
                  "(sunucu yeniden baslatilmis)")
    except Exception as e:  # noqa: BLE001 - temizlik servisi engellememeli
        print(f"[ADGS] yarida kalan kayit taramasi yapilamadi: {e}")
    yield


app = FastAPI(
    title="ADGS - Trafik Video Analiz Sistemi",
    description="Karar destek araci. Baglayici bir tespit uretmez.",
    version="0.1.0",
    lifespan=_yasam,
)


def _db():
    return store.baglan()


def _guvenli_ad(ad: str | None) -> str:
    """Yukleme dosya adini diske yazilabilir hale getirir.

    Dosya adi artik TEK kullanici girdisi: hem diske yazilan yol, hem de
    turetimin (kamera, tarih) okudugu kaynak. Path().name yol gezinmesini
    keser ama Windows'ta yasak karakterleri (< > : " | ? *) birakir - onlarla
    open() OSError verir ve yukleme 500 doner.

    Turetimin ihtiyaci olan karakterler (harf, rakam, . _ -) korunur; gerisi
    alt cizgiye doner. Boylece "20260715_kavsak.mp4" bozulmadan gecer.
    """
    taban = Path(ad or "video.mp4").name
    temiz = re.sub(r"[^A-Za-z0-9._-]", "_", taban)[:120].lstrip(".")
    return temiz or "video.mp4"


def _isle(video_id: int, dosya: Path, profil: str, detect: str,
          camera: str | None, tarih: str | None) -> None:
    """Arka plan isi: analizi calistirir ve sonucu veritabanina yazar."""
    from adgs.cli import run as cli_run

    yol_modeli = "runs/rdd2022/yolo26s/weights/best.pt"
    # Dedektorler artik OTOMATIK seciliyor; profil kararsiz kaldiginda listeye
    # roaddamage de giriyor. cli.run yol hasari modeli yoksa TUM kosuyu
    # reddediyor - eksik bir ek dedektor yuzunden kaza/ihlal analizini de
    # kaybetmemek icin burada listeden dusuruluyor.
    if not Path(yol_modeli).exists():
        kalan = [t for t in detect.split(",") if t.strip() != "roaddamage"]
        detect = ",".join(kalan) or "accident"

    conn = _db()
    try:
        store.durum_guncelle(conn, video_id, "ISLENIYOR")
        outdir = CIKTI_DIZINI / str(video_id)
        kod = cli_run(
            str(dosya), model=yol_modeli,
            out=str(outdir), profile=profil, conf=0.35,
            detect_tipleri=detect, camera=camera, tarih_metni=tarih,
        )
        rapor = outdir / "rapor.json"
        if kod != 0 or not rapor.exists():
            store.durum_guncelle(conn, video_id, "HATA",
                                 f"analiz tamamlanamadi (cikis kodu {kod})")
            return
        n = store.rapordan_kaydet(conn, video_id, rapor)
        store.durum_guncelle(conn, video_id, "TAMAM", None if n else "olay bulunamadi")
    except Exception as e:  # noqa: BLE001 - hata istemciye durum olarak donmeli
        store.durum_guncelle(conn, video_id, "HATA", f"{type(e).__name__}: {e}")
    finally:
        conn.close()


@app.get("/", response_class=HTMLResponse)
def arayuz() -> str:
    return (_WEB / "index.html").read_text(encoding="utf-8")


@app.post("/videos")
async def video_yukle(arkaplan: BackgroundTasks, file: UploadFile = File(...)):
    """Video yukler ve analizi arka planda baslatir.

    TEK GIRDI VIDEODUR. Kaynak profili, calisacak dedektorler, kamera ve
    cekim tarihi kullanicidan SORULMAZ; adgs.probe bunlari videonun kendisinden
    turetir (kamera kaymasi olcumu + dosya adi). Kullanicinin elle girdigi
    "sabit kamera" veya "15.07.2026" gibi degerler dogrulanamayan beyanlardi;
    olculen deger yanlis olabilir ama nedeni `gerekce` ile birlikte raporlanir.

    Turetilemeyen alan BOS kalir ve ilgili modul kendini kapatir - tahmin
    edilmez (bkz. probe modul basligi).
    """
    from adgs import probe

    YUKLEME_DIZINI.mkdir(parents=True, exist_ok=True)
    hedef = YUKLEME_DIZINI / _guvenli_ad(file.filename)
    with hedef.open("wb") as f:
        shutil.copyfileobj(file.file, f)

    t = probe.incele(hedef)
    conn = _db()
    try:
        vid = store.video_kaydet(conn, str(hedef), camera_id=t["camera"],
                                 kaynak_profil=t["profile"],
                                 cekim_tarihi=t["tarih"])
    finally:
        conn.close()
    arkaplan.add_task(_isle, vid, hedef, t["profile"], t["detect"],
                      t["camera"], t["tarih"])
    return {"video_id": vid, "durum": "BEKLIYOR", "turetilen": t,
            "uyari": "Analiz arka planda calisiyor; /videos/{id}/status ile takip edin."}


@app.get("/videos")
def videolar(limit: int = Query(50, ge=1, le=500)):
    conn = _db()
    try:
        return {"videolar": store.videolari_getir(conn, limit=limit)}
    finally:
        conn.close()


@app.get("/videos/{video_id}/status")
def video_durumu(video_id: int):
    conn = _db()
    try:
        d = store.video_getir(conn, video_id)
    finally:
        conn.close()
    if d is None:
        raise HTTPException(404, "video bulunamadi")
    return d


@app.get("/events")
def olaylar(tip: str | None = None, tarih: str | None = None,
            camera_id: str | None = None, video_id: int | None = None,
            limit: int = Query(200, ge=1, le=1000)):
    conn = _db()
    try:
        kayitlar = store.olaylari_getir(conn, tip=tip, tarih=tarih,
                                        camera_id=camera_id, video_id=video_id,
                                        limit=limit)
    finally:
        conn.close()
    return {
        "sayi": len(kayitlar),
        "uyari": "Bu cikti bir karar destek analizidir; baglayici bir tespit degildir.",
        "events": kayitlar,
    }


@app.get("/events/{event_id}")
def olay(event_id: str):
    conn = _db()
    try:
        d = store.olay_getir(conn, event_id)
    finally:
        conn.close()
    if d is None:
        raise HTTPException(404, "olay bulunamadi")
    return d


def _isaretli_video(video_id: int) -> Path | None:
    """Isaretlenmis MP4'u bulur.

    Yol veritabaninda tutulmuyor, cikti dizininden turetiliyor: ayri bir kolon
    ve migration eklemek mevcut veritabanini riske atmaya degmezdi - dosya tek
    bir yerde (cli.run) ve sabit ad kalibiyla uretiliyor.
    """
    d = CIKTI_DIZINI / str(video_id)
    return next(iter(sorted(d.glob("*_annotated.mp4"))), None) if d.exists() else None


@app.get("/videos/{video_id}/annotated")
def isaretli_video(video_id: int):
    """Isaretlenmis video - arayuzdeki oynatici bunu gosterir.

    Dosya yoksa 404 SEBEBIYLE birlikte doner: "henuz islenmedi" ile "islendi
    ama cikti uretilemedi" ayni sey degil. Cipla 404 sunucu kaydinda ayirt
    edilemiyordu ve calisan analiz ariza gibi gorunuyordu.
    """
    p = _isaretli_video(video_id)
    if p is not None:
        return FileResponse(p, media_type="video/mp4")
    conn = _db()
    try:
        d = store.video_getir(conn, video_id)
    finally:
        conn.close()
    if d is None:
        raise HTTPException(404, "video kaydi yok")
    durum = d.get("durum")
    if durum in ("BEKLIYOR", "ISLENIYOR"):
        raise HTTPException(404, f"analiz suruyor (durum: {durum}) - "
                                 "isaretli video analiz bitince olusur")
    raise HTTPException(404, f"isaretlenmis video uretilmedi (durum: {durum}"
                             + (f", hata: {d['hata']}" if d.get("hata") else "") + ")")


@app.get("/events/{event_id}/klip")
def olay_klibi(event_id: str):
    """Olayin kanit klibi."""
    conn = _db()
    try:
        d = store.olay_getir(conn, event_id)
    finally:
        conn.close()
    if d is None:
        raise HTTPException(404, "olay bulunamadi")
    klip = d.get("klip")
    if not klip or not Path(klip).exists():
        raise HTTPException(404, "klip bulunamadi")
    return FileResponse(klip, media_type="video/mp4")


@app.get("/events/{event_id}/is-emri.pdf")
def is_emri(event_id: str):
    """Fen Isleri icin PDF is emri."""
    from adgs.workorder import is_emri_pdf

    conn = _db()
    try:
        d = store.olay_getir(conn, event_id)
    finally:
        conn.close()
    if d is None:
        raise HTTPException(404, "olay bulunamadi")
    cikti = CIKTI_DIZINI / "is_emri" / f"{event_id}.pdf"
    is_emri_pdf(d, cikti)
    return FileResponse(cikti, media_type="application/pdf",
                        filename=f"{event_id}.pdf")


@app.get("/kara-nokta")
def kara_nokta(yaricap_m: float = Query(50.0, gt=0, le=2000),
               tip: str | None = None):
    """Faz 8 - kara nokta analizi (planlama ciktisi, kusur/ceza dogurmaz)."""
    from adgs import hotspot

    conn = _db()
    try:
        olaylar = store.olaylari_getir(conn, tip=tip, limit=1000)
    finally:
        conn.close()
    return hotspot.analiz(olaylar, yaricap_m=yaricap_m)


@app.get("/kameralar")
def kameralar():
    """Faz 8 - coklu kamera ozeti: kamera basina video ve olay sayisi."""
    conn = _db()
    try:
        videolar = store.videolari_getir(conn, limit=500)
        olaylar = store.olaylari_getir(conn, limit=1000)
    finally:
        conn.close()
    ozet: dict[str, dict] = {}
    for v in videolar:
        ad = v.get("camera_id") or "(kamera tanimsiz)"
        k = ozet.setdefault(ad, {"camera_id": ad, "video": 0, "olay": 0,
                                 "tipler": {}})
        k["video"] += 1
    for o in olaylar:
        ad = o.get("camera_id") or "(kamera tanimsiz)"
        k = ozet.setdefault(ad, {"camera_id": ad, "video": 0, "olay": 0,
                                 "tipler": {}})
        k["olay"] += 1
        k["tipler"][o["tip"]] = k["tipler"].get(o["tip"], 0) + 1
    return {"kameralar": sorted(ozet.values(), key=lambda k: -k["olay"])}


@app.post("/kvkk/purge")
def kvkk_imha(gun: int | None = None):
    """Saklama suresi dolan kayitlari ve klipleri imha eder.

    `gun` verilmezse config/pipeline.yaml icindeki saklama_gun kullanilir.
    """
    if gun is None:
        import yaml

        cfg = yaml.safe_load(
            (Path(__file__).resolve().parent.parent / "config" / "pipeline.yaml")
            .read_text(encoding="utf-8")
        ) or {}
        gun = int(cfg.get("saklama_gun", 30))
    conn = _db()
    try:
        return store.saklama_uygula(conn, gun=gun)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    finally:
        conn.close()
