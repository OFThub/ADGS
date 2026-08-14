"""M10 - Olay/klip/metadata kaliciligi (SQLite).

Dort tablo: videos - events - parties - violations. Sema PostgreSQL'e
tasinabilir yazilir; gecis baglanti dizesi degisikligi olsun diye SQLite'a ozgu
tip ve sozdizimi kullanilmaz (TEXT/INTEGER/REAL ve duz FOREIGN KEY yeter).

Coklu kamera hazirligi: videos.camera_id ve events.gps_* alanlari BASTAN vardir.
Ayri bir cameras tablosu, kullanici yonetimi ve tenant izolasyonu SIMDI
YAZILMAZ - iki kolon ileriye donuk yeter; ORM ve migration altyapisi gercek
ikinci kamera gelince eklenir.

KVKK: saklama_uygula() suresi dolan videolari VE kliplerini siler. Silme
kayitla sinirli kalirsa diskte kisisel veri (plaka, yuz) imha edilmemis olur -
bu yuzden dosya silme de buradadir ve raporlanir.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from adgs.schema import Event

VARSAYILAN_DB = Path("data/adgs.db")

_SEMA = """
CREATE TABLE IF NOT EXISTS videos (
    id                INTEGER PRIMARY KEY,
    dosya             TEXT    NOT NULL,
    camera_id         TEXT,
    kaynak_profil     TEXT,
    cekim_tarihi      TEXT,               -- ISO YYYY-MM-DD (ceza tablosu secimi)
    islenme_zamani    TEXT    NOT NULL,   -- ISO 8601 UTC (saklama suresi)
    durum             TEXT    NOT NULL,   -- BEKLIYOR|ISLENIYOR|TAMAM|HATA
    hata              TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id            INTEGER PRIMARY KEY,
    video_id      INTEGER NOT NULL REFERENCES videos(id) ON DELETE CASCADE,
    event_id      TEXT    NOT NULL,
    tip           TEXT    NOT NULL,
    alt_tip       TEXT,
    t_start       REAL,
    t_end         REAL,
    frame_start   INTEGER,
    frame_end     INTEGER,
    conf          REAL,
    gps_lat       REAL,
    gps_lon       REAL,
    klip          TEXT,
    notlar        TEXT,                   -- JSON dizi
    UNIQUE (video_id, event_id)
);
CREATE TABLE IF NOT EXISTS parties (
    id            INTEGER PRIMARY KEY,
    event_row_id  INTEGER NOT NULL REFERENCES events(id) ON DELETE CASCADE,
    track_id      INTEGER,
    plaka         TEXT,
    hasar         TEXT                    -- JSON (M8)
);
CREATE TABLE IF NOT EXISTS violations (
    id                  INTEGER PRIMARY KEY,
    party_row_id        INTEGER NOT NULL REFERENCES parties(id) ON DELETE CASCADE,
    ihlal_kodu          TEXT    NOT NULL,
    ktk_madde           TEXT,
    kusur_sinifi        TEXT,
    ceza_tutari_try     INTEGER,
    ceza_puani          INTEGER,
    ceza_tablosu_tarihi TEXT,             -- ISO YYYY-MM-DD
    conf                REAL
);
CREATE INDEX IF NOT EXISTS ix_events_tip   ON events (tip);
CREATE INDEX IF NOT EXISTS ix_events_video ON events (video_id);
"""


def baglan(db: str | Path | None = None) -> sqlite3.Connection:
    """Baglanti acar ve semayi kurar.

    Varsayilan yol CAGRI ANINDA okunur (imzaya gomulmez): imzaya gomulen bir
    varsayilan import aninda sabitlenir ve VARSAYILAN_DB'yi sonradan degistirmek
    hicbir ise yaramaz - hem test hem dagitim yapilandirmasi kirilirdi.

    foreign_keys PRAGMA'si SQLite'ta VARSAYILAN OLARAK KAPALIDIR; acilmazsa
    ON DELETE CASCADE sessizce calismaz ve saklama suresi dolan videonun
    olaylari veritabaninda oksuz kalir.
    """
    p = Path(db if db is not None else VARSAYILAN_DB)
    if str(p.parent) not in ("", "."):
        p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SEMA)
    conn.commit()
    return conn


def _simdi() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def video_kaydet(conn: sqlite3.Connection, dosya: str, camera_id: str | None = None,
                 kaynak_profil: str | None = None, cekim_tarihi: str | None = None,
                 durum: str = "BEKLIYOR") -> int:
    cur = conn.execute(
        "INSERT INTO videos (dosya, camera_id, kaynak_profil, cekim_tarihi, "
        "islenme_zamani, durum) VALUES (?, ?, ?, ?, ?, ?)",
        (str(dosya), camera_id, kaynak_profil, cekim_tarihi, _simdi(), durum),
    )
    conn.commit()
    return int(cur.lastrowid)


def durum_guncelle(conn: sqlite3.Connection, video_id: int, durum: str,
                   hata: str | None = None) -> None:
    conn.execute("UPDATE videos SET durum = ?, hata = ? WHERE id = ?",
                 (durum, hata, video_id))
    conn.commit()


YARIDA_KALDI = ("sunucu yeniden baslatildi, analiz yarida kaldi - "
                "videoyu tekrar yukleyin")


def yarida_kalanlari_isaretle(conn: sqlite3.Connection) -> int:
    """BEKLIYOR/ISLENIYOR kalmis kayitlari HATA olarak isaretler.

    Analiz sunucu sureci icinde calisir; surec olunce is kaybolur ama kayit
    "isleniyor" olarak kalir ve kimse onu bir daha ele almaz. O hali
    veritabaninda YALAN soyler: kullanici bekler, video hic gelmez.

    Doner: isaretlenen kayit sayisi.
    """
    cur = conn.execute(
        "UPDATE videos SET durum = 'HATA', hata = ? "
        "WHERE durum IN ('BEKLIYOR', 'ISLENIYOR')", (YARIDA_KALDI,))
    conn.commit()
    return cur.rowcount


def olaylari_kaydet(conn: sqlite3.Connection, video_id: int,
                    events: list[Event]) -> int:
    """Olaylari yazar. KANITSIZ OLAY YAZILMAZ - rapor kuraliyla ayni."""
    n = 0
    for evt in events:
        if not evt.kanitli_mi():
            continue
        gps = evt.gps or (None, None)
        cur = conn.execute(
            "INSERT OR REPLACE INTO events (video_id, event_id, tip, alt_tip, "
            "t_start, t_end, frame_start, frame_end, conf, gps_lat, gps_lon, "
            "klip, notlar) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (video_id, evt.event_id, evt.tip, evt.alt_tip, evt.t_start, evt.t_end,
             evt.frame_start, evt.frame_end, evt.conf, gps[0], gps[1],
             evt.evidence.get("clip"), json.dumps(evt.notes, ensure_ascii=False)),
        )
        eid = int(cur.lastrowid)
        for taraf in evt.parties:
            pc = conn.execute(
                "INSERT INTO parties (event_row_id, track_id, plaka, hasar) "
                "VALUES (?,?,?,?)",
                (eid, taraf.track_id, taraf.plate,
                 json.dumps(taraf.hasar, ensure_ascii=False) if taraf.hasar else None),
            )
            pid = int(pc.lastrowid)
            for ihlal in taraf.violations:
                d = asdict(ihlal)
                conn.execute(
                    "INSERT INTO violations (party_row_id, ihlal_kodu, ktk_madde, "
                    "kusur_sinifi, ceza_tutari_try, ceza_puani, ceza_tablosu_tarihi, "
                    "conf) VALUES (?,?,?,?,?,?,?,?)",
                    (pid, d["ihlal_kodu"], d["ktk_madde"], d["kusur_sinifi"],
                     d["ceza_tutari_try"], d["ceza_puani"], d["ceza_tablosu_tarihi"],
                     d["conf"]),
                )
        n += 1
    conn.commit()
    return n


def olaylari_coz(rapor: dict | str | Path) -> list[Event]:
    """rapor.json (veya ayni yapidaki sozluk) icerigini Event listesine cevirir.

    JSON, write_report'ta asdict(Event) ile uretildigi icin yapi Event ile
    birebir aynidir. Ayri fonksiyon olmasinin sebebi: hem veritabanina yazma
    (rapordan_kaydet) hem de videoyu YENIDEN CIZME (cli.rerender) ayni cozmeyi
    kullanir - iki kopya, iki farkli davranan cozucu demekti.
    """
    from adgs.schema import Party, Violation

    if isinstance(rapor, (str, Path)):
        rapor = json.loads(Path(rapor).read_text(encoding="utf-8"))

    events: list[Event] = []
    for d in rapor.get("events") or []:
        taraflar = [
            Party(
                track_id=p.get("track_id"),
                plate=p.get("plate"),
                hasar=p.get("hasar"),
                violations=[Violation(**v) for v in (p.get("violations") or [])],
            )
            for p in (d.get("parties") or [])
        ]
        gps = d.get("gps")
        events.append(Event(
            event_id=d["event_id"], tip=d["tip"], alt_tip=d.get("alt_tip", ""),
            t_start=d.get("t_start", 0.0), t_end=d.get("t_end", 0.0),
            frame_start=d.get("frame_start", 0), frame_end=d.get("frame_end", 0),
            source_video=d.get("source_video", ""),
            source_profile=d.get("source_profile", "cctv_fixed"),
            conf=d.get("conf", 0.0), evidence=d.get("evidence") or {},
            gps=tuple(gps) if gps else None, parties=taraflar,
            notes=list(d.get("notes") or []),
        ))
    return events


def rapordan_kaydet(conn: sqlite3.Connection, video_id: int,
                    rapor: dict | str | Path) -> int:
    """rapor.json icerigini veritabanina yazar."""
    return olaylari_kaydet(conn, video_id, olaylari_coz(rapor))


def _olay_sozluk(conn: sqlite3.Connection, satir: sqlite3.Row) -> dict:
    taraflar = []
    for p in conn.execute("SELECT * FROM parties WHERE event_row_id = ?",
                          (satir["id"],)):
        ihlaller = [dict(v) for v in conn.execute(
            "SELECT ihlal_kodu, ktk_madde, kusur_sinifi, ceza_tutari_try, "
            "ceza_puani, ceza_tablosu_tarihi, conf FROM violations "
            "WHERE party_row_id = ?", (p["id"],))]
        taraflar.append({
            "track_id": p["track_id"], "plaka": p["plaka"],
            "hasar": json.loads(p["hasar"]) if p["hasar"] else None,
            "violations": ihlaller,
        })
    gps = None
    if satir["gps_lat"] is not None and satir["gps_lon"] is not None:
        gps = [satir["gps_lat"], satir["gps_lon"]]
    return {
        "event_id": satir["event_id"], "tip": satir["tip"], "alt_tip": satir["alt_tip"],
        "t_start": satir["t_start"], "t_end": satir["t_end"],
        "frame_start": satir["frame_start"], "frame_end": satir["frame_end"],
        "conf": satir["conf"], "gps": gps, "klip": satir["klip"],
        "video": satir["dosya"], "video_id": satir["video_id"],
        "camera_id": satir["camera_id"], "cekim_tarihi": satir["cekim_tarihi"],
        "notlar": json.loads(satir["notlar"]) if satir["notlar"] else [],
        "parties": taraflar,
    }


_SECIM = (
    "SELECT e.*, v.dosya, v.camera_id, v.cekim_tarihi FROM events e "
    "JOIN videos v ON v.id = e.video_id"
)


def olaylari_getir(conn: sqlite3.Connection, tip: str | None = None,
                   tarih: str | None = None, camera_id: str | None = None,
                   video_id: int | None = None, limit: int = 200) -> list[dict]:
    """Filtreli olay listesi. tarih = videonun cekim tarihi (ISO)."""
    kosullar, degerler = [], []
    for alan, deger in (("e.tip", tip), ("v.cekim_tarihi", tarih),
                        ("v.camera_id", camera_id), ("e.video_id", video_id)):
        if deger is not None:
            kosullar.append(f"{alan} = ?")
            degerler.append(deger)
    sql = _SECIM + (" WHERE " + " AND ".join(kosullar) if kosullar else "")
    sql += " ORDER BY e.id DESC LIMIT ?"
    degerler.append(int(limit))
    return [_olay_sozluk(conn, r) for r in conn.execute(sql, degerler)]


def olay_getir(conn: sqlite3.Connection, event_id: str) -> dict | None:
    r = conn.execute(_SECIM + " WHERE e.event_id = ? ORDER BY e.id DESC LIMIT 1",
                     (event_id,)).fetchone()
    return _olay_sozluk(conn, r) if r else None


def video_getir(conn: sqlite3.Connection, video_id: int) -> dict | None:
    r = conn.execute("SELECT * FROM videos WHERE id = ?", (video_id,)).fetchone()
    if not r:
        return None
    d = dict(r)
    d["olay_sayisi"] = conn.execute(
        "SELECT COUNT(*) FROM events WHERE video_id = ?", (video_id,)).fetchone()[0]
    return d


def videolari_getir(conn: sqlite3.Connection, limit: int = 100) -> list[dict]:
    return [dict(r) for r in conn.execute(
        "SELECT * FROM videos ORDER BY id DESC LIMIT ?", (int(limit),))]


def saklama_uygula(conn: sqlite3.Connection, gun: int,
                   klip_sil: bool = True) -> dict:
    """KVKK: saklama suresi dolan kayitlari VE kliplerini siler.

    Yalnizca veritabani satirlarini silmek yetmez - klip dosyalari diskte
    kalirsa kisisel veri imha edilmemis olur.
    """
    if gun <= 0:
        raise ValueError("saklama suresi pozitif gun olmali")
    sinir = (datetime.now(timezone.utc) - timedelta(days=gun)).isoformat(timespec="seconds")
    video_ids = [int(r["id"]) for r in conn.execute(
        "SELECT id FROM videos WHERE islenme_zamani < ?", (sinir,))]
    silinen_klip, silinemeyen = 0, []
    if video_ids and klip_sil:
        soru = ",".join("?" * len(video_ids))
        for r in conn.execute(
            f"SELECT klip FROM events WHERE video_id IN ({soru}) AND klip IS NOT NULL",
            video_ids,
        ):
            p = Path(r["klip"])
            try:
                if p.exists():
                    p.unlink()
                    silinen_klip += 1
            except OSError as e:
                silinemeyen.append(f"{p}: {e}")
    for vid in video_ids:
        conn.execute("DELETE FROM videos WHERE id = ?", (vid,))
    conn.commit()
    return {
        "sinir": sinir, "silinen_video": len(video_ids),
        "silinen_klip": silinen_klip, "silinemeyen": silinemeyen,
    }
