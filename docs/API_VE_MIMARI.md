# ADGS — API ve Mimari

Bu belge sistemin **iç yapısını** ve **HTTP arayüzünü** tanımlar. Operasyonel kullanım
için [KULLANICI_KILAVUZU.md](KULLANICI_KILAVUZU.md), iş akışları için
[KULLANIM_SENARYOLARI.md](KULLANIM_SENARYOLARI.md).

---

## 1. Mimari ilkeler

Sistem dört karardan doğar. Bunlar teknik tercih değil, projenin bağlamının dayattığı
kısıtlardır.

### İlke 1 — Tek merkezi veri sözleşmesi, doğrudan modül çağrısı yok

Modüller birbirini **doğrudan çağırmaz**; hepsi `adgs/schema.py` içindeki `Event`
sözleşmesini üretir ve tüketir. Sonucu: bir modül henüz yazılmamışken de boru hattı
çalışır, ilgili alanlar `None` kalır.

```
M4 (ihlal)  →  Violation(ihlal_kodu="KIRMIZI_ISIK", ktk_madde=None, ...)
M6 (kusur)  →  aynı nesnenin ktk_madde ve kusur_sinifi alanlarını doldurur
M7 (ceza)   →  aynı nesnenin ceza_* alanlarını doldurur
```

M4 yalnızca *ne olduğunu* söyler; hukuki yorum M6'nın, tutar M7'nin işidir.

### İlke 2 — Ön koşul yoksa modül tahmin üretmez, reddeder

Sistemin en sık tekrarlanan davranışı budur:

| Eksik | Reddeden modül |
|---|---|
| Doğrulanmış kalibrasyon | `speed`, `tailgating`, dünya koordinatı, hız |
| `dur_cizgisi` / `isik_roi` | `redlight` |
| `seritler[].yon` | `wrongway` |
| Ceza tablosu tarihi | M7 (ceza hesaplanmaz) |
| Çözünürlük eşiği | M8 (`guvenilir: false`) |
| Kanıt (klip/keyframe) | Olay rapora **yazılmaz** |

Sessizce boş dönmek en tehlikeli yanlıştır: okuyan kişi "ihlal yok" sanar, oysa gerçek
anlamı "bakılmadı"dır. Bu yüzden reddin sebebi hem konsola hem `rapor.json` içindeki
`calisamayan_moduller` alanına yazılır.

### İlke 3 — Hukuki kısıt mimariye gömülür, sonradan eklenmez

| Kısıt | Mimarideki karşılığı |
|---|---|
| Belediye ceza kesemez (KTK) | Çıktılar "karar destek"; hiçbir uç idari işlem üretmez |
| KTY m.156/3 — tutanak oran belirtmez | M6 **yüzde üretmez**; `ASLI`/`TALI`/`TESPIT_EDILEMEDI` döner |
| Ceza tutarları tarihe bağlıdır | `penalty.py` içinde tek bir tutar yok; tarih damgalı YAML tabloları |
| KVKK — kişisel veri | Bulanıklaştırma varsayılan açık, saklama süresi + cascade imha |
| Ön değerlendirme uyarısı | `Event.__post_init__` KAZA/IHLAL olaylarına notu **otomatik** ekler |

Son satır önemlidir: not kod tarafından kaldırılamaz, çünkü dataclass'ın kurucusunda
eklenir.

### İlke 4 — Hukuken hassas kısım GPU'suz ve saf tutulur

M6 (kusur) ve M7 (ceza) yalnızca `Event` + YAML okur. Görüntü işlemeden tamamen
kopukturlar. Bu sayede sistemin en hassas parçası aynı zamanda birim testle **tam
kapsanan** parçasıdır.

---

## 2. Modül haritası

```
                    ┌──────────────────────────────────────────┐
  video ──────────► │ M1 ingest — kare örnekleme, ön işleme    │
                    └───────────────┬──────────────────────────┘
                                    │ kareler
              ┌─────────────────────┼─────────────────────┐
              │                     │                     │
   ┌──────────▼─────────┐  ┌────────▼─────────┐           │
   │ M2 detect + track  │  │ M5 roaddamage    │           │
   │ (YOLO26 + BoTSORT) │  │ (RDD2022)        │           │
   │ + M2.1 plate (OCR) │  └────────┬─────────┘           │
   └──────────┬─────────┘           │                     │
     Track[]  │                     │ Event(ALTYAPI)      │
     ┌────────┼────────┐            │                     │
     │        │        │            │                     │
┌────▼───┐ ┌──▼─────┐ ┌▼────────┐   │            ┌────────▼─────────┐
│M3      │ │M4      │ │M8       │   │            │ calib            │
│accident│ │violat. │ │vehicle  │   │            │ (homografi)      │
│        │ │(6 det.)│ │damage   │   │            │ hız/mesafe kapısı│
└────┬───┘ └──┬─────┘ └┬────────┘   │            └──────────────────┘
     │        │        │            │
     │  Event(KAZA/IHLAL) + hasar   │
     └────────┴────┬───┴────────────┘
                   │
          ┌────────▼─────────┐
          │ M6 fault (KTK 84)│  ← saf, GPU'suz
          └────────┬─────────┘
          ┌────────▼─────────┐
          │ M7 penalty       │  ← saf, GPU'suz, tarih damgalı tablo
          └────────┬─────────┘
                   │
      ┌────────────┼─────────────┐
      │            │             │
┌─────▼─────┐ ┌────▼──────┐ ┌────▼──────────┐
│M9 render  │ │M10 store  │ │workorder (PDF)│
│işaretli   │ │SQLite     │ │Fen İşleri     │
│video+JSON │ │4 tablo    │ │               │
└───────────┘ └────┬──────┘ └───────────────┘
                   │
        ┌──────────┼──────────┐
   ┌────▼────┐ ┌───▼────┐ ┌───▼──────┐
   │api      │ │hotspot │ │acceptance│
   │FastAPI  │ │kara nk.│ │kabul     │
   └─────────┘ └────────┘ └──────────┘
```

| Modül | Dosya | Satır | İş |
|---|---|--:|---|
| M1 | `ingest.py` | 84 | Video alım, kare örnekleme, ön işleme |
| M2 | `detect.py` | 120 | Tespit + takip — **sistemin tek gerçek kaynağı** |
| M2.1 | `plate.py` | 102 | TR plaka okuma ve **doğrulama** (asıl işi OCR'ı reddetmek) |
| M3 | `accident.py` | 342 | Kaza tespiti (kural tabanlı, 3 sinyal + oklüzyon filtresi) |
| M4 | `violations/` | — | 6 bağımsız ihlal dedektörü, ortak sözleşme |
| M5 | `roaddamage.py` | 130 | Yol/altyapı hasarı (RDD2022) |
| M6 | `fault.py` | 130 | Kusur — KTK m.84 kapalı listesi |
| M7 | `penalty.py` | 144 | Ceza — tarih damgalı tablo |
| M8 | `vehicledamage.py` | 290 | Araç hasarı (CarDD) + çözünürlük kapısı |
| M9 | `render.py` | 545 | İşaretli video + JSON rapor + KVKK bulanıklaştırma |
| M10 | `store.py` | 332 | SQLite kalıcılık, 4 tablo |
| — | `calib.py` | 136 | Homografi; hıza bağlı her çıktının kapısı |
| — | `probe.py` | 317 | Videodan parametre türetme |
| — | `api.py` | 351 | FastAPI + tek sayfa arayüz |
| — | `hotspot.py` | 145 | Kara nokta analizi |
| — | `workorder.py` | 169 | Fen İşleri PDF iş emri |
| — | `acceptance.py` | 359 | Kabul kriteri ölçümü |
| — | `trainguard.py` | 80 | Eğitim kayıp muhafızı |
| — | `cli.py` | 868 | Komut satırı arayüzü |

Toplam ~4.760 satır uygulama kodu, 365 test.

---

## 3. Veri sözleşmesi (`adgs/schema.py`)

### 3.1 `Event` — sistemin tek merkezi sözleşmesi

```python
@dataclass
class Event:
    event_id: str                    # "kaza_0001", "ihlal_0042", "altyapi_0007"
    tip: EventTip                    # "KAZA" | "IHLAL" | "ALTYAPI"
    alt_tip: str
    t_start: float                   # saniye
    t_end: float
    frame_start: int
    frame_end: int
    source_video: str
    source_profile: SourceProfile    # "cctv_fixed" | "vehicle_mounted"
    conf: float
    evidence: dict = {}              # {"clip": str, "keyframes": [int]}
    gps: tuple[float, float] | None = None
    parties: list[Party] = []
    notes: list[str] = []

    def __post_init__(self):
        # KAZA/IHLAL olaylarına ön değerlendirme notu OTOMATİK eklenir.
        # ALTYAPI taşımaz: belediyenin kendi görev alanı, kusur/ceza doğurmaz.

    def kanitli_mi(self) -> bool:
        # Kanıtsız Event rapora YAZILMAZ.
```

### 3.2 `Party` ve `Violation`

```python
@dataclass
class Party:
    track_id: int
    plate: str | None = None
    violations: list[Violation] = []
    hasar: dict | None = None            # M8

@dataclass
class Violation:
    ihlal_kodu: str                      # M4 üretir
    conf: float
    ktk_madde: str | None = None         # M6 doldurur
    kusur_sinifi: KusurSinifi | None = None   # M6: ASLI|TALI|TESPIT_EDILEMEDI
    ceza_tutari_try: int | None = None   # M7
    ceza_puani: int | None = None        # M7
    ceza_tablosu_tarihi: str | None = None    # M7 — hangi tablodan hesaplandı
```

`ceza_tablosu_tarihi` alanı denetlenebilirlik içindir: bir tutarın hangi tablodan geldiği
çıktıdan okunabilmelidir.

### 3.3 `Track` — kalibrasyon kapısı sözleşmede

```python
@dataclass
class Track:
    track_id: int
    cls: str
    frames: dict[int, Detection] = {}
    plate: str | None = None
    plate_conf: float = 0.0
    world_xy: dict[int, tuple[float, float]] | None = None
```

`world_xy is None` **olması**, hız/mesafe modüllerinin çalışmaması gerektiği anlamına
gelir. Kapı sözleşmenin kendisine gömülüdür; her modülün ayrıca kontrol etmesi gerekmez.

---

## 4. Veritabanı şeması (M10)

SQLite, dört tablo. PostgreSQL'e taşınabilir yazıldı; geçiş bağlantı dizesi
değişikliğidir.

```sql
videos (id, dosya, camera_id, kaynak_profil, cekim_tarihi,
        islenme_zamani, durum, hata)
        -- durum: BEKLIYOR | ISLENIYOR | TAMAM | HATA
        -- cekim_tarihi: ISO YYYY-MM-DD  (ceza tablosu seçimi)
        -- islenme_zamani: ISO 8601 UTC  (saklama süresi)

events (id, video_id → videos ON DELETE CASCADE, event_id, tip, alt_tip,
        t_start, t_end, frame_start, frame_end, conf,
        gps_lat, gps_lon, klip, notlar)

parties (id, event_row_id → events ON DELETE CASCADE, track_id, plaka, hasar)

violations (id, party_row_id → parties ON DELETE CASCADE, ihlal_kodu,
            ktk_madde, kusur_sinifi, ceza_tutari_try, ceza_puani,
            ceza_tablosu_tarihi, conf)

INDEX ix_events_tip, ix_events_video
```

**Kritik ayrıntı:** SQLite'ta `foreign_keys` PRAGMA'sı varsayılan **kapalıdır**.
Açılmazsa KVKK imhası sırasında taraf/ihlal satırları öksüz kalır — yani kişisel veri
silinmemiş olur. Açık olduğu testle doğrulanır.

**Bilinçli olarak yazılmayanlar:** ayrı `cameras` tablosu, kullanıcı yönetimi, tenant
izolasyonu. `videos.camera_id` ve `events.gps_*` çoklu kamera için baştan var.

---

## 5. HTTP API

Taban adres: `http://127.0.0.1:8000` · Canlı şema: `/docs` (OpenAPI)

| Uç | Yöntem | İş |
|---|---|---|
| `/` | GET | Tek sayfa arayüz (HTML) |
| `/videos` | POST | Video yükle, analizi arka planda başlat |
| `/videos` | GET | Video listesi |
| `/videos/{id}/status` | GET | İşleme durumu |
| `/videos/{id}/annotated` | GET | İşaretlenmiş video (mp4) |
| `/events` | GET | Filtreli olay listesi |
| `/events/{event_id}` | GET | Tek olay (taraflar, kusur, ceza) |
| `/events/{event_id}/klip` | GET | Olay klibi (mp4) |
| `/events/{event_id}/is-emri.pdf` | GET | Fen İşleri PDF iş emri |
| `/kara-nokta` | GET | Kara nokta analizi |
| `/kameralar` | GET | Kamera başına video/olay özeti |
| `/kvkk/purge` | POST | Saklama süresi dolanları imha et |

---

### 5.1 `POST /videos` — yükleme

**Tek girdi videodur.** Kaynak profili, dedektörler, kamera ve tarih kullanıcıya
sorulmaz; `adgs/probe.py` bunları videodan türetir. Gerekçe: kullanıcının elle girdiği
"sabit kamera" veya "15.07.2026" doğrulanamayan bir beyandı; ölçülen değer yanlış
olabilir ama nedeni raporlanır.

```bash
curl -F "file=@arnavutkoy_kavsak_01_2026-07-15.mp4" http://127.0.0.1:8000/videos
```

```json
{
  "video_id": 12,
  "durum": "BEKLIYOR",
  "turetilen": {
    "profile": "cctv_fixed",
    "detect": "accident,ihlal",
    "camera": "arnavutkoy_kavsak_01",
    "tarih": "2026-07-15",
    "conf": 0.35,
    "kamera_kaymasi": 0.004,
    "isik": 137,
    "gerekce": [
      "kamera kayması %0.4 (<%1) → sabit kamera",
      "tarih dosya adından: 2026-07-15",
      "kamera dosya adından eşleşti: arnavutkoy_kavsak_01",
      "ışık medyanı 137/255 (gündüz) → conf 0.35"
    ]
  },
  "uyari": "Analiz arka planda calisiyor; /videos/{id}/status ile takip edin."
}
```

**Türetme kuralları**

| Alan | Nereden | Ölçüm / eşik |
|---|---|---|
| `profile` | Faz korelasyonuyla kamera kayması | <%1 sabit · >%3 araca monteli |
| `detect` | Profilden | sabit → `accident,ihlal` · araca monteli → `roaddamage` |
| `camera` | Dosya adı | `config/cameras/` altındaki tanımlı kimlikle **birebir** eşleşme |
| `tarih` | Dosya adı | `2026-07-15` · `20260715` · `15.07.2026` |
| `conf` | HSV parlaklık medyanı | <100/255 gece → `conf 0.15` |

Kamera kayması ölçüldü (medyan): sabit sahne **%0.004**, kaydırılan kamera **%19.3** —
iki sınıf arasında ~50× fark var, eşiğin tam yeri kritik değil. Bitişik karelerdeki
*piksel yoğunluğu farkı* bu iş için yetersizdi: 30 fps'te sahne birkaç piksel kayar ve
düz yüzeylerde (asfalt, gökyüzü) fark eşiğin altında kalır — dashcam görüntüsü "sabit"
ölçülüyordu.

**Türetilemeyen alan `null` kalır, tahmin edilmez.** İkisi özellikle önemli: tarih bugüne
düşmez (arşiv videosu bugünün ceza tablosuyla hesaplanırsa sessizce yanlış tutar üretir),
kamera çözünürlükten tahmin edilmez (yanlış kamera = yanlış dur çizgisi = bir vatandaş
adına yanlış ihlal kaydı).

**İşleme senkron değildir.** Analiz dakikalar sürdüğü için HTTP isteği bekletilmez;
istemci durumu yoklar. Boru hattı burada yeniden yazılmaz — `cli.run` çağrılır, ürettiği
`rapor.json` veritabanına alınır.

---

### 5.2 `GET /videos/{id}/status`

```json
{
  "id": 12,
  "dosya": "data/uploads/arnavutkoy_kavsak_01_2026-07-15.mp4",
  "camera_id": "arnavutkoy_kavsak_01",
  "kaynak_profil": "cctv_fixed",
  "cekim_tarihi": "2026-07-15",
  "islenme_zamani": "2026-07-16T09:14:02+00:00",
  "durum": "TAMAM",
  "hata": null,
  "olay_sayisi": 3
}
```

Durum akışı: `BEKLIYOR → ISLENIYOR → TAMAM | HATA`. `HATA` ise sebep `hata` alanındadır.
404: video bulunamadı.

---

### 5.3 `GET /events`

**Sorgu parametreleri:** `tip` (`KAZA`|`IHLAL`|`ALTYAPI`), `tarih` (ISO, videonun çekim
tarihi), `camera_id`, `video_id`, `limit` (1–1000, varsayılan 200)

```bash
curl "http://127.0.0.1:8000/events?tip=IHLAL&camera_id=arnavutkoy_kavsak_01"
```

```json
{
  "sayi": 2,
  "uyari": "Bu çıktı bir karar destek analizidir; bağlayıcı bir tespit değildir.",
  "events": [ /* bkz. 5.4 */ ]
}
```

---

### 5.4 `GET /events/{event_id}`

```json
{
  "event_id": "ihlal_0042",
  "tip": "IHLAL",
  "alt_tip": "KIRMIZI_ISIK",
  "t_start": 84.2,
  "t_end": 86.7,
  "frame_start": 2526,
  "frame_end": 2601,
  "conf": 0.81,
  "gps": null,
  "klip": "runs/api/12/klipler/ihlal_0042.mp4",
  "video": "data/uploads/arnavutkoy_kavsak_01_2026-07-15.mp4",
  "video_id": 12,
  "camera_id": "arnavutkoy_kavsak_01",
  "cekim_tarihi": "2026-07-15",
  "notlar": [
    "Bu çıktı görüntü analizine dayalı bir ön değerlendirmedir. Kusur oranı; kaza tespit tutanağı, tanık ifadeleri, bilirkişi/eksper raporu ve TRAMER değerlendirmesi ile yetkili merciler tarafından belirlenir. Bu sistem bağlayıcı bir tespit üretmez.",
    "ceza tablosu doğrulanmamıştır (meta.dogrulama_tarihi boş)"
  ],
  "parties": [
    {
      "track_id": 17,
      "plaka": null,
      "hasar": null,
      "violations": [
        {
          "ihlal_kodu": "KIRMIZI_ISIK",
          "ktk_madde": "47/1-c",
          "kusur_sinifi": "ASLI",
          "ceza_tutari_try": 2167,
          "ceza_puani": 20,
          "ceza_tablosu_tarihi": "2026-02-27",
          "conf": 0.81
        }
      ]
    }
  ]
}
```

> Yukarıdaki tutar ve puan **örnek gösterimdir**; gerçek değer olayın tarihine ait
> tablodan okunur. `plaka: null` OCR'ın doğrulamayı geçemediği anlamına gelir — yanlış
> okunan bir plaka, okunmamış plakadan kötüdür.

404: olay bulunamadı.

---

### 5.5 `GET /events/{id}/klip` ve `GET /videos/{id}/annotated`

`video/mp4` döner. Her ikisinde de KVKK bulanıklaştırması uygulanmıştır ve işaretli
videoda kaldırılamayan *ÖN DEĞERLENDİRME* filigranı vardır. Kalibrasyon doğrulanmamışsa
kareye `KALIBRASYON YOK` uyarısı basılır.

404: dosya bulunamadı (canlı akış koşularında işaretli video üretilmez — akış geri
sarılamaz).

---

### 5.6 `GET /events/{id}/is-emri.pdf`

Fen İşleri iş emri. Şiddet ve öncelik M5'in notlarından **okunur, yeniden
hesaplanmaz** — iki farklı cevap veren iki gerçek kaynağı olmasın diye.

**Yeni bağımlılık eklenmedi.** reportlab/fpdf2'nin gömülü fontları Latin-1'dir ve
`ş/ğ/İ` basamaz; resmî bir iş emrinde bozuk metin kabul edilemez. matplotlib zaten
Ultralytics ile kurulu gelir ve DejaVu Sans'ı paket içinde taşır.

---

### 5.7 `GET /kara-nokta`

**Parametreler:** `yaricap_m` (0 < x ≤ 2000, varsayılan 50), `tip`

```json
{
  "yaricap_m": 50.0,
  "toplam_olay": 34,
  "gps_li_olay": 12,
  "cografi_kumeler": [
    {"lat": 41.1842, "lon": 28.7402, "olay": 5, "skor": 13, "tipler": {"KAZA": 2, "ALTYAPI": 3}}
  ],
  "kamera_gruplari": [
    {"camera_id": "arnavutkoy_kavsak_01", "olay": 22, "skor": 48, "tipler": {"IHLAL": 19, "KAZA": 3}}
  ],
  "uyari": "Kara nokta analizi bir PLANLAMA çıktısıdır; kusur veya ceza doğurmaz. Ağırlıklar göreli sıralama ölçeğidir, mutlak risk skoru değildir — gerçek analiz yaralanma ve maddi hasar verisi gerektirir."
}
```

**İki gruplama ayrı alanlarda döner ve birleştirilmez.** GPS'li olaylar haversine ile
coğrafi kümelenir; GPS'siz sabit kamera olayları kamera bazlı gruplanır. Sabit kameranın
olaylarına uydurma koordinat verip haritaya basmak, var olmayan bir kara nokta üretirdi.

Ağırlık: `KAZA 5 > IHLAL 2 > ALTYAPI 1`, yol hasarında şiddet çarpanıyla.

---

### 5.8 `GET /kameralar`

```json
{
  "kameralar": [
    {"camera_id": "arnavutkoy_kavsak_01", "video": 4, "olay": 22,
     "tipler": {"IHLAL": 19, "KAZA": 3}},
    {"camera_id": "(kamera tanimsiz)", "video": 2, "olay": 0, "tipler": {}}
  ]
}
```

Olay sayısına göre azalan sıralı. `(kamera tanimsiz)` = kamera türetilemedi.

---

### 5.9 `POST /kvkk/purge`

**Parametre:** `gun` (verilmezse `config/pipeline.yaml` → `saklama_gun`, varsayılan 30)

```bash
curl -X POST "http://127.0.0.1:8000/kvkk/purge?gun=30"
```

Süresi dolan kayıtları **ve klip dosyalarını** siler. Sadece satırı silmek diskte kişisel
veri bırakırdı. Cascade ile taraf ve ihlal satırları da gider.

400: `gun` pozitif değil.

---

### 5.10 Hata davranışı

| Kod | Anlamı |
|---|---|
| 200 | Başarılı |
| 400 | Geçersiz parametre (örn. `gun <= 0`) |
| 404 | Video / olay / dosya bulunamadı |
| 422 | FastAPI doğrulama hatası (parametre tipi/aralığı) |

Analiz hataları HTTP hatası **değildir** — işleme arka planda çalıştığı için hata
`videos.durum = "HATA"` ve `videos.hata` alanına yazılır, `/videos/{id}/status` ile
okunur.

---

## 6. Kamera yapılandırma şeması

`config/cameras/<camera_id>.yaml`

```yaml
camera_id: arnavutkoy_kavsak_01
kaynak_profil: cctv_fixed
maks_hata_yuzde: 10.0

homografi:                     # piksel ↔ yer düzlemi (metre)
  piksel: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
  dunya:  [[X1,Y1], [X2,Y2], [X3,Y3], [X4,Y4]]

dogrulama:                     # BAĞIMSIZ ölçüm olmalı
  - piksel: [[ax,ay], [bx,by]]
    gercek_m: 5.0
    aciklama: "kaldırım köşesi → rögar, şerit ölçümü"

roi: null

# --- Faz 4 geometrisi -------------------------------------------------------
dur_cizgisi: [[x1,y1], [x2,y2]]
isik_roi: [x1, y1, x2, y2]

seritler:
  - ad: "sol_serit_asagi"
    poligon: [[...], [...], [...], [...]]
    yon: [0, 1]                # görüntü uzayında doğru yön vektörü

yasak_park:
  - ad: "kose_yasak_alan"
    poligon: [[...], [...], [...], [...]]

hiz_limiti_kmh: 50
```

**Alan → yetenek eşlemesi** [KULLANICI_KILAVUZU.md §4.1](KULLANICI_KILAVUZU.md)'de.

Doğrulama noktası homografi fitine **girmemelidir**; girerse hata ~0 çıkar ve test
hiçbir şey kanıtlamaz. `demo_sentetik.yaml` bilerek bu kusuru taşır — yalnızca kod
yolunu çalıştırmak içindir.

---

## 7. M4 → M6 → M7 zinciri (hukuki çekirdek)

### M4 — altı dedektör, ortak sözleşme

| Kod | Tetikleme | Ön koşul |
|---|---|---|
| `KIRMIZI_ISIK` | ışık KIRMIZI **ve** dur çizgisi geçildi **ve** araç ilerlemeye devam etti | `dur_cizgisi` + `isik_roi` |
| `TERS_YON` | şerit yönüyle >120° açı, ≥8 ardışık karede | `seritler[].yon` |
| `HATALI_PARK` | yasak park poligonunda ≥60 sn hareketsiz | `yasak_park[].poligon` |
| `TAKIP_MESAFESI` | takip süresi <2 sn, ≥20 km/s hızda | **doğrulanmış kalibrasyon** |
| `HIZ_IHLALI` | limit + %10 tolerans aşımı, ≥5 ardışık karede | **doğrulanmış kalibrasyon** + `hiz_limiti_kmh` |
| `SERIT_IHLALI` | ani şerit değişimi — **varsayılan kapalı** | ≥2 `seritler[].poligon` |

M4 `ktk_madde`, `kusur_sinifi` ve ceza alanlarını **boş bırakır**.

### M6 — KTK m.84 kapalı listesi

`config/rules/fault_ktk84.yaml` (a–l, 12 bent). Üç sonuç: `ASLI`, `TALI`,
`TESPIT_EDILEMEDI`. Sonuncusu **"kusursuz" demek değildir** — o tarafta ihlal
çıkarılamadı demektir.

Kapalı liste disiplininin somut sonucu — 6 koddan yalnızca ikisi asli kusur doğurur:

| Kod | Sınıf | Neden |
|---|---|---|
| `KIRMIZI_ISIK` | **ASLI** 84/a | listede |
| `TERS_YON` | **ASLI** 84/b | listede |
| `HATALI_PARK` | TALI | 84/k yalnızca yerleşim **dışı** karayolu içindir |
| `SERIT_IHLALI` | TALI | 84/g "şeride tecavüz"tür, şerit *değişimi* değil |
| `HIZ_IHLALI` | TALI | m.84'te sayılmaz |
| `TAKIP_MESAFESI` | TALI | m.84'te sayılmaz |

### M7 — tarih damgalı ceza tablosu

`config/penalties/<geçerlilik-tarihi>.yaml`. Kurallar:

- `penalty.py` içinde **tek bir tutar yoktur** — `grep -nE '[0-9]{4,}'` boş döner, test zorlar
- Tablo **olayın** tarihine göre seçilir
- `--tarih` verilmezse ceza **hesaplanmaz** (bugüne düşülmez)
- Uygun tablo yoksa eski tabloya **sessizce düşülmez**
- Karşılığı olmayan ihlal için tutar **uydurulmaz**
- Her çıktı hangi tablodan hesaplandığını taşır
- Erken ödeme indirimi hesaplanmaz (tebliğ tarihine bağlı, video bilemez)

> Her iki tablo da henüz **doğrulanmamıştır**. `meta.dogrulama_tarihi` boş olduğu sürece
> motor her çıktıya bunu bildiren uyarı ekler.

---

## 8. Ölçülmüş davranışlar

Bu tablodaki her sayı çalıştırılan bir ölçümden gelir; hiçbiri tahmin değildir.

| Konu | Ölçüm |
|---|---|
| Araç hasarı (CarDD, M8) | `mAP@0.5 = 0.509` — hedef ≥0.40, 13 epoch, yolo26s, 561 görüntülük val |
| Çözünürlük kapısı | 640 px: 0.514 · 384: 0.465 · **256: 0.371 (−%27.8) ← kapı** · 192: 0.270 · 128: 0.149 · 96: 0.104 |
| Sınıf bazında (AP@0.5:0.95) | tire flat 0.705 · glass shatter 0.674 · lamp broken 0.386 · scratch 0.167 · dent 0.147 · crack 0.025 |
| Kamera kayması | sabit sahne %0.004 · kaydırılan kamera %19.3 (~50× fark) |
| Gece güveni | aynı sahne: `conf 0.35` → 0 araç · `conf 0.15` → 9 araç |
| Sahne kesmesi | HSV histogram korelasyonu <0.35 → kesme |
| Kaza eleme dağılımı | 552 iz · 122.265 çift · 74 çakışma epizodu → 1 kabul |
| Test kapsamı | 365 test, tamamı geçiyor |

**Çözünürlük kapısı bir bulgudur, başarısızlık değil.** İlk yazımda eşik 128 idi ve bu
bir tahmindi; ölçüm onu çürüttü. Pratik sonuç: CCTV ölçeğinde yalnızca "lastik patlak" ve
"cam kırılması" güvenilir; çizik/çatlak/göçük ayrımı için yakın çekim gerekir.

---

## 9. Tanı ve yeniden üretim

Analizin pahalı kısmı takip (YOLO). Çizim ve tespit ayarını denemek için tekrarlanmaz:

```bash
adgs redetect runs/api/12/izler.json --video kayit.mp4 --tani   # 30 dk yerine ~2 sn
adgs rerender runs/api/12/rapor.json --video kayit.mp4
```

`--tani` her aday çiftin **neden** elendiğini sayar:

```
552 iz · 122.265 çift · 74 çakışma epizodu
  sahne kesmesi (montaj)         : 3
  okluzyon (yer düzleminde uzak) : 0
  ani hareket değişimi YOK       : 58   ← baskın eleme
  hareketsizlik YOK (devam etti) : 3
  hareketsizlik ÖLÇÜLEMEDİ       : 9
  KABUL EDİLEN                   : 1
```

Bu tablo olmadan eşik ayarı tahmine dönüşür.

---

## 10. Teknoloji yığını

| Katman | Seçim | Gerekçe |
|---|---|---|
| Detektör | Ultralytics YOLO26 (`yolo26n`/`yolo26s`) | Hazır takip entegrasyonu (BoTSORT), hızlı fine-tune |
| Takip | BoTSORT | Ultralytics içinde, kalıcı ID |
| Görüntü işleme | OpenCV | Homografi, HSV, faz korelasyonu, video I/O |
| Derin öğrenme | PyTorch + CUDA (cu128) | — |
| API | FastAPI + Uvicorn | OpenAPI şeması bedava, `BackgroundTasks` yeterli |
| Kalıcılık | SQLite | Tek kullanıcılı staj kapsamı; şema PostgreSQL'e taşınabilir |
| PDF | matplotlib (DejaVu Sans) | **Türkçe karakter** — reportlab/fpdf2 gömülü fontları Latin-1 |
| Yapılandırma | YAML | Kod değişmeden kural/tutar güncellemesi |
| Test | pytest | 365 test |

**Yeni bağımlılık eklememe disiplini:** PDF için matplotlib seçilmesi bunun örneğidir —
zaten kurulu gelen bir paket, doğru fontu paket içinde taşıyor.

---

## 11. Bilinçli olarak yapılmayanlar

| Madde | Neden |
|---|---|
| PostgreSQL + PostGIS geçişi | SQLite şeması zaten taşınabilir; gerçek ikinci kamera gelmeden migration altyapısı erken |
| Kullanıcı yönetimi / tenant izolasyonu | Staj kapsamı tek kullanıcı; kimlik doğrulama kurumsal dizinle entegre edilmeli, sahte bir tablo yanıltıcı olurdu |
| Araç içi davranış (kemer/telefon/kask) | Sabit CCTV 6–10 m yükseklikten bakar; ön cam yansımasıyla sürücü gövdesi çoğu karede görünmez |
| Ayrı plaka/yüz dedektörü | Geometrik yaklaşım kullanılıyor; fazladan bulanıklaştırmak eksikten iyidir. Eklenirse yalnızca `render.kvkk_bolgeleri()` değişir |
| Kusur yüzdesi | KTY m.156/3 — oranı TRAMER belirler |
| Otomatik ceza kesme | Belediyenin yetkisi yok (KTK) |

---

## 12. Lisans zinciri

| Bileşen | Lisans | Sonuç |
|---|---|---|
| Ultralytics YOLO26 | **AGPL-3.0** | Fine-tune ağırlıklar ve doğrudan bağlanan kod devralır |
| RDD2022 (yol hasarı) | **CC BY-SA 4.0** | Atıf zorunlu; türev çalışmalar (ağırlıklar dâhil) aynı lisansla paylaşılmalı |
| CarDD (araç hasarı) | **Ticari kullanıma kapalı** | Staj/akademik uygun; operasyonel teslimatta M8 kapatılmalı veya yeniden eğitilmeli |

Kapalı kaynak teslimat gerekirse detektör Apache-2.0 muadille (RF-DETR, D-FINE,
RT-DETRv2) değiştirilebilir — mimari değişmez, yalnızca `config/pipeline.yaml` içindeki
detektör seçimi ve ağırlıklar değişir. Ancak **bir model ağırlığı eğitildiği verinin
lisansını taşır**: RDD2022 yükümlülüğü detektör değişse de sürer.

**Atıflar**

> Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Sekimoto, Y. (2022). *RDD2022: A
> multi-national image dataset for automatic road damage detection.* arXiv:2209.08538

> Wang, X., Li, W., Wu, Z. (2023). *CarDD: A New Dataset for Vision-Based Car Damage
> Detection.* IEEE T-ITS
