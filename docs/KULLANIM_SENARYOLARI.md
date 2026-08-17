# ADGS — Kullanım Senaryoları

> **Kapsam notu.** ADGS bir *karar destek* aracıdır. Buradaki senaryoların hiçbiri
> otomatik ceza kesme, bağlayıcı kusur tespiti veya idari işlem üretmez. Belediyenin
> KTK ihlalleri için idari para cezası kesme yetkisi yoktur; asli görev alanı yol ve
> altyapı bakımıdır. Ayrıntı için [README](../README.md) "Kapsam ve Hukuki
> Konumlandırma" bölümü.

Her senaryo şu başlıklarla verilir: **aktör**, **tetikleyici**, **ön koşul**, **akış**,
**çıktı**, **başarısızlık davranışı**. Başarısızlık davranışı burada özellikle önemlidir:
ADGS'nin tasarım kararı, ön koşul sağlanmadığında **tahmin üretmek yerine çalışmayı
reddetmektir**.

---

## Aktörler

| Aktör | Kim | Sistemden beklentisi |
|---|---|---|
| **Fen İşleri personeli** | Yol bakım birimi | Yol hasarı için bakım iş emri |
| **Trafik/Ulaşım birimi** | Trafik güvenliği planlama | Kara nokta analizi, olay yoğunluğu |
| **Zabıta / Güvenlik** | Kamera izleme merkezi | Kaza anında kanıt paketi |
| **Sistem yöneticisi** | BT | Kurulum, kalibrasyon, KVKK imhası |
| **Veri sorumlusu** | KVKK sorumlusu | Saklama süresi, bulanıklaştırma, imha kaydı |

---

## S1 — Araca monteli kamerayla yol hasarı taraması

**Aktör:** Fen İşleri personeli
**Tetikleyici:** Belediye aracına takılan kamerayla güzergâh kaydı alındı.
**Ön koşul:** RDD2022 üzerinde eğitilmiş yol hasarı modeli (`runs/rdd2022/.../best.pt`).

**Akış**

1. Personel `data/` altına video dosyasını koyar.
2. Analizi başlatır:
   ```bash
   adgs run yol_20260715.mp4 --profile vehicle_mounted --detect roaddamage --out runs/f1
   ```
3. M5 (`roaddamage.py`) her karede hasar arar; aynı hasarın ardışık karelerdeki
   tekrarları tek olayda birleştirilir.
4. Her `ALTYAPI` olayı için şiddet ve öncelik notu üretilir.
5. Personel iş emrini alır:
   ```bash
   adgs workorder runs/f1/rapor.json --event altyapi_0007 --out is_emri.pdf --video yol_20260715.mp4
   ```

**Çıktı**

- `runs/f1/rapor.json` — olay listesi
- `runs/f1/*_isaretli.mp4` — hasarın çerçevelendiği video
- `is_emri.pdf` — Fen İşleri iş emri (konum, şiddet, öncelik, kanıt karesi)

**Başarısızlık davranışı**

- Model ağırlığı yoksa `adgs doctor` bunu bildirir; `run` sessizce boş rapor üretmez.
- GPS yoksa olay üretilir ama `gps: null` kalır — konum uydurulmaz; kara nokta analizi
  bu olayları coğrafi değil kamera bazlı gruplar.

**Bu senaryo neden belediye için birincil senaryodur:** üç çıktı tipinden yalnızca
`ALTYAPI` doğrudan belediyenin yetki alanındadır ve doğrudan bir operasyonel işe
(bakım iş emri) dönüşür.

---

## S2 — Sabit CCTV'de kaza tespiti ve kanıt paketi

**Aktör:** Zabıta / kamera izleme merkezi
**Tetikleyici:** Kavşak kamerasının günlük kaydı incelenecek.
**Ön koşul:** `config/cameras/<id>.yaml` sahada ölçülmüş; kalibrasyon doğrulaması geçer.

**Akış**

1. Kalibrasyon doğrulanır — geçmezse çıkış kodu 1:
   ```bash
   adgs calibrate --camera arnavutkoy_kavsak_01 --verify
   ```
2. Kaza tespiti çalıştırılır:
   ```bash
   adgs run kavsak.mp4 --detect accident --profile cctv_fixed \
           --camera arnavutkoy_kavsak_01 --out runs/f3
   ```
3. M3 (`accident.py`) bir olay üretmek için **üç sinyalin üçünü birden** arar:
   (1) iki izin kutuları çakışır, (2) *aynı* araçta çakışma sonrası ani hız düşüşü veya
   yön değişimi, (3) *aynı* araç ardından ≥2 sn hareketsiz kalır.
4. Kalibrasyon geçerliyse dördüncü filtre: görüntüde çakışan ama yer düzleminde 6 m'den
   uzak çiftler oklüzyon sayılıp elenir.
5. Her olay için klip ve keyframe kesilir. **Kanıtsız olay rapora yazılmaz.**

**Çıktı**

- `runs/f3/klipler/kaza_0001.mp4` — olay klibi (plaka/yüz bulanık)
- `runs/f3/rapor.json` — taraflar, izler, kanıt referansları
- Her `KAZA` olayında kaldırılamayan ön değerlendirme notu

**Başarısızlık davranışı**

- Sinyallerden biri eksikse olay üretilmez. `--tani` bayrağı hangi adayın **neden**
  elendiğini sayar — eşik ayarı tahmine dönüşmesin diye.
- Montaj/derleme videolarda sahne kesmesi sahte çakışma imzası üretir; sahne kesmesi
  tespiti bunları eler.

---

## S3 — Trafik ihlali ön değerlendirmesi ve kusur/ceza gerekçesi

**Aktör:** Trafik/Ulaşım birimi
**Tetikleyici:** Bir kavşakta kırmızı ışık ihlali şikâyeti geldi.
**Ön koşul:** Kamera geometrisi (`dur_cizgisi`, `isik_roi`, `seritler`) ölçülmüş.

**Akış**

1. İhlal tespiti — çekim tarihi **verilmelidir**, ceza tablosu ona göre seçilir:
   ```bash
   adgs run kavsak.mp4 --detect ihlal --camera arnavutkoy_kavsak_01 \
           --tarih 2026-07-15 --out runs/f5
   ```
2. M4 altı bağımsız dedektörü çalıştırır (`redlight`, `wrongway`, `parking`,
   `tailgating`, `speed`, `lane` — sonuncusu varsayılan kapalı).
3. M6 (`fault.py`) her ihlali KTK m.84 **kapalı listesiyle** eşleştirir:
   `ASLI` / `TALI` / `TESPIT_EDILEMEDI`. **Yüzde üretmez.**
4. M7 (`penalty.py`) tutarı, **olayın tarihine** ait tablodan okur.
5. Gerekçe okunur:
   ```bash
   adgs explain runs/f5/rapor.json --event ihlal_0042
   ```

**Çıktı**

Olay başına: ihlal kodu, KTK maddesi, kusur sınıfı, ceza tutarı + hangi tablodan
hesaplandığı, ön değerlendirme notu.

**Başarısızlık davranışı — bu senaryonun en kritik parçası**

| Durum | Sistem ne yapar |
|---|---|
| `dur_cizgisi` tanımlı değil | `redlight` **çalışmaz**, sebebi `rapor.json` → `calisamayan_moduller` |
| Kalibrasyon doğrulanmamış | `speed` ve `tailgating` **çalışmayı reddeder** |
| `--tarih` verilmedi | Ceza **hesaplanmaz** — bugüne düşülmez |
| Tarihe uygun tablo yok | Eski tabloya **sessizce düşülmez** |
| İhlal tabloda yok | Tutar **uydurulmaz** |

Sessizce boş dönmek en tehlikeli yanlıştır: okuyan kişi "ihlal yok" sanar, oysa gerçek
anlamı "bakılmadı"dır.

---

## S4 — Web arayüzünden tek dosyayla analiz (operatör senaryosu)

**Aktör:** Teknik olmayan operatör
**Tetikleyici:** Elde bir video var, komut satırı bilgisi yok.
**Ön koşul:** `pip install -e ".[api]"`, servis ayakta.

**Akış**

1. `adgs serve` → tarayıcıda `http://127.0.0.1:8000/`
2. Operatör **yalnızca video dosyasını** yükler. Profil, dedektör, kamera ve tarih
   **sorulmaz** — `adgs/probe.py` bunları videodan türetir:

   | Alan | Nereden | Nasıl |
   |---|---|---|
   | `profile` | ölçüm | yarım saniyedeki kamera kayması (faz korelasyonu) |
   | `detect` | profilden | sabit → `accident,ihlal` · araca monteli → `roaddamage` |
   | `camera` | dosya adı | `config/cameras/` altındaki **tanımlı** kimlikle birebir eşleşme |
   | `tarih` | dosya adı | `2026-07-15` · `20260715` · `15.07.2026` |

3. Durum `BEKLIYOR → ISLENIYOR → TAMAM|HATA` diye ilerler; arayüz kendiliğinden yoklar.
4. Türetilen değerler ve **gerekçeleri** yüklemenin hemen altında gösterilir.
5. Olaylar tablosundan klip ve PDF iş emri indirilir.

**Başarısızlık davranışı**

Türetilemeyen alan **boş bırakılır**, tahmin edilmez. İkisi özellikle önemli:

- **Tarih** bugüne düşmez — arşiv videosu bugünün ceza tablosuyla hesaplanırsa sessizce
  yanlış tutar üretir.
- **Kamera** çözünürlükten tahmin edilmez — yanlış kamera = yanlış dur çizgisi = bir
  vatandaş adına yanlış ihlal kaydı.

Boş kalan alan ilgili modülün kendi reddetme mekanizmasını çalıştırır.

---

## S5 — Kara nokta analizi (yatırım planlaması)

**Aktör:** Trafik/Ulaşım birimi, planlama
**Tetikleyici:** Yıllık trafik güvenliği yatırım listesi hazırlanacak.
**Ön koşul:** Veritabanında birikmiş olay kaydı (`adgs store` ile alınmış).

**Akış**

```bash
adgs hotspot --db data/adgs.db --yaricap 50
# veya: GET /kara-nokta?yaricap_m=50
```

Ağırlık: `KAZA 5 > IHLAL 2 > ALTYAPI 1`; yol hasarında şiddet çarpanıyla.

**İki gruplama birbirine karıştırılmaz — bu kasıtlıdır:**

| Kaynak | Gruplama |
|---|---|
| GPS var (araca monteli) | haversine ile coğrafi kümeleme, yarıçap ayarlanabilir |
| GPS yok (sabit CCTV) | kamera bazlı gruplama |

Sabit kameranın olaylarına uydurma koordinat verip haritaya basmak, var olmayan bir kara
nokta üretirdi.

**Çıktı:** noktaların olay sayısı ve ağırlıklı skora göre sıralaması.

**Sınır:** ağırlıklar **göreli sıralama ölçeğidir**, mutlak risk skoru değil. Gerçek kara
nokta analizi yaralanma ve maddi hasar verisi gerektirir; bu sistemde o veri yoktur.

---

## S6 — Canlı RTSP akışından izleme

**Aktör:** Kamera izleme merkezi
**Tetikleyici:** Canlı kamerada anlık takip görüntüsü isteniyor.

```bash
adgs track rtsp://kamera/stream --sure 60
```

- `--sure` **zorunludur**: akışın sonu yoktur, sınır verilmezse döngü hiç bitmez.
- Canlı akışta işaretli video üretilmez (akış geri sarılamaz); izler yazılır.
- `adgs run` dosya gerektirir — klip kesme ve işaretleme kaynağı yeniden okur.

---

## S7 — KVKK saklama süresi ve imha

**Aktör:** Veri sorumlusu
**Tetikleyici:** Saklama süresi dolan kayıtlar imha edilecek.

```bash
adgs purge                       # config/pipeline.yaml → saklama_gun (varsayılan 30)
# veya: POST /kvkk/purge?gun=30
```

Üç mekanizma, üçü de testli:

| Mekanizma | Davranış |
|---|---|
| Saklama süresi | Süresi dolan **kayıtları ve klip dosyalarını** siler — sadece satırı silmek diskte kişisel veri bırakırdı |
| Bulanıklaştırma | `plaka_bulaniklastir` / `yuz_bulaniklastir` varsayılan **açık**; yaya kutusunun üstü ve araç kutusunun alt-ortası Gaussian blur |
| Cascade | SQLite `foreign_keys` PRAGMA'sı açık — kapalıysa imhada taraf/ihlal satırları öksüz kalırdı |

Bulanıklaştırma bölgeleri **geometrik yaklaşımdır** — ayrı plaka/yüz dedektörü yoktur.
Fazladan alan bulanıklaştırmak, eksik bulanıklaştırmaktan iyidir.

---

## S8 — Tanı ve yeniden üretim (geliştirici senaryosu)

**Aktör:** Sistem yöneticisi / geliştirici
**Tetikleyici:** Çizim veya eşik ayarı denenecek; 30 dakikalık takip tekrarlanmamalı.

```bash
adgs redetect runs/api/8/izler.json --video kayit.mp4 --tani   # ~2 sn
adgs rerender runs/api/8/rapor.json --video kayit.mp4          # tespit tekrar çalışmaz
```

`--tani` her aday çiftin neden elendiğini sayar. Gerçek bir koşudan:

```
552 iz · 122.265 çift · 74 çakışma epizodu
  sahne kesmesi (montaj)         : 3
  okluzyon (yer düzleminde uzak) : 0
  ani hareket değişimi YOK       : 58   ← baskın eleme
  hareketsizlik YOK (devam etti) : 3
  hareketsizlik ÖLÇÜLEMEDİ       : 9
  KABUL EDİLEN                   : 1
```

---

## S9 — Kabul kriterlerinin ölçülmesi

**Aktör:** Proje yürütücüsü / danışman

```bash
adgs kabul          # tablo
adgs kabul --json   # rapora gömmek için
```

Her satır üç durumdan birini alır. **"Kod yazıldı" bir durum değildir.**

| Durum | Anlamı |
|---|---|
| `GECTI` | ölçüldü ve hedefi tutturdu |
| `KALDI` | **ölçüldü** ve hedefin altında kaldı |
| `OLCULMEDI` | ölçüm koşulu yok (eğitilmiş model veya etiketli veri gerekiyor) |

Çıkış kodu yalnızca `KALDI` varsa sıfırdan farklıdır: ölçülememiş bir kriter hata
değildir, ölçülüp hedefin altında kalan kriter hatadır.

---

## Senaryo–modül matrisi

| Senaryo | M1 | M2 | M3 | M4 | M5 | M6 | M7 | M8 | M9 | M10 |
|---|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| S1 Yol hasarı | ● | | | | ● | | | | ● | ○ |
| S2 Kaza | ● | ● | ● | | | | | ○ | ● | ○ |
| S3 İhlal + kusur/ceza | ● | ● | | ● | | ● | ● | | ● | ○ |
| S4 Web arayüzü | ● | ● | ● | ● | ● | ● | ● | ○ | ● | ● |
| S5 Kara nokta | | | | | | | | | | ● |
| S6 RTSP | ● | ● | | | | | | | | |
| S7 KVKK imha | | | | | | | | | | ● |

● zorunlu · ○ koşullu (model/kalibrasyon varsa)

---

## Senaryoların kapsamadıkları

| Talep | Neden yok |
|---|---|
| Otomatik ceza kesme | Belediyenin yetkisi yok (KTK) |
| Kusur yüzdesi | KTY m.156/3 — tutanağı düzenleyen görevli bile oran belirtmez; oranı TRAMER belirler |
| Araç içi davranış (kemer/telefon/kask) | Sabit CCTV 6–10 m yükseklikten bakar; ön cam yansımasıyla sürücü gövdesi çoğu karede görünmez |
| Şerit ihlali (varsayılan) | KTK m.55 şerit değiştirmeyi değil *sinyal vermeden* değiştirmeyi yasaklar; CCTV mesafesinde sinyal lambası güvenilir okunamaz |
| CCTV'de çizik/göçük ayrımı | Ölçüldü: 256 px altında mAP %27.8 düşüyor, ince dokulu hasar çalışmıyor |
