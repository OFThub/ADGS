# ADGS — Trafik Video Analiz Sistemi

Trafik videolarını analiz ederek **yol/altyapı hasarı**, **kaza** ve **trafik kural ihlali**
tespiti yapan bir karar destek aracı. Arnavutköy Belediyesi staj projesi.

---

## ⚠️ Kapsam ve Hukuki Konumlandırma

**Bu sistem bir karar destek / analiz aracıdır. Otomatik veya bağlayıcı bir
cezalandırma mekanizması değildir.**

Bu, teknik bir tercih değil hukuki bir zorunluluktur:

### Yetki

Belediyenin 2918 sayılı Karayolları Trafik Kanunu (KTK) ihlalleri için **idari para
cezası kesme yetkisi yoktur**; bu yetki Emniyet Genel Müdürlüğü / Jandarma'dadır.
Belediyenin asli görev alanı **yol ve altyapı bakımıdır**.

Buna göre sistemin çıktıları şu şekilde konumlanır:

| Çıktı | Belediye açısından niteliği |
|---|---|
| Yol/altyapı hasarı | **Operasyonel** — Fen İşleri için bakım iş emri girdisi |
| Kaza tespiti | **Planlama** — kara nokta analizi, trafik güvenliği yatırım kararı |
| İhlal tespiti | **Bilgilendirme** — gerektiğinde yetkili mercilere iletilecek kanıt paketi |
| Kusur / ceza tahmini | **Yalnızca ön değerlendirme** — hiçbir idari işleme dayanak oluşturmaz |

### Delil

Gerçek kusur tespiti; kaza tespit tutanağı, tanık ifadeleri, bilirkişi/eksper raporu
ve TRAMER değerlendirmesi gibi videonun tek başına sağlayamayacağı unsurları kullanır.

Karayolları Trafik Yönetmeliği m.156/3 uyarınca tutanağı düzenleyen görevliler dahi
**kusur oranı belirtmez**; yalnızca hangi tarafın hangi trafik kuralını ihlal ettiğini
kaydeder. Kusur oranını TRAMER belirler.

Bu nedenle sistemin kusur modülü **yüzde üretmez**. Bunun yerine tutanak mantığını izler:
her taraf için hangi KTK maddesinin ihlal edildiğini ve bunun KTK m.84 kapsamında
**asli** mi **tali** mi olduğunu raporlar.

Her KAZA ve İHLAL çıktısı, kaldırılamayan şu notu taşır:

> Bu çıktı görüntü analizine dayalı bir ön değerlendirmedir. Kusur oranı; kaza tespit
> tutanağı, tanık ifadeleri, bilirkişi/eksper raporu ve TRAMER değerlendirmesi ile
> yetkili merciler tarafından belirlenir. Bu sistem bağlayıcı bir tespit üretmez.

### Kullanım amacı

Bu proje akademik / staj / araştırma amaçlıdır. Gerçek dünyada operasyonel kullanım
(örn. otomatik ceza kesme) ayrı bir hukuki ve idari süreç, yetkilendirme ve denetim
gerektirir.

---

## 🔒 KVKK

Video görüntüleri kişisel veri içerir (yüz, plaka). Belediye bu veriler açısından
**veri sorumlusu** konumundadır.

- Ham video, kare ve model ağırlıkları repoya **girmez** (`data/` gitignore'dadır)
- Saklama süresi `config/pipeline.yaml` içinde tanımlıdır (`saklama_gun`), süre sonunda imha
- Plaka ve yüz bulanıklaştırma varsayılan olarak **açıktır**
- Aydınlatma metni ve VERBİS kaydının bu işleme faaliyetini kapsadığı doğrulanmalıdır
- Erişim kontrolü ve erişim logu gereklidir

---

## 🎬 Demo — 2 komut

Sunum için en kısa yol:

```bash
pip install -e ".[api,dev]"
adgs serve                       # http://127.0.0.1:8000/
```

Tarayıcıda açılan sayfada:

1. **Video yükle** — `data/testset/demo_pan.mp4`, profil `cctv_fixed`,
   tespit `kaza + ihlal`, kamera `demo_sentetik`, tarih seç → *Yükle ve işle*
2. Durum `BEKLIYOR → ISLENIYOR → TAMAM` diye kendiliğinden ilerler (~20 sn)
3. **İşaretlenmiş video** bölümünde sonuç oynar: her araç kalıcı takip
   numarasıyla çerçeveli, altta kaldırılamayan *ÖN DEĞERLENDİRME* filigranı
4. **Olaylar** tablosunda tespitler; satıra tıklayınca gerekçe notları açılır,
   sağdan **klip** ve **PDF iş emri** indirilir
5. **Kara nokta analizi** bölümünde konum/kamera bazlı risk sıralaması

> `demo_sentetik` kamerasının şerit yönleri **uydurmadır**, gerçek ölçüm
> değildir. Sunumda çıkan `TERS_YON` tespitleri kod yolunun çalıştığını
> gösterir; gerçek ihlal değildir. Gerçek kamera için
> `config/cameras/arnavutkoy_kavsak_01.yaml` sahada ölçülüp doldurulmalıdır —
> ölçülmediği sürece sistem o kamerada **tespit yapmayı reddeder**.

Sadece takip görüntüsü isteniyorsa (en hızlı görsel, veritabanı gerekmez):

```bash
adgs track data/testset/demo_pan.mp4 --out runs/demo
# runs/demo/demo_pan_tracked.mp4
```

## Kurulum

```bash
python -m venv .venv && .venv/Scripts/activate   # Windows
pip install -e ".[dev]"
adgs doctor
```

`adgs doctor` GPU/CUDA, detektör, yapılandırma ve şema kontrollerini çalıştırır.

> **Python 3.12 gerekir.** PyTorch'un 3.14 için wheel'i yoktur. Ortam
> `uv venv --python 3.12 .venv` ile kurulmalıdır.

## Kullanım

### Faz 1 — yol hasarı (araca monteli kamera)

```bash
python training/prep_rdd2022.py --src data/raw/RDD2022_ext --out data/rdd_yolo
python training/train_rdd2022.py            # amp KAPALI, batch 4 (bkz. not)
adgs run yol.mp4 --profile vehicle_mounted --out runs/f1
```

> **Eğitimde `--amp` açmayın.** RTX 3050 4 GB üzerinde fp16, `cls_loss`
> taşmasıyla NaN üretiyor ve eğitim çöpe gidiyor. Ölçüm ve elenen hipotezler
> `training/train_rdd2022.py` modül başlığında kayıtlıdır. `--amp` kapalıyken
> aynı taşma sonlu kalıyor ve gradyan kırpması sayesinde zarar vermiyor.
> Eğitim betiği ayrıca ilk NaN'da **anında durur** — sessiz çöp koşu üretmez.

### Faz 2 — tespit ve takip (sabit CCTV)

```bash
adgs calibrate --camera arnavutkoy_kavsak_01 --verify   # geçmezse çıkış kodu 1
adgs track kavsak.mp4 --camera arnavutkoy_kavsak_01 --out runs/f2
```

Kalibrasyon doğrulanmazsa dünya koordinatı ve hız **üretilmez**; işaretli
videoya `KALIBRASYON YOK` uyarısı basılır. Bu kasıtlıdır: doğrulanmamış bir
homografi yanlış hız, yanlış hız da yanlış ön değerlendirme üretir.

### Faz 3 — kaza tespiti

```bash
adgs run kavsak.mp4 --detect accident --profile cctv_fixed \
        --camera arnavutkoy_kavsak_01 --out runs/f3
```

Kural tabanlı (Aşama A). Bir olay üretilmesi için **üç sinyalin üçü birden**
gerekir: (1) iki izin kutuları çakışır, (2) *aynı* araçta çakışma sonrası ani
hız düşüşü veya yön değişimi olur, (3) *aynı* araç ardından ≥2 sn hareketsiz
kalır. Sinyal 2 ve 3'ün aynı tarafta aranması kritiktir — ayrı taraflarda
aranırsa kavşakta park hâlinde duran her araç yanlış pozitif üretir.

Kamera kalibrasyonu geçerliyse dördüncü filtre devreye girer: görüntüde çakışan
ama yer düzleminde 6 m'den uzak çiftler oklüzyon sayılıp elenir.

Her kaza olayı klip (`runs/f3/klipler/kaza_0001.mp4`) ve keyframe kanıtı taşır;
kanıtsız olay rapora yazılmaz.

### Faz 4 — trafik ihlali tespiti

```bash
adgs run kavsak.mp4 --detect ihlal --camera arnavutkoy_kavsak_01 \
        --profile cctv_fixed --out runs/f4
# tek tip:
adgs run kavsak.mp4 --detect redlight,wrongway --camera arnavutkoy_kavsak_01
```

Altı bağımsız dedektör; her biri `config/rules/violations.yaml` içindeki
`aktif` bayrağıyla ayrı ayrı açılıp kapanır.

| Tip | Tetikleme | Ön koşul |
|---|---|---|
| `redlight` | ışık KIRMIZI **ve** dur çizgisi geçildi **ve** araç ilerlemeye devam etti | `dur_cizgisi` + `isik_roi` |
| `wrongway` | şerit yönüyle >120° açı, ≥8 ardışık karede | `seritler[].yon` |
| `parking` | yasak park poligonu içinde ≥60 sn hareketsiz | `yasak_park[].poligon` |
| `tailgating` | takip süresi <2 sn, ≥20 km/s hızda | **doğrulanmış kalibrasyon** |
| `speed` | limit + %10 tolerans aşımı, ≥5 ardışık karede | **doğrulanmış kalibrasyon** + `hiz_limiti_kmh` |
| `lane` | ani şerit değişimi — **varsayılan kapalı** | ≥2 `seritler[].poligon` |

**Gerekli geometri yoksa modül tahmin üretmez, çalışmayı reddeder** ve sebebini
hem konsola hem `rapor.json` içindeki `calisamayan_moduller` alanına yazar.
Sessizce boş dönmek en tehlikeli yanlıştır: okuyan kişi "ihlal yok" sanar,
oysa gerçek anlamı "bakılmadı"dır.

> **Hız ve takip mesafesi, kalibrasyon doğrulamayı geçmeden çalışmaz.** Bu
> davranış test edilmiştir (`test_hiz_kalibrasyonsuz_calismayi_REDDEDER`).
> Gerekçe: hız homografinin doğruluğuna doğrudan bağlıdır; %20 hatalı bir
> homografi 50 km/s'lik aracı 60 km/s gösterir ve bu sayı ceza doğuran bir
> çıktı olarak dolaşıma girer.

> **Şerit ihlali varsayılan olarak kapalıdır.** KTK m.55 şerit değiştirmeyi
> değil, *sinyal vermeden* şerit değiştirmeyi yasaklar; sabit CCTV mesafesinde
> sinyal lambası güvenilir okunamaz. Açılırsa çıktı bir ihlal tespiti değil,
> insan incelemesi için adaydır (güven bilerek ≤0.3).

Kamera geometrisi (`dur_cizgisi`, `isik_roi`, `seritler`, `yasak_park`,
`hiz_limiti_kmh`) sahada ölçülüp `config/cameras/<id>.yaml` içine yazılır.
`config/cameras/demo_sentetik.yaml` yalnızca kod yolunu çalıştırmak içindir —
şerit yönleri uydurmadır ve gerçek videoda **yanlış ters yön tespiti üretir**.

M4 yalnızca *ne olduğunu* söyler. `ktk_madde`, `kusur_sinifi` ve ceza alanları
boş bırakılır; onları Faz 5'te M6/M7 doldurur.

### Faz 5 — kusur ve ceza motorları

```bash
adgs run kavsak.mp4 --detect ihlal --camera arnavutkoy_kavsak_01 \
        --tarih 2026-07-15 --out runs/f5
adgs explain runs/f5/rapor.json --event ihlal_0042
```

İkisi de GPU'suz ve saf: yalnızca `Event` + YAML okurlar. Bu yüzden sistemin
hukuken en hassas kısmı aynı zamanda birim testle tam kapsanan kısmıdır.

**M6 — kusur (`fault.py`).** KTK m.84 asli kusur hâllerini **kapalı liste**
olarak uygular (`config/rules/fault_ktk84.yaml`, a–l 12 bent):

| Sonuç | Anlamı |
|---|---|
| `ASLI` | ihlal m.84 listesinde eşleşti |
| `TALI` | ihlal tespit edildi ama listede yok |
| `TESPIT_EDILEMEDI` | o tarafta ihlal çıkarılamadı — **"kusursuz" değil** |

**Yüzde üretmez.** KTY m.156/3 uyarınca tutanağı düzenleyen görevli bile kusur
oranı belirtmez; oranı TRAMER belirler. Motor tutanak mantığını izler: hangi
madde, asli mi tali mi.

Kapalı liste disiplininin somut sonucu — M4'ün ürettiği 6 koddan yalnızca ikisi
asli kusur doğurur:

| M4 kodu | Sınıf | Neden |
|---|---|---|
| `KIRMIZI_ISIK` | **ASLI** 84/a | listede |
| `TERS_YON` | **ASLI** 84/b | listede |
| `HATALI_PARK` | TALI | 84/k yalnızca yerleşim **dışı** karayolu içindir |
| `SERIT_IHLALI` | TALI | 84/g "şeride tecavüz"tür, şerit *değişimi* değil |
| `HIZ_IHLALI` | TALI | m.84'te sayılmaz |
| `TAKIP_MESAFESI` | TALI | m.84'te sayılmaz |

**M7 — ceza (`penalty.py`).** Tutarlar tarih damgalı tablolardan gelir
(`config/penalties/<geçerlilik-tarihi>.yaml`). Kurallar:

- `penalty.py` içinde **tek bir tutar yoktur** — `grep -nE '[0-9]{4,}' adgs/penalty.py` boş döner, test bunu zorlar
- Tablo **olayın** tarihine göre seçilir; arşiv videosu bugünün tablosuyla hesaplanmaz
- `--tarih` verilmezse ceza **hesaplanmaz** (bugüne düşülmez)
- Uygun tablo yoksa **eski tabloya sessizce düşülmez**, hesaplama yapılmaz
- Tabloda karşılığı olmayan ihlal için tutar **uydurulmaz**
- Her çıktı hangi tablodan hesaplandığını taşır
- Erken ödeme indirimi hesaplanmaz (tebliğ tarihine bağlı, video bilemez)

> **Her iki tablo da henüz doğrulanmamıştır.** `meta.dogrulama_tarihi` boş
> olduğu sürece motor her çıktıya bunu bildiren bir uyarı ekler. KTK m.84 metni
> iki bağımsız kaynaktan karşılaştırılarak aktarıldı; operasyonel kullanımdan
> önce mevzuat.gov.tr konsolide metniyle, ceza tutarları da trafik.gov.tr resmî
> rehberiyle karşılaştırılıp meta alanları doldurulmalıdır.

### Faz 6 — araç hasar değerlendirmesi

```bash
python training/prep_cardd.py                    # CarDD -> YOLO
python training/train_cardd.py                   # amp KAPALI, batch 2
python training/cardd_cozunurluk_etkisi.py       # FAZ 6 BULGUSU (aşağıya bkz.)
adgs run kavsak.mp4 --detect accident --camera arnavutkoy_kavsak_01 --out runs/f6
```

Kaza olaylarının her tarafına `hasar` alanı eklenir:

```json
{"guvenilir": true, "tipler": ["crack", "scratch"],
 "etiketler": ["Catlak", "Cizik"], "bolge": "on",
 "siddet": "ORTA", "alan_orani": 0.061, "keyframe": 412}
```

**Kabul kriteri ölçüldü:** `mAP@0.5 = 0.509` (hedef ≥ 0.40), 13 epoch,
`yolo26s`, 561 görüntülük val kümesi.

Sınıf bazında tablo, modülün asıl sınırını gösteriyor:

| sınıf | AP@0.5:0.95 |
|---|---|
| tire flat | 0.705 |
| glass shatter | 0.674 |
| lamp broken | 0.386 |
| scratch | 0.167 |
| dent | 0.147 |
| **crack** | **0.025** |

Büyük ve yapısal hasar tutuyor; ince dokulu hasar (çatlak, çizik, göçük)
çalışmıyor. Bu bir başarısızlık değil, **bulgudur** — ve çözünürlük kapısının
dayanağıdır.

**Çözünürlük kapısı — bu modülün en önemli davranışı.** CarDD yakın çekim
fotoğraflardan oluşur (araç kareyi doldurur). Sabit CCTV ise aracı 60×40
piksellik bir kutuda görür. O boyutta "çizik mi göçük mü" sorusunun görüntüde
cevabı yoktur — ama model yine de bir sınıf üretir.

`MIN_KENAR_PIKSEL` **tahmin değil ölçümdür**
(`python training/cardd_cozunurluk_etkisi.py`):

| uzun kenar | mAP@0.5 | referansa göre düşüş |
|---|---|---|
| 640 px | 0.514 | — |
| 384 px | 0.465 | %9.6 |
| **256 px** | **0.371** | **%27.8** ← kapı burada |
| 192 px | 0.270 | %47.5 |
| 128 px | 0.149 | %71.0 |
| 96 px | 0.104 | %79.7 |

Tam tablo (64 px dahil): `runs/cardd_olcek/sonuc.json`

Sınıf bazında düşüş, kapının **neden** orada olduğunu açıklıyor (640 → 256 px):

| sınıf | 640 px | 256 px | düşüş |
|---|---|---|---|
| tire flat | 0.708 | 0.646 | %9 |
| glass shatter | 0.670 | 0.566 | %15 |
| dent | 0.148 | 0.076 | %48 |
| lamp broken | 0.405 | 0.178 | %56 |
| scratch | 0.166 | 0.073 | %56 |
| crack | 0.024 | 0.007 | %72 |

Büyük ve yapısal hasar küçültmeye dayanıyor; ince doku dayanmıyor. **Pratik
sonuç:** CCTV ölçeğinde yalnızca "lastik patlak" ve "cam kırılması" güvenilir;
çizik/çatlak/göçük ayrımı için yakın çekim veya olay yeri fotoğrafı gerekir.

İlk yazımda eşik 128 idi ve bu bir tahmindi; ölçüm onu çürüttü. Araç kırpmasının
uzun kenarı eşiğin altındaysa **sınıf bazlı çıktı üretilmez**: `guvenilir: false`
ve sebep yazılır. Yakın çekim / olay yeri fotoğrafı beslendiğinde eşik
gevşetilebilir (`degerlendir(..., min_kenar=...)`).

**Bölge (ön/arka/yan) aracın yönünü gerektirir.** Yön, izin hareket
vektöründen çıkarılır. Duran araç için yön bilinemez ve `bolge: null` kalır —
"ön" varsayılmaz.

Altyapı hasarının şiddeti burada **yeniden hesaplanmaz**; M5 onu zaten üretir.
İkinci bir uygulama, iki farklı cevap veren iki gerçek kaynağı olurdu.

> **Veri kümesi uyarısı.** Kullanılan HuggingFace aynası 2816 görüntü içerir
> (CarDD makalesi 4000 der) ve resmî train/val/test bölünmesini **taşımaz**.
> `prep_cardd.py` dosya adı hash'inden deterministik bir bölünme üretir. Bu
> nedenle ölçülen mAP makaleyle **doğrudan karşılaştırılamaz**.

### Faz 7 — depolama, API, arayüz, iş emri

```bash
pip install -e ".[api]"
adgs serve                      # http://127.0.0.1:8000/  (API dokümanı: /docs)
adgs store runs/f5/rapor.json --video kavsak.mp4 --camera arnavutkoy_kavsak_01 \
           --tarih 2026-07-15
adgs workorder runs/f5/rapor.json --event altyapi_0007 --out is_emri.pdf
adgs purge                      # KVKK: saklama süresi dolanları imha et
```

**M10 — SQLite, dört tablo:** `videos` · `events` · `parties` · `violations`.
Şema PostgreSQL'e taşınabilir yazıldı; geçiş bağlantı dizesi değişikliğidir.
Çoklu kamera için `videos.camera_id` ve `events.gps_*` baştan var — ayrı
`cameras` tablosu, kullanıcı yönetimi ve tenant izolasyonu **bilinçli olarak
yazılmadı**.

**API uçları:**

| Uç | İş |
|---|---|
| `POST /videos` | **yalnızca dosya** — kalanı videodan türetilir |
| `GET /videos/{id}/status` | `BEKLIYOR → ISLENIYOR → TAMAM\|HATA` |
| `GET /events?tip=&tarih=&camera_id=` | filtreli olay listesi |
| `GET /events/{id}` | tek olay (taraflar, kusur, ceza) |
| `GET /events/{id}/is-emri.pdf` | Fen İşleri PDF iş emri |
| `POST /kvkk/purge` | saklama süresi dolanları imha |
| `GET /` | tek sayfa arayüz |

**Yükleme yalnızca videoyu ister.** Kaynak profili, çalışacak dedektörler,
kamera ve çekim tarihi kullanıcıya sorulmaz; `adgs/probe.py` bunları videonun
kendisinden türetir:

| Alan | Nereden | Nasıl |
|---|---|---|
| `profile` | **ölçüm** | yarım saniyedeki kamera kayması (faz korelasyonu), kare genişliğinin oranı olarak |
| `detect` | profilden | sabit → `accident,ihlal` · araca monteli → `roaddamage` |
| `camera` | dosya adı | `config/cameras/` altındaki **tanımlı** kimlikle birebir eşleşme |
| `tarih` | dosya adı | `2026-07-15` · `20260715` · `15.07.2026` |

Kamera kayması ölçüldü (bu depodaki videolar, medyan): sabit sahne **%0.004**,
kaydırılan kamera **%19.3** — iki sınıf arasında ~50× fark var, eşiğin tam yeri
kritik değil. Bitişik karelerdeki *piksel yoğunluğu farkı* bu iş için
yetersizdi: 30 fps'te sahne birkaç piksel kayar ve düz yüzeylerde (asfalt,
gökyüzü) fark eşiğin altında kalır — dashcam görüntüsü "sabit" ölçülüyordu.

**Türetilemeyen alan boş bırakılır, tahmin edilmez.** İkisi özellikle önemli:

- **Tarih** bugüne düşmez. Arşiv videosu bugünün ceza tablosuyla hesaplanırsa
  sessizce yanlış tutar üretir.
- **Kamera** çözünürlükten tahmin edilmez. Yanlış kamera = yanlış dur çizgisi =
  bir vatandaş adına yanlış ihlal kaydı. Bu projede precision recall'dan önce
  gelir.

Boş kalan alan ilgili modülün kendi reddetme mekanizmasını çalıştırır. Her
kararın gerekçesi yanıtın `turetilen.gerekce` alanında döner ve arayüzde
yüklemenin hemen altında gösterilir — türetilmiş bir değer kullanıcıdan
saklanmaz.

İşleme **senkron değildir**: analiz dakikalar sürdüğü için HTTP isteği
bekletilmez, istemci durumu sorgular. Boru hattı burada yeniden yazılmaz —
`cli.run` çağrılır, ürettiği `rapor.json` veritabanına alınır.

**PDF iş emri.** Yeni bağımlılık eklenmedi: reportlab/fpdf2'nin gömülü fontları
Latin-1'dir ve `ş/ğ/İ` basamaz — resmî bir iş emrinde bozuk metin kabul
edilemez. matplotlib zaten Ultralytics ile kurulu geliyor ve DejaVu fontunu
paket içinde taşıyor. Şiddet/öncelik M5'in notlarından **okunur, yeniden
hesaplanmaz**.

**KVKK — üç mekanizma, üçü de test edilmiş:**

- **Saklama süresi:** `config/pipeline.yaml` → `saklama_gun` (varsayılan 30).
  `adgs purge` süresi dolan kayıtları **ve klip dosyalarını** siler. Sadece
  satırı silmek diskte kişisel veri bırakırdı.
- **Bulanıklaştırma:** `plaka_bulaniklastir` / `yuz_bulaniklastir` varsayılan
  **açık**. İşaretli video ve kliplerde yaya kutusunun üstü (baş) ve araç
  kutusunun alt-ortası (plaka) Gaussian blur ile kapatılır.
  > Bölgeler **geometrik yaklaşımdır** — ayrı plaka/yüz dedektörü yok. Fazladan
  > alan bulanıklaştırmak, eksik bulanıklaştırmaktan iyidir; gerçek dedektör
  > eklenirse yalnızca `render.kvkk_bolgeleri()` değişir.
- **Cascade:** SQLite'ta `foreign_keys` PRAGMA'sı varsayılan **kapalıdır**;
  açılmazsa imha sırasında taraf/ihlal satırları öksüz kalırdı. Açık olduğu
  testle doğrulanıyor.

### Faz 8 — ileri özellikler

```bash
adgs hotspot --db data/adgs.db --yaricap 50      # kara nokta analizi
adgs track rtsp://kamera/stream --sure 60        # canlı akış, 60 sn
```

**Kara nokta analizi (M8'in planlama çıktısı).** Belediyenin bu sistemden
çıkarabileceği en değerli yatırım girdisi: hangi noktada, ne sıklıkta, ne
ağırlıkta olay oluyor. Ağırlık `KAZA 5 > IHLAL 2 > ALTYAPI 1`, yol hasarında
şiddet çarpanıyla.

**İki gruplama birbirine karıştırılmaz** — bu kasıtlıdır:

| Kaynak | Gruplama |
|---|---|
| GPS var (araca monteli) | haversine ile coğrafi kümeleme, yarıçap ayarlanabilir |
| GPS yok (sabit CCTV) | kamera bazlı gruplama |

Sabit kameranın olaylarına uydurma koordinat verip haritaya basmak, var
olmayan bir kara nokta üretirdi. Ağırlıklar **göreli sıralama ölçeğidir**,
mutlak risk skoru değil — gerçek kara nokta analizi yaralanma ve maddi hasar
verisi gerektirir, bu sistemde o veri yok.

**Canlı akış (RTSP).** `adgs track` `rtsp://` / `rtmp://` kaynaklarını okur.
`--sure` **zorunludur**: akışın sonu yoktur, sınır verilmezse döngü hiç bitmez.
Canlı akışta işaretli video üretilmez (akış geri sarılamaz), izler yazılır.
`adgs run` dosya gerektirir — klip kesme ve işaretleme kaynağı yeniden okur.

**Çoklu kamera.** `GET /kameralar` kamera başına video/olay özeti verir;
`videos.camera_id` ve `events.gps_*` şemada baştan vardı.

**Bilinçli olarak yapılMAyanlar:**

| Madde | Neden |
|---|---|
| PostgreSQL + PostGIS geçişi | SQLite şeması zaten taşınabilir yazıldı; geçiş bağlantı dizesi değişikliği. Gerçek ikinci kamera gelmeden migration altyapısı kurmak erken. |
| Araç içi davranış (kemer/telefon/kask) | Plan §5'te fizibilitesi düşük işaretlenmişti ve Faz 4 bunu doğruladı: sabit CCTV 6-10 m yükseklikten bakar, ön cam yansıması nedeniyle sürücü gövdesi çoğu karede görünmez. Araca monteli iç kamera olmadan anlamlı değil. |

## Kabul kriterleri — ölçüm

Plandaki her kabul kriteri tek komutla ölçülür:

```bash
adgs kabul                # tablo
adgs kabul --json         # rapora gömmek için
```

Her satır üç durumdan birini alır. **"Kod yazıldı" bir durum değildir.**

| Durum | Anlamı |
|---|---|
| `GECTI` | ölçüldü ve hedefi tutturdu |
| `KALDI` | **ölçüldü** ve hedefin altında kaldı |
| `OLCULMEDI` | ölçüm koşulu yok (eğitilmiş model veya etiketli veri gerekiyor) |

`OLCULMEDI` bir başarısızlık değildir ama başarı olarak da gösterilemez; her
satır nedenini ve o nedeni kaldırmak için gerekeni yazar. Model mAP'leri
`adgs eval`'in yazdığı `runs/<ad>/kabul.json`'dan okunur — tablo dakikalarca
süren val koşusunu tekrarlamaz.

Çıkış kodu yalnızca `KALDI` varsa sıfırdan farklıdır: ölçülememiş bir kriter
hata değildir, ölçülüp hedefin altında kalan kriter hatadır.

## Test

```bash
pytest -q
pytest tests/test_fault.py tests/test_penalty.py -v   # Faz 5 kabul kriteri
grep -nE '[0-9]{4,}' adgs/penalty.py                  # BOŞ dönmeli
adgs kabul                                            # tüm kabul kriterleri
```

## Lisans notu

Detektör olarak Ultralytics YOLO26 kullanılmaktadır. Ultralytics **AGPL-3.0**
lisanslıdır; fine-tune edilmiş ağırlıklar ve doğrudan bağlanan kod bu lisansı devralır.
Kapalı kaynak bir teslimat gerekirse detektör Apache-2.0 lisanslı bir muadille
(RF-DETR, D-FINE, RT-DETRv2) değiştirilmelidir — mimari değişmez, yalnızca
`config/pipeline.yaml` içindeki detektör seçimi ve ağırlıklar değişir.

### Veri kümesi atfı (zorunlu)

Yol hasarı modeli **RDD2022** üzerinde ince ayar yapılarak eğitilmiştir.
RDD2022 **CC BY-SA 4.0** lisanslıdır: atıf zorunludur ve **türev çalışmalar
(eğitilen ağırlıklar dâhil) aynı lisansla paylaşılmalıdır.**

> Arya, D., Maeda, H., Ghosh, S. K., Toshniwal, D., Sekimoto, Y. (2022).
> *RDD2022: A multi-national image dataset for automatic road damage detection.*
> arXiv:2209.08538 — https://doi.org/10.6084/m9.figshare.21431547

Not: Bu yükümlülük Ultralytics AGPL-3.0'a **eklenir**, onun yerine geçmez.
Detektör Apache-2.0 muadiliyle değiştirilse bile RDD2022 ile eğitilmiş
ağırlıklar CC BY-SA 4.0 taşımaya devam eder.

### ⚠️ CarDD — ticari kullanıma kapalı (Faz 6)

Araç hasar modeli (M8) **CarDD** üzerinde eğitilir. CarDD **ticari olmayan
araştırma ve eğitim** kullanımıyla sınırlıdır ve görüntüleri Flickr /
Shutterstock lisans şartlarına tabidir.

Bu, planın Bosch trafik ışığı veri kümesi için koyduğu kısıtın aynısıdır ve
**kodla çözülemez**: bir model ağırlığı, eğitildiği verinin lisansını taşır.

| Kullanım | Durum |
|---|---|
| Staj / akademik / araştırma | Uygun |
| Belediyeye operasyonel teslimat | **Uygun değil** |

Operasyonel bir teslimat gerekirse M8 ya kapatılmalı, ya ticari kullanıma açık
bir veri kümesiyle yeniden eğitilmeli, ya da CarDD sahiplerinden ayrı izin
alınmalıdır. Sistemin geri kalanı M8 olmadan çalışır — kaza olayları `hasar`
alanı `null` ile üretilir.

> Wang, X., Li, W., Wu, Z. (2023). *CarDD: A New Dataset for Vision-Based Car
> Damage Detection.* IEEE T-ITS — https://cardd-ustc.github.io/

## Plan

Tam mimari plan, modül kırılımı ve yol haritası için proje planına bakınız.
