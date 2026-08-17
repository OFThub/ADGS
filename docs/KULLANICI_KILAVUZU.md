# ADGS — Kullanıcı Kılavuzu

Bu kılavuz sistemi **çalıştıran** kişi içindir: kurulum, günlük kullanım, yapılandırma,
hata mesajlarının anlamı ve sorun giderme. Mimari ve API ayrıntısı için
[API_VE_MIMARI.md](API_VE_MIMARI.md), örnek iş akışları için
[KULLANIM_SENARYOLARI.md](KULLANIM_SENARYOLARI.md).

> **Sistem bağlayıcı bir tespit üretmez.** Her KAZA ve İHLAL çıktısı kaldırılamayan bir
> ön değerlendirme notu taşır. Ayrıntı: [README](../README.md).

---

## 1. Kurulum

### 1.1 Gereksinimler

| Bileşen | Sürüm / not |
|---|---|
| Python | **3.12** (3.13'e kadar çalışır, 3.14 **çalışmaz** — aşağıya bkz.) |
| GPU | NVIDIA + CUDA önerilir; CPU'da çalışır ama 10–20× yavaştır |
| Disk | Model ağırlıkları + video için birkaç GB |
| İşletim sistemi | Windows / Linux (geliştirme Windows 11 üzerinde yapıldı) |

### 1.2 Adımlar

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -e ".[dev]"         # temel + test
pip install -e ".[api,dev]"     # web arayüzü de isteniyorsa

adgs doctor
```

`adgs doctor` GPU/CUDA, detektör, yapılandırma ve şema kontrollerini çalıştırır.
**Kuruluma buradan devam etmeden önce bu komutun çıktısını okuyun.**

### 1.3 ⚠️ Python 3.14 tuzağı — en sık karşılaşılan kurulum hatası

PyTorch'un CUDA wheel'leri `cp38`–`cp313` içindir; **`cp314` yoktur.** Sistem Python'u
3.14 ise `import torch` çalışır ama **CPU sürümü** gelir ve `pip install --upgrade torch`
"zaten karşılanmış" deyip hiçbir şey yapmaz.

**Belirti:** her şey çalışır, sadece 10–20× yavaşlar. 5 dakikalık video 25 dakikada
işlenir, eğitim ilerlemez.

```bash
.venv/Scripts/python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available())"
# beklenen : 2.11.0+cu128 True
# yanlışsa : 2.13.0+cpu   False   -> sistem Python'unu kullanıyorsunuz
```

Tüm komutları `.venv\Scripts\python.exe -m adgs.cli ...` ile veya ortamı aktive ederek
çalıştırın — `adgs serve` dahil.

---

## 2. İlk çalıştırma — 2 komut

```bash
pip install -e ".[api,dev]"
adgs serve                       # http://127.0.0.1:8000/
```

Tarayıcıda:

1. **Video yükle** — örn. `data/testset/demo_pan.mp4` → *Yükle ve işle*
2. Durum `BEKLIYOR → ISLENIYOR → TAMAM` diye kendiliğinden ilerler (~20 sn)
3. **İşaretlenmiş video** bölümünde sonuç oynar; her araç kalıcı takip numarasıyla
   çerçeveli, altta kaldırılamayan *ÖN DEĞERLENDİRME* filigranı
4. **Olaylar** tablosunda tespitler; satıra tıklayınca gerekçe notları açılır, sağdan
   **klip** ve **PDF iş emri** indirilir
5. **Kara nokta analizi** bölümünde konum/kamera bazlı risk sıralaması

Sadece takip görüntüsü yeterliyse (veritabanı gerekmez):

```bash
adgs track data/testset/demo_pan.mp4 --out runs/demo
# runs/demo/demo_pan_tracked.mp4
```

> `demo_sentetik` kamerasının şerit yönleri **uydurmadır**. Sunumda çıkan `TERS_YON`
> tespitleri kod yolunun çalıştığını gösterir; gerçek ihlal değildir.

---

## 3. Komut referansı

Tüm komutlar: `adgs <komut> --help`

| Komut | İş | Zorunlu argüman |
|---|---|---|
| `doctor` | Ortam ve yapılandırma kontrolü | — |
| `run` | Videoyu işle (hasar / kaza / ihlal) | `video` |
| `track` | Tespit + takip, kalıcı ID'ler | `video` (veya `rtsp://`) |
| `calibrate` | Kamera kalibrasyonunu doğrula | `--camera` |
| `eval` | Model mAP@0.5 ölçümü | — |
| `kabul` | Tüm kabul kriterlerini ölç ve tabloya bas | — |
| `explain` | Bir olayın kusur/ceza gerekçesi | `rapor` |
| `redetect` | Kayıtlı izlerden tespiti yeniden çalıştır | `izler.json`, `--video` |
| `rerender` | `rapor.json`'dan işaretli videoyu yeniden çiz | `rapor`, `--video` |
| `serve` | Web arayüzü + API | — |
| `store` | `rapor.json`'u veritabanına al | `rapor`, `--video` |
| `workorder` | Fen İşleri PDF iş emri | `rapor`, `--event` |
| `hotspot` | Kara nokta analizi | — |
| `purge` | KVKK: saklama süresi dolanları imha et | — |

### 3.1 `adgs run` — ana analiz komutu

```bash
adgs run <video> [seçenekler]
```

| Seçenek | Varsayılan | Açıklama |
|---|---|---|
| `--detect` | `roaddamage` | `roaddamage` · `accident` · `ihlal` · tek tip (`redlight,wrongway`) |
| `--profile` | `vehicle_mounted` | `cctv_fixed` · `vehicle_mounted` |
| `--camera` | yok | `config/cameras/<id>.yaml` kimliği |
| `--tarih` | yok | `YYYY-AA-GG` — **ceza hesabı için zorunlu** |
| `--out` | `runs/f1` | Çıktı dizini |
| `--conf` | `0.35` | Detektör güven eşiği |
| `--model` | `runs/rdd2022/yolo26s/weights/best.pt` | Yol hasarı modeli |
| `--track-model` | `yolo26s.pt` | Takip modeli |
| `--damage-model` | `runs/cardd/yolo26s/weights/best.pt` | Araç hasarı modeli (M8) |
| `--rules` | `config/rules/violations.yaml` | İhlal eşikleri |

**Örnekler**

```bash
# Yol hasarı (araca monteli)
adgs run yol.mp4 --profile vehicle_mounted --out runs/f1

# Kaza (sabit CCTV)
adgs run kavsak.mp4 --detect accident --profile cctv_fixed \
        --camera arnavutkoy_kavsak_01 --out runs/f3

# İhlal + kusur + ceza
adgs run kavsak.mp4 --detect ihlal --camera arnavutkoy_kavsak_01 \
        --tarih 2026-07-15 --out runs/f5
```

### 3.2 Çıktı dizininin içeriği

```
runs/f5/
├─ rapor.json              # tüm olaylar, taraflar, kusur/ceza, çalışamayan modüller
├─ izler.json              # M2 çıktısı — redetect bunu okur
├─ <video>_isaretli.mp4    # işaretli video (KVKK bulanıklaştırma uygulanmış)
└─ klipler/
   ├─ kaza_0001.mp4
   └─ ihlal_0042.mp4
```

---

## 4. Kamera yapılandırması

Sabit CCTV senaryolarında **her şey buradan başlar**. Şablon:
`config/cameras/arnavutkoy_kavsak_01.yaml`

### 4.1 Alanlar ve hangi modülü açtıkları

| Alan | Açtığı yetenek | Yoksa ne olur |
|---|---|---|
| `homografi.piksel` / `.dunya` | Dünya koordinatı, hız, mesafe | Hız/mesafe **üretilmez** |
| `dogrulama` | Kalibrasyonun doğrulanması | `speed`, `tailgating` **reddeder** |
| `maks_hata_yuzde` | Doğrulama eşiği (varsayılan 10) | — |
| `dur_cizgisi` + `isik_roi` | `redlight` | `KIRMIZI_ISIK` çalışmaz |
| `seritler[].yon` | `wrongway` | `TERS_YON` çalışmaz |
| `seritler[].poligon` (≥2) | `lane` | `SERIT_IHLALI` çalışmaz |
| `yasak_park[].poligon` | `parking` | `HATALI_PARK` çalışmaz |
| `hiz_limiti_kmh` | `speed` | `HIZ_IHLALI` çalışmaz |
| `roi` | İlgi alanı kırpması | Tüm kare işlenir |

### 4.2 Kalibrasyon doğrulama

```bash
adgs calibrate --camera arnavutkoy_kavsak_01 --verify
# geçmezse çıkış kodu 1
```

Doğrulama noktası **bağımsız bir ölçüm olmalıdır**. Homografi fitine kullanılan
noktalarla doğrulama yapılırsa hata ~0 çıkar ve test hiçbir şey kanıtlamaz.
(`demo_sentetik.yaml` tam olarak bu kusuru taşır — sadece kod yolunu çalıştırmak
içindir.)

Kalibrasyon doğrulanmazsa dünya koordinatı ve hız **üretilmez**, işaretli videoya
`KALIBRASYON YOK` uyarısı basılır. Bu kasıtlıdır: %20 hatalı bir homografi 50 km/s'lik
aracı 60 km/s gösterir ve bu sayı ceza doğuran bir çıktı olarak dolaşıma girer.

### 4.3 Sahada ölçüm önerisi

1. Kamera görüntüsünde yerdeki **dört tanınabilir noktayı** işaretleyin (kaldırım köşesi,
   şerit çizgisi başı/sonu, rögar kapağı).
2. Bu noktalar arasındaki gerçek mesafeleri **metreyle ölçün** → `homografi.dunya`.
3. Fite **girmeyen** ayrı bir çift nokta seçin, mesafesini ölçün → `dogrulama`.
4. `adgs calibrate --verify` çalıştırın; `maks_hata_yuzde` üstündeyse noktaları gözden
   geçirin.

---

## 5. Yapılandırma dosyaları

### 5.1 `config/pipeline.yaml`

```yaml
source_profile: vehicle_mounted   # cctv_fixed | vehicle_mounted
sample_fps: 2.0                   # yol hasarında 2 fps yeterli
target_long_edge: 1280
on_isleme: false                  # KAPALI — önce ham kareyle mAP ölç
saklama_gun: 30                   # KVKK
plaka_bulaniklastir: true
yuz_bulaniklastir: true
```

> `on_isleme` varsayılan kapalıdır. Kör ön işleme çoğu zaman doğruluğu **düşürür**;
> açmadan önce ham kareyle mAP ölçün.

### 5.2 `config/rules/violations.yaml`

`aktif` bloğu her dedektörü ayrı ayrı açar/kapatır. Eşik blokları rapora yazılan ihlal
koduyla anahtarlanır (`KIRMIZI_ISIK`, `TERS_YON`, …) — raporda gördüğünüz kodu doğrudan
burada arayabilirsiniz.

| Anahtar | Varsayılan | Kritik eşik |
|---|---|---|
| `redlight` | açık | `isik_min_piksel: 20`, `devam_pencere_kare: 8` |
| `wrongway` | açık | `aci_esik_derece: 120`, `min_ardisik_kare: 8` |
| `parking` | açık | `sure_sn: 60` — düşürmek kırmızıda bekleyen her aracı ihlal yapar |
| `tailgating` | açık | `guvenli_sn: 2.0`, `min_hiz_kmh: 20` |
| `speed` | açık | `tolerans_yuzde: 10` — müsamaha değil, ölçüm belirsizliği payı |
| `lane` | **kapalı** | açılırsa çıktı ihlal değil, inceleme adayıdır |

**Bir tipi açmak geometriyi tanımladığınız anlamına gelmez.** Geometri eksikse modül
tahmin üretmez; çalışmayı reddeder ve sebebini uyarı olarak basar.

### 5.3 `config/penalties/<tarih>.yaml`

**Dosya adı = geçerlilik başlangıç tarihidir.** Motor, olayın tarihinde geçerli tabloyu
seçer.

```yaml
meta:
  gecerlilik_baslangic: "2026-02-27"
  kaynak: "EGM Trafik Başkanlığı — Trafik Ceza Rehberi"
  dogrulama_tarihi: null      # !!! doldurulmalı
  dogrulayan: null
cezalar:
  - ihlal_kodu: KIRMIZI_ISIK
    ktk_madde: "47/1-c"
    ...
```

> **Tablo henüz doğrulanmamıştır.** `dogrulama_tarihi` boş olduğu sürece motor her
> çıktıya bunu bildiren uyarı ekler. Operasyonel kullanımdan önce trafik.gov.tr resmî
> rehberiyle karşılaştırıp `meta` alanlarını doldurun.

Yeni tablo eklemek: `config/penalties/2027-01-01.yaml` dosyasını oluşturun. Kod
değişikliği gerekmez — `penalty.py` içinde **tek bir tutar yoktur**
(`grep -nE '[0-9]{4,}' adgs/penalty.py` boş döner, test bunu zorlar).

### 5.4 `config/rules/fault_ktk84.yaml`

KTK m.84 asli kusur hâllerinin **kapalı listesi** (a–l, 12 bent). M6 yalnızca bu listeyle
eşleştirme yapar; listede olmayan ihlal `TALI` olur.

---

## 6. Web arayüzü

`adgs serve --host 127.0.0.1 --port 8000`

| Bölüm | Ne yapar |
|---|---|
| Yükleme | **Yalnızca video** ister; profil/dedektör/kamera/tarih videodan türetilir |
| Türetilen değerler | Her kararın **gerekçesi** yüklemenin hemen altında gösterilir |
| Durum | `BEKLIYOR → ISLENIYOR → TAMAM \| HATA`, kendiliğinden yoklanır |
| İşaretlenmiş video | Takip ID'leri + ön değerlendirme filigranı |
| Olaylar tablosu | Tip, video, zaman, güven; satıra tıklayınca notlar açılır |
| İndirmeler | Olay klibi, PDF iş emri |
| Kara nokta | Konum/kamera bazlı risk sıralaması |

**Dosya adını doğru vermek işinizi kolaylaştırır.** Kamera ve tarih dosya adından
türetilir:

```
arnavutkoy_kavsak_01_2026-07-15.mp4     → kamera + tarih türetilir
kayit1.mp4                              → ikisi de boş kalır
```

Boş kalan alan ilgili modülün reddetme mekanizmasını çalıştırır — bu bir hata değil,
tasarımdır.

API ayrıntısı: [API_VE_MIMARI.md](API_VE_MIMARI.md) · Canlı şema: `http://127.0.0.1:8000/docs`

---

## 7. Model eğitimi

### 7.1 Yol hasarı (RDD2022)

```bash
python training/prep_rdd2022.py --src data/raw/RDD2022_ext --out data/rdd_yolo
python training/train_rdd2022.py            # amp KAPALI, batch 4
adgs eval --model runs/rdd2022/yolo26s/weights/best.pt --hedef 0.45
```

> **`--amp` açmayın.** RTX 3050 4 GB üzerinde fp16, `cls_loss` taşmasıyla NaN üretir ve
> eğitim çöpe gider. `--amp` kapalıyken aynı taşma sonlu kalır ve gradyan kırpması
> sayesinde zarar vermez.

Eğitim betiği ilk NaN'da **anında durur** (`adgs/trainguard.py`) — sessiz çöp koşu
üretmez. Muhafız yalnızca anlık kaybı değil **koşan ortalama kaybı** (`trainer.tloss`)
da izler; tek karelik NaN'ın ortalamayı zehirleyip fark edilmeden ilerlemesi böyle
engellenir.

### 7.2 Araç hasarı (CarDD)

```bash
python training/prep_cardd.py
python training/train_cardd.py                   # amp KAPALI, batch 2
python training/cardd_cozunurluk_etkisi.py       # çözünürlük kapısı ölçümü
```

**Ölçülen sonuç:** `mAP@0.5 = 0.509` (hedef ≥ 0.40), 13 epoch, `yolo26s`, 561 görüntülük
val kümesi.

> **CarDD ticari kullanıma kapalıdır.** Staj/akademik kullanım uygundur; belediyeye
> operasyonel teslimatta M8 kapatılmalı veya ticari kullanıma açık veriyle yeniden
> eğitilmelidir.

---

## 8. KVKK işlemleri

| İşlem | Komut / ayar |
|---|---|
| Saklama süresini belirle | `config/pipeline.yaml` → `saklama_gun` |
| Süresi dolanları imha et | `adgs purge` veya `POST /kvkk/purge` |
| Bulanıklaştırmayı kapat (önerilmez) | `plaka_bulaniklastir: false` |

`adgs purge` hem veritabanı satırlarını **hem klip dosyalarını** siler. Sadece satırı
silmek diskte kişisel veri bırakırdı.

**Kurulum sırasında doğrulanması gerekenler** (kod bunları kontrol edemez):

- Aydınlatma metni bu işleme faaliyetini kapsıyor mu?
- VERBİS kaydı güncel mi?
- Erişim kontrolü ve erişim logu var mı?
- Ham video ve model ağırlıkları repoya girmiyor (`data/` gitignore'da)

---

## 9. Hata ve uyarı mesajları

| Mesaj | Anlamı | Ne yapmalı |
|---|---|---|
| `KALIBRASYON YOK` (videoda) | Homografi doğrulanmadı | `adgs calibrate --verify`, kamera YAML'ını ölç |
| `calisamayan_moduller` (rapor.json) | Geometri eksik olduğu için modül çalışmadı | İlgili kamera alanını doldur (§4.1) |
| `ceza tablosu doğrulanmamıştır` | `meta.dogrulama_tarihi` boş | Resmî kaynakla karşılaştırıp doldur |
| `tarih verilmedi, ceza hesaplanmadı` | `--tarih` eksik | `--tarih YYYY-AA-GG` ver |
| `TESPIT_EDILEMEDI` (kusur) | O tarafta ihlal çıkarılamadı | **"Kusursuz" demek değildir** |
| `guvenilir: false` (hasar) | Araç kırpması çözünürlük kapısının altında | Yakın çekim/olay yeri fotoğrafı gerekir |
| `durum: HATA` (API) | Arka plan analizi çöktü | `videos.hata` alanını ve sunucu logunu oku |

---

## 10. Sorun giderme

**Analiz çok yavaş (5 dk video → 25 dk)**
Python 3.14 / CPU torch. §1.3'e bakın.

**Hiç olay çıkmıyor**
`rapor.json` → `calisamayan_moduller` alanını okuyun. Büyük olasılıkla modül
çalışmadı ("olay yok" ile "bakılmadı" farklı şeylerdir).

**Gece görüntüsünde araç görünmüyor**
Sistem HSV parlaklık medyanını ölçer; <100/255 ise güveni 0.15'e düşürür. Ölçüm: aynı
sahnede `conf 0.35` → 0 araç, `conf 0.15` → 9 araç. Elle zorlamak için `--conf 0.15`.

**Montaj/derleme videoda sahte kaza**
Sahne kesmesi tespiti HSV histogram korelasyonu <0.35 olduğunda kesme sayar ve o çifti
eler. `adgs redetect ... --tani` ile kaç kesmenin elendiğini görün.

**Eşik ayarı yapacağım ama takip 30 dakika sürüyor**
Tekrarlamayın:
```bash
adgs redetect runs/<koşu>/izler.json --video kayit.mp4 --tani   # ~2 sn
adgs rerender runs/<koşu>/rapor.json --video kayit.mp4
```

**Eğitim NaN veriyor**
`--amp` kapalı olduğundan emin olun, batch'i düşürün. Muhafız zaten ilk NaN'da durur.

**PDF'te Türkçe karakterler bozuk**
Olmamalı — iş emri PDF'i matplotlib + DejaVu Sans ile üretilir (reportlab/fpdf2'nin
gömülü fontları Latin-1'dir ve `ş/ğ/İ` basamaz). Bozuksa matplotlib kurulumunu kontrol
edin.

---

## 11. Doğrulama — sistem çalışıyor mu?

```bash
pytest -q                                             # 365 test
pytest tests/test_fault.py tests/test_penalty.py -v   # Faz 5 kabul kriteri
grep -nE '[0-9]{4,}' adgs/penalty.py                  # BOŞ dönmeli
adgs kabul                                            # tüm kabul kriterleri
adgs doctor                                           # ortam
```

`adgs kabul` çıkış kodu yalnızca `KALDI` (ölçülüp hedefin altında kalan) satır varsa
sıfırdan farklıdır; `OLCULMEDI` hata sayılmaz.

---

## 12. Sık sorulanlar

**Sistem otomatik ceza kesebilir mi?**
Hayır. Belediyenin KTK ihlalleri için idari para cezası kesme yetkisi yoktur; bu yetki
EGM/Jandarma'dadır. Sistem karar destek üretir.

**Neden kusur yüzdesi vermiyor?**
KTY m.156/3 uyarınca tutanağı düzenleyen görevli bile kusur oranı belirtmez; oranı TRAMER
belirler. Motor tutanak mantığını izler: hangi madde ihlal edildi, m.84 kapsamında asli
mi tali mi.

**`TESPIT_EDILEMEDI` "kusursuz" demek mi?**
Hayır. O tarafta ihlal **çıkarılamadı** demektir.

**CCTV'de çizik/göçük neden ayırt edilemiyor?**
Ölçüldü: 640 px referansına göre 256 px'te mAP %27.8 düşer, 128 px'te %71. Sabit CCTV
aracı 60×40 piksellik kutuda görür. `MIN_KENAR_PIKSEL` bu ölçüme dayanır; altında kalan
kırpma için sınıf bazlı çıktı üretilmez.

**Ceza tablosu değişti, kod güncellemem gerekir mi?**
Hayır. `config/penalties/` altına yeni tarihli YAML ekleyin.

**PostgreSQL'e geçebilir miyim?**
Şema taşınabilir yazıldı; geçiş bağlantı dizesi değişikliğidir. Migration altyapısı
bilinçli olarak kurulmadı — gerçek ikinci kamera gelmeden erken.

**Lisans durumu?**
Ultralytics YOLO26 **AGPL-3.0** (fine-tune ağırlıklar devralır), RDD2022 **CC BY-SA 4.0**
(atıf + aynı lisansla paylaşım zorunlu), CarDD **ticari kullanıma kapalı**. Ayrıntı:
[README](../README.md) "Lisans notu".
