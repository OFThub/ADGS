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

## Test

```bash
pytest -q
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

## Plan

Tam mimari plan, modül kırılımı ve yol haritası için proje planına bakınız.
