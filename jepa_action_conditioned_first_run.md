# Video JEPA: Bottom-Up First-Run Plan

## Amaç

Bu çalışmanın ana odağı Video JEPA'dır. I-JEPA yalnızca fikrin görsel temsil
öğrenme kökenini açıklamak için kısa tutulacaktır.

Bottom-up sıra:

1. Action kullanmadan video dinamiğini latent uzayda tahmin et (`video_jepa`).
2. Eğitilmiş modelde tek ve çok adımlı gelecek latent tahminini incele.
3. Ardından action bilgisini ekle (`ac_video_jepa`).
4. Doğru action ile yanlış/karıştırılmış action'ın tahmin hatasını karşılaştır.

İlk amaç benchmark yeniden üretmek değildir. Amaç, JEPA veri akışını kodda
bulmak, küçük bir eğitimi çalıştırmak ve modelin gerçekten ne öğrendiğini
gösteren birkaç kontrollü çıktı almaktır.

> Terminoloji notu: Bu repodaki Moving MNIST `video_jepa` örneği, Meta'nın
> orijinal V-JEPA mimarisinin küçük birebir reprodüksiyonu değildir. Maskeli
> spatiotemporal prediction ve EMA target encoder yerine, JEPA prensibini
> gelecekteki video latent'ini tahmin ederek gösteren sade bir eğitim örneğidir.

---

## Deney 1 — State-Only Video JEPA (Ana Run)

### Soru

Model, yalnızca geçmiş video karelerinden gelecekteki durumun temsilini
tahmin edebiliyor mu?

```text
geçmiş kareler
    |
    v
encoder (ResNet5)
    |
    v
geçmiş latent'ler
    |
    v
state-only predictor (ResUNet)
    |
    v
tahmin edilen gelecek latent
    |
    v
gerçek gelecek karenin latent'i ile karşılaştırma
```

Basitleştirilmiş ifade:

```text
z_t       = encoder(o_t)
z_target  = encoder(o_t+1)
z_pred    = predictor(previous_latents)
loss_pred = distance(z_pred, z_target)
```

Bu örnekte ayrı bir teacher/EMA target encoder yoktur. Gelecek hedefi de aynı
encoder ile temsil edilir.

### Veri

- Dataset: Moving MNIST
- İki rakam görüntü içinde hareket eder ve sınırlardan seker.
- Repo ilk kullanımda yaklaşık 800 MB veri indirir.
- Kod varsayılan olarak 9.000 train ve 1.000 validation videosu kullanır.
- Her örnek bir video dizisi, digit-location hedefi ve değerlendirme bilgisidir.

### Model ve loss

- Encoder: `ResNet5`
- Predictor: `ResUNet`, iki latent karelik context kullanır
- Projector: MLP
- Prediction loss: tahmin edilen ve gerçek gelecek latent arasında MSE
- VC regularizer:
  - variance/std loss collapse'ı önler
  - covariance loss redundant latent boyutlarını azaltır
- Yardımcı evaluation head'leri:
  - pixel decoder
  - digit-location detection head

Toplam eğitimde JEPA loss yanında iki probe/decoder loss'u da optimize edilir.
Bu head'ler latent rollout'u yorumlamak ve görselleştirmek içindir.

### İki seviyeli çalışma

#### A. Smoke test

Amaç yalnızca şunları doğrulamaktır:

- veri yükleniyor,
- forward/backward çalışıyor,
- loss finite,
- GPU gerçekten kullanılıyor,
- checkpoint yazılıyor.

Gerçek bir tiny smoke run için train/validation subset desteği eklenmesi önerilir:

```yaml
data:
  train_size: 512
  val_size: 128
```

Subset desteği eklenmeden config'deki batch size veya epoch sayısını azaltmak,
bir epoch içindeki 9.000 train videosunu azaltmaz.

Hedef smoke ayarı:

```text
train_size=512
val_size=128
batch_size=32
model.steps=1 veya 2
epochs=1
wandb=false
```

#### B. Mini learning run

Smoke test geçtikten sonra:

```text
train_size=2.000–4.000
val_size=256–512
batch_size=32 veya 64
model.steps=2
epochs=5–10
wandb=false (ilk denemede)
```

Bu aşamada “model öğrendi” diyebilmek için yalnız son loss'a değil, aşağıdaki
karşılaştırmalara bakılacaktır:

- başlangıç ve bitiş prediction loss'u,
- train ve validation prediction loss'u,
- untrained ve trained modelin aynı video üzerindeki rollout'u,
- kısa horizon ve daha uzun horizon hatası.

### Mevcut kodla başlangıç komutu

Subset desteği eklenmeden, tam 9.000 örneklik dataset üzerinde kısa çalışma:

```bash
uv run python -m examples.video_jepa.main \
  --fname examples/video_jepa/cfgs/default.yaml \
  logging.log_wandb=false \
  data.num_workers=2 \
  data.batch_size=32 \
  model.steps=2 \
  optim.epochs=3
```

Colab'de GPU belleği uygunsa `batch_size=64` daha hızlı olabilir. OOM oluşursa
önce `batch_size=32`, sonra `16` denenmelidir.

### Kaydedilecek çıktılar

- exact command ve kullanılan final config
- GPU modeli (`nvidia-smi`)
- epoch süresi ve toplam süre
- train/validation prediction loss
- VC loss bileşenleri
- checkpoint
- bir input video ve bir decoded rollout
- tensor shape'leri:
  - input video
  - encoder latent'i
  - predictor context'i
  - predicted future latent
  - target future latent

---

## Deney 2 — Action-Conditioned Video JEPA (Kısa Devam Run'ı)

### Soru

Gelecek yalnızca mevcut görüntüye değil, seçilen action'a bağlı olduğunda model
action bilgisini doğru biçimde kullanıyor mu?

```text
observation o_t
    |
    v
Impala encoder
    |
    v
latent z_t + action a_t
    |
    v
GRU predictor
    |
    v
predicted z_(t+1)
```

```text
z_t      = encoder(o_t)
z_next   = encoder(o_t+1)
pred     = predictor(z_t, a_t)
pred_err = distance(pred, z_next)
```

Burada da ayrı bir target/teacher encoder yoktur.

### Veri ve loss

- Dataset/environment: runtime'da üretilen Two Rooms
- Observation: agent ve duvar görüntüsü
- Action: iki boyutlu hareket vektörü
- Encoder output: global 512-boyutlu latent
- Predictor: action alan GRU
- Loss:
  - latent prediction
  - variance
  - covariance
  - temporal similarity
  - inverse dynamics (IDM)

### Minimal run

Planning evaluation, W&B ve compile ilk eğitimde kapatılır:

```bash
uv run python -m examples.ac_video_jepa.main \
  --fname examples/ac_video_jepa/cfgs/train.yaml \
  logging.log_wandb=false \
  meta.load_model=false \
  meta.enable_plan_eval=false \
  model.compile=false \
  training.dtype=float16 \
  data.size=2048 \
  data.val_size=128 \
  data.batch_size=32 \
  data.num_workers=2 \
  model.nsteps=2 \
  optim.epochs=3
```

Bu ikinci run'ın amacı planning başarı oranı üretmek değildir. Amaç yalnızca
action-conditioned prediction mekanizmasını doğrulamaktır.

T4 üzerinde `bfloat16` yerine `float16` seçilmiştir. Daha yeni bir GPU verilirse
`bfloat16` ayrıca denenebilir.

### Asıl sanity check

Sadece iki farklı action'ın farklı prediction vermesi yeterli değildir;
eğitilmemiş bir ağ da bunu yapabilir. Bunun yerine aynı batch üzerinde action'lar
karıştırılır:

```text
error_correct = distance(P(z_t, a_t),        z_(t+1))
error_wrong   = distance(P(z_t, shuffled_a), z_(t+1))
```

Beklenti:

```text
error_correct < error_wrong
```

Bu fark batch ortalamasıyla ve mümkünse birkaç farklı seed ile raporlanmalıdır.

---

## Google Colab Ortamı

### Önerilen runtime

- Runtime type: Python 3
- Hardware accelerator: NVIDIA GPU
- Minimum pratik hedef: T4 sınıfı GPU, yaklaşık 15 GB VRAM
- Daha hızlı seçenek: L4/A100 gibi premium GPU (erişim garanti değildir)
- TPU gerekli değildir; repo PyTorch/CUDA akışına göre yazılmıştır
- High-RAM runtime ilk denemede gerekli değildir

Colab GPU tipi, kullanım kotası ve oturum süresi dinamik olduğu için notebook
başında donanım mutlaka kaydedilmelidir:

```bash
!nvidia-smi
!python --version
```

Python sürümü proje gereksinimi olan 3.12 ile uyuşmazsa `uv` üzerinden Python
3.12 kurulup proje o ortamda çalıştırılmalıdır.

### Kurulum hücreleri

```python
from google.colab import drive
drive.mount('/content/drive')
```

```bash
%cd /content
!git clone https://github.com/facebookresearch/eb_jepa.git
%cd /content/eb_jepa
!pip -q install uv
!uv python install 3.12
!uv sync
```

Komutlar notebook hücresinde `uv run python ...` biçiminde çalıştırılmalıdır.
İlk `uv sync` PyTorch ve diğer bağımlılıkları indireceği için sonraki çalışmalara
göre daha uzun sürer.

### Dataset ve checkpoint kalıcılığı

Colab VM'i geçicidir; bağlantı kesilince `/content` silinebilir.

- Moving MNIST dosyasını ilk indirmeden sonra Google Drive'a kopyala.
- Deney sonunda `config.yaml`, log, görsel ve checkpoint klasörünü Drive'a kopyala.
- Eğitimi doğrudan Drive üzerinde çalıştırmak çok sayıda küçük yazma nedeniyle
  yavaşlayabilir. Eğitim `/content` altında, sonuç kopyalama en sonda yapılmalıdır.

Örnek:

```bash
!mkdir -p /content/drive/MyDrive/eb_jepa_artifacts
!cp -r /content/eb_jepa/checkpoints/video_jepa \
  /content/drive/MyDrive/eb_jepa_artifacts/
```

### Tahmini süreler

Aşağıdaki değerler ölçüm değil, T4 sınıfı GPU için planlama aralığıdır. Colab'ın
verdiği GPU, disk/ağ hızı, batch size ve validation sıklığı süreyi değiştirebilir.

| İş | T4 tahmini | L4/A100 tahmini |
|---|---:|---:|
| Repo + environment kurulumu | 5–15 dk | 5–15 dk |
| Moving MNIST ilk indirme (~800 MB) | 2–15 dk | 2–15 dk |
| Video JEPA subset smoke run | 2–6 dk | 1–4 dk |
| Video JEPA mini run (2k–4k, 5 epoch) | 15–45 dk | 8–25 dk |
| Video JEPA mevcut tam dataset, 3 epoch | 20–60 dk | 10–35 dk |
| AC-JEPA smoke/mini run (2k, 3 epoch) | 10–30 dk | 5–20 dk |
| Prediction + shape + sanity check | 2–10 dk | 2–8 dk |

İlk epoch gerçek süre ölçümü için kullanılmalıdır:

```text
tahmini toplam eğitim süresi
  = ilk epoch süresi × kalan epoch sayısı
```

İlk epoch veri indirme, CUDA warm-up ve cache nedeniyle biraz daha yavaş olabilir.

### Colab Free yeterli mi?

Bu küçültülmüş deneyler için çoğu durumda evet. Ancak ücretsiz GPU erişimi,
GPU modeli ve kullanım kotası garanti değildir; runtime beklenmedik biçimde
kapanabilir. Bu yüzden her epoch checkpoint almak ve sonuçları Drive'a kopyalamak
gereklidir. Premium Colab ancak:

- ücretsiz GPU verilmiyorsa,
- oturum sürekli kesiliyorsa,
- full dataset veya daha uzun sweep yapılacaksa

gerekli hâle gelir. İlk smoke ve mini run için doğrudan ücretli plana geçmek
zorunlu değildir.

---

## Hocaya Sunulacak Kısa Sonuç

Sunumun omurgası:

1. JEPA piksel üretmek yerine latent uzayda tahmin yapar.
2. State-only Video JEPA geçmiş karelerden gelecekteki latent'i tahmin eder.
3. Prediction objective tek başına collapse riski taşıdığı için temsil
   regularization'ı kullanılır.
4. Action-conditioned model aynı fikri `state + action -> next state` biçimine
   genişletir.
5. Video run'ında trained/untrained rollout; action-conditioned run'da doğru ve
   shuffled action hatası mekanizmayı somutlaştırır.

Bu çalışma orijinal V-JEPA benchmark reprodüksiyonu olarak değil, V-JEPA/JEPA
mekanizmasını kodda izleyen ve küçük ölçekte doğrulayan bottom-up bir inceleme
olarak sunulmalıdır.

---

## Definition of Done

### Video JEPA

- [ ] Colab GPU ve environment doğrulandı
- [ ] smoke run tamamlandı
- [ ] mini learning run tamamlandı
- [ ] loss'lar ve epoch süreleri kaydedildi
- [ ] checkpoint Drive'a kopyalandı ve yeniden yüklendi
- [ ] bir future-latent prediction çalıştı
- [ ] trained/untrained rollout karşılaştırıldı
- [ ] önemli tensor shape'leri kaydedildi

### Action-Conditioned JEPA

- [ ] küçük eğitim tamamlandı
- [ ] checkpoint yeniden yüklendi
- [ ] `observation + action -> future latent` çalıştı
- [ ] correct-action ve shuffled-action error karşılaştırıldı

### Anlatım

- [ ] state-only ve action-conditioned veri akışını açıklayabiliyorum
- [ ] collapse regularizer'larının neden gerektiğini açıklayabiliyorum
- [ ] bu repo örneği ile orijinal V-JEPA arasındaki farkı doğru ifade edebiliyorum
