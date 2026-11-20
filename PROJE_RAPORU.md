# COMPUTER VISION – ÖDEV RAPORU
## Derin Öğrenme Tabanlı Gerçek Zamanlı Yüz Tespiti (YOLOv8n & Streamlit)

---

## 1. Ödevin Amacı
Bu projenin amacı, açık kaynaklı **WIDER FACE** veri seti üzerinde derin öğrenme tabanlı bir **yüz tespiti (Face Detection)** modeli geliştirmek, modelin başarımını nesne tespiti metrikleri (mAP, IoU, Precision, Recall, F1-Score, eğitim ve tahmin süresi) ile değerlendirmek ve eğitilen modeli **Streamlit** kullanarak son kullanıcıların kolayca test edebileceği çalışan bir web uygulamasına dönüştürmektir.

---

## 2. Görev (Task)
- **Seçilen Problem:** Yüz Tespiti (**Face Detection**) — tek sınıflı (`face`) nesne tespiti problemi.
- **Amaç:** Verilen herhangi bir RGB görüntü içindeki tüm insan yüzlerinin konumlarını (sınırlayıcı kutu / bounding box: `x1, y1, x2, y2`) ve güven skorlarını (`confidence`) yüksek doğrulukla ve düşük gecikmeyle tespit etmek.

---

## 3. Model Mimarisi (Architecture)
Bu çalışmada tek aşamalı (one-stage) modern nesne tespit mimarisi olan **YOLOv8n (You Only Look Once v8 Nano)** seçilmiştir.
- **Neden YOLOv8n?**
  - **Hız ve Hafiflik:** Yalnızca ~3,13 milyon parametre ve 8,6 GFLOPs hesaplama maliyetine sahiptir; GPU bulunmayan sistemlerde dahi gerçek zamanlı (~34,5 FPS) çalışabilmektedir.
  - **Anchor-Free Tasarım:** Önceden tanımlı çapa kutularına (anchor boxes) ihtiyaç duymadan farklı ölçeklerdeki yüzleri doğrudan tahmin eder.
  - **Transfer Öğrenme (Transfer Learning):** COCO veri seti üzerinde önceden eğitilmiş `yolov8n.pt` ağırlıkları başlangıç noktası olarak alınmış, çıkış katmanı tek sınıf (`nc=1: face`) olacak şekilde WIDER FACE veri seti üzerinde ince ayar (fine-tuning) yapılmıştır.

---

## 4. Veri Seti (Dataset)
- **Veri Seti:** **WIDER FACE** (CUHK-CSE)
- **Resmî URL:** http://shuoyang1213.me/WIDERFACE/
- **Hugging Face Aynası:** https://huggingface.co/datasets/CUHK-CSE/wider_face
- **Ön İşleme ve Bölümleme:**
  - Geçersiz (`invalid=1`) etiketler ve 16 pikselden küçük mikroskobik arka plan kalabalık yüzleri filtrelenmiştir.
  - Her görüntüde 1–12 arası net/orta ölçekli yüz barındıran örnekler arasından sabit rastgelelik tohumu (`seed=42`) ile birbirinden tamamen ayrık (disjoint) **Eğitim / Doğrulama / Test** kümeleri oluşturulmuştur.
  - Etiketler `[class_id, x_center, y_center, width, height]` normalize YOLO formatına dönüştürülmüştür.

| Küme | Görüntü Sayısı | Oran | Toplam Yüz Etiketi |
| :--- | :---: | :---: | :---: |
| **Eğitim (Train)** | 600 | %66,7 | 1.907 |
| **Doğrulama (Validation)** | 150 | %16,7 | 449 |
| **Test (Test)** | 150 | %16,7 | 483 |
| **Toplam** | **900** | **%100** | **2.839** |

---

## 5. Model Ayarları ve Eğitim
Eğitim süreci `train_and_evaluate.py` betiği ile gerçekleştirilmiştir. Seçilen ayarlar ve gerekçeleri:

- **Girdi Boyutu (`imgsz=320`):** CPU ortamında hızlı eğitim ve gerçek zamanlı çıkarım (inference) dengesi için `320 × 320` çözünürlük seçilmiştir.
- **Hiperparametreler:**
  - `epochs = 10`: Transfer öğrenme ile kayıp değerlerinin dengeli biçimde yakınsaması ve aşırı öğrenmenin (overfitting) önlenmesi için 10 epoch belirlenmiştir.
  - `batch_size = 16`: CPU bellek verimliliği ve kararlı gradyan güncellemeleri için 16 seçilmiştir.
  - `learning_rate (lr0 = 0.002, lrf = 0.01)`: Ön-eğitimli ağırlıkları bozmadan ince ayar yapabilmek için düşük başlangıç öğrenme oranı ve doğrusal azaltma uygulanmıştır.
  - `weight_decay = 0.0005`, `warmup_epochs = 1.0`: İlk epoch'ta kararlı ısınma ve düzenlileştirme (regularization) sağlanmıştır.
  - **Veri Artırma (Data Augmentation):** Yatay çevirme (`fliplr=0.5`), renk/parlaklık jitter (`HSV`) ve ölçekleme (`scale=0.5`) ile farklı ışık ve kamera açılarına genelleme artırılmıştır.
- **Optimizer (`AdamW`):** Ağırlık azaltmayı (weight decay) gradyan güncellemesinden ayrıştıran `AdamW`, özellikle transfer öğrenme ve küçük/orta ölçekli veri setlerinde SGD'ye kıyasla daha hızlı ve kararlı yakınsama sağladığı için tercih edilmiştir.
- **Aktivasyon / Çıkış Fonksiyonu:**
  - Ara katmanlarda pürüzsüz gradyan akışı sağlayan **SiLU (Swish)** aktivasyon fonksiyonu kullanılmıştır.
  - Çıkış katmanında (Decoupled Head) sınıf olasılığı için **Sigmoid**, kutu koordinat dağılımı için **Softmax (DFL)** kullanılmaktadır.
- **Kayıp Fonksiyonu (Loss Function):**
  - **CIoU Loss (Complete IoU, ağırlık=7.5):** Sınırlayıcı kutuların örtüşme alanı, merkez mesafesi ve en-boy oranını birlikte optimize eder.
  - **DFL (Distribution Focal Loss, ağırlık=1.5):** Kutu kenar koordinatlarını ayrık olasılık dağılımı olarak öğrenerek hassas konumlandırma sağlar.
  - **BCE Loss (Binary Cross-Entropy, ağırlık=0.5):** Her aday kutunun yüz (`face`) olup olmadığını sınıflandırır.

---

## 6. Performans Değerlendirmesi
Eğitim tamamlandıktan sonra en yüksek doğrulama skoruna sahip ağırlık (`models/face_detector_best.pt`), eğitim sırasında hiç görülmemiş **150 görüntülük (483 yüz) Test Kümesi** üzerinde değerlendirilmiştir.

### Test Kümesi Metrikleri
| Metrik | Değer |
| :--- | :---: |
| **mAP@0.5** | **0.7100** (%71,00) |
| **mAP@0.5:0.95** | **0.3693** (%36,93) |
| **Ortalama IoU (Mean IoU, eşleşen kutular)** | **0.7978** (%79,78) |
| **Precision (Kesinlik)** | **0.8177** (%81,77) |
| **Recall (Duyarlılık)** | **0.6316** (%63,16) |
| **F1-Score** | **0.7127** (%71,27) |
| **Toplam Eğitim Süresi** | **696.07 saniye** (~11 dk 36 sn) |
| **Ortalama Tahmin Süresi (Inference Time)** | **28.98 ms / görüntü** (~34,5 FPS) |

### Eğitim Eğrileri ve Test Tahminleri
- Eğitim ve doğrulama kayıp/metrik grafikleri: `outputs/training_curves.png`
- Test kümesinden örnek görsel çıktılar: `outputs/sample_predictions.png`

**Yorum:** Eğitim ve doğrulama kayıpları 10 epoch boyunca paralel ve düzenli şekilde düşmüştür; aşırı öğrenme (overfitting) gözlenmemiştir. %81,77 Precision ve %79,78 Mean IoU değerleri, modelin bulduğu yüzleri çok yüksek doğrulukla ve isabetli kutularla çevrelediğini göstermektedir. Recall değerinin (%63,16) Precision'dan düşük olmasının temel nedeni, `320 × 320` çözünürlükte aşırı küçük, yan profilden veya kısmen örtülü (occluded) yüzlerin daha düşük güven skoru üretmesidir.

---

## 7. Uygulama (Deployment)
Eğitilen model (`models/face_detector_best.pt`), **Streamlit** kullanılarak etkileşimli bir web uygulaması (`app.py`) hâline getirilmiştir.

- **Çalıştırma Komutu:** `streamlit run app.py`
- **Uygulama Özellikleri:**
  1. **Tekli Görüntü Tespiti:** Kullanıcı fotoğraf yükler; güven eşiği (`Confidence Threshold`) ve `NMS IoU` eşiği anlık olarak ayarlanabilir; tespit edilen yüz sayısı, tahmin süresi (ms), koordinat tablosu ve indirilebilir işaretlenmiş görsel sunulur.
  2. **Toplu Görüntü Testi:** Birden fazla fotoğrafı topluca test eder, gerçek yüz sayısı ile model tahminlerini karşılaştırma tablosu hâlinde sunar.
  3. **Kamera ile Anlık Test:** Bilgisayar kamerasından anlık görüntü alarak yüz tespiti yapar.
  4. **Model Performansı & Grafikler:** Test metriklerini, veri seti dağılımını ve eğitim grafiklerini arayüzde gösterir.

---

## 8. Gerçek Görüntülerle Test ve Değerlendirme (5 Telefon Fotoğrafı)
Modelin gerçek dünya koşullarındaki genelleme başarımını ölçmek amacıyla cep telefonu ile **5 farklı senaryoda** (önden çekim, grup fotoğrafı, yandan çekim, yetersiz ışık ve aksesuarlı çekim) fotoğraf çekilmiş ve Streamlit uygulaması üzerinden **`Confidence = 0.25`** ve **`NMS IoU = 0.45`** sabit eşik değerleriyle test edilmiştir. İşaretlenmiş model çıktıları `outputs/real_test_results/` klasöründe yer almaktadır.

### Gerçek Görüntü Test Sonuçları
| Fotoğraf | Senaryo / Koşul | Gerçek Yüz Sayısı | Tahmin Edilen Yüz | Ortalama Güven | Tahmin Süresi (ms) | Durum |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Fotoğraf 1** | Önden Çekim | 1 | 1 | %83,20 | 37,8 | Başarılı |
| **Fotoğraf 2** | Grup Fotoğrafı | 4 | 4 | %62,66 | 70,2 | Başarılı |
| **Fotoğraf 3** | Yandan Çekim (Profil) | 1 | 0 | %0,00 | 46,4 | Eksik Tespit (FN) |
| **Fotoğraf 4** | Yetersiz Işık | 1 | 1 | %28,91 | 55,5 | Başarılı |
| **Fotoğraf 5** | Aksesuarlı Çekim | 1 | 1 | %80,58 | 72,1 | Başarılı |
| **Toplam / Ortalama** | **5 Senaryo** | **8** | **7** | **%63,84*** | **56,4 ms** | **7/8 Yüz (%87,5 Recall, %100 Precision)** |

*\*Ortalama güven skoru tespit yapılan 4 fotoğraf üzerinden hesaplanmıştır.*

### Başarılı ve Başarısız Durumların Değerlendirilmesi
- **Başarılı Olduğu Durumlar:**
  - **Önden çekim (Fotoğraf 1, %83,20)** ve **aksesuarlı çekim (Fotoğraf 5, %80,58)** senaryolarında model çok yüksek güven skoru ile yüzü hatasız tespit etmiştir. Bu durum, modelin gözlük/şapka gibi kısmi aksesuarlara karşı güçlü bir genelleme kazandığını göstermektedir.
  - **Grup fotoğrafında (Fotoğraf 2)** 4 farklı yüzün tamamı tek seferde doğru tespit edilmiştir (4/4, ortalama %62,66 güven); hiçbir yanlış pozitif (FP) üretilmemiştir.
  - **Yetersiz ışıkta (Fotoğraf 4)** düşük aydınlatmaya rağmen yüz tespit edilebilmiştir; ancak kontrast azalmasına bağlı olarak güven skoru %28,91 seviyesine gerilemiştir.
  - Yüksek çözünürlüklü gerçek telefon fotoğraflarında dahi ortalama tahmin süresi **56,4 ms** olarak ölçülmüş, CPU üzerinde akıcı bir performans elde edilmiştir.
- **Başarısız Olduğu Durumlar ve İyileştirme Önerileri:**
  - **Yandan çekim / tam profil (Fotoğraf 3):** Model yüzü tespit edememiştir (Eksik Tespit / False Negative). Bunun temel nedeni, tek göz ve burun/çene hattının ön cepheye göre büyük ölçüde değişmesi ve `320 × 320` çözünürlükte 10 epoch ile eğitilen hafif `YOLOv8n` modelinin eğitim verisinde çoğunlukla ön/yarı-ön yüz özelliklerine odaklanmasıdır.
  - **Çözüm Önerisi:** Daha fazla yan profil yüz içeren örneklerle eğitim veri setinin zenginleştirilmesi, girdi boyutunun `640 × 640`'a çıkarılması ve daha uzun epoch (ör. 30–50 epoch, GPU üzerinde) ile eğitilmesi yan açı ve düşük ışık performansını belirgin biçimde artıracaktır.

---

## 9. Proje Raporu Şablonu

| Alan | Açıklama / Öğrenci yanıtı |
| :--- | :--- |
| **Project Title** | Derin Öğrenme Tabanlı Gerçek Zamanlı Yüz Tespiti (YOLOv8n & Streamlit) |
| **Problem / Task** | Yüz Tespiti (Face Detection — Single-Class Object Detection) |
| **Dataset** | WIDER FACE (900 görüntü / 2.839 yüz etiketi; 600 Train / 150 Val / 150 Test) |
| **Dataset URL** | http://shuoyang1213.me/WIDERFACE/ (HF Mirror: https://huggingface.co/datasets/CUHK-CSE/wider_face) |
| **Model Architecture** | YOLOv8n (Ultralytics, COCO ön-eğitimli transfer öğrenme, 3.13M parametre) |
| **Input Size** | 320 × 320 × 3 (RGB) |
| **Hyperparameters** | epochs=10, batch_size=16, lr0=0.002, lrf=0.01, weight_decay=0.0005, warmup_epochs=1.0, seed=42 |
| **Optimizer** | AdamW |
| **Activation / Output Function** | Gizli katmanlar: SiLU (Swish) / Çıkış katmanı: Sigmoid (sınıf) & Softmax-DFL (kutu koordinatı) |
| **Loss Function** | CIoU Loss (box=7.5) + Distribution Focal Loss (dfl=1.5) + Binary Cross-Entropy (cls=0.5) |
| **Evaluation Metrics** | mAP@0.5, mAP@0.5:0.95, Mean IoU, Precision, Recall, F1-Score, Training Time, Inference Time |
| **Training Environment** | Windows 10, AMD Ryzen 5 4600H CPU (12 threads), Python 3.11.18, PyTorch 2.15.0+cpu, Ultralytics 8.4.216 |
| **Best Performance** | Test mAP@0.5: 0.7100, mAP@0.5:0.95: 0.3693, Mean IoU: 0.7978, Precision: 0.8177, Recall: 0.6316, F1-Score: 0.7127 (Eğitim süresi: 696.07 sn) |
| **Inference Time** | 28.98 ms / görüntü (~34.5 FPS, CPU üzerinde) |
| **Deployment Technology** | Streamlit (v1.67.0) |
| **Repository** | https://github.com/gunsozkrt/face-recognation-w-YOLO |
