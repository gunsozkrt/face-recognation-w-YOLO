# Yüz Tespiti (Face Detection) — YOLOv8 & Streamlit Uygulaması

Bu proje, **Bilgisayarlı Görü (Computer Vision) Derin Öğrenme Uygulaması** ödevi kapsamında **WIDER FACE** açık kaynaklı veri seti üzerinde fine-tune edilen **YOLOv8n** yüz tespiti modelini ve **Streamlit** tabanlı etkileşimli web uygulamasını içerir.

## Proje Yapısı

```text
face-detection-cv/
├── prepare_dataset.py        # WIDER FACE indirme & YOLO formatında train/val/test bölme
├── train_and_evaluate.py     # YOLOv8n eğitimi, test metrikleri (mAP, IoU, P, R, F1) ve grafikler
├── app.py                    # Streamlit web uygulaması (tekli test, 5 telefon fotoğrafı testi, kamera)
├── requirements.txt          # Python bağımlılıkları
├── PROJE_RAPORU.md           # Ödev şablonuna tam uyumlu proje raporu
├── data/                     # Hazırlanan veri seti, dataset.yaml ve istatistikler
├── models/                   # Eğitilmiş en iyi model (face_detector_best.pt)
└── outputs/                  # Metrikler (metrics.json), grafikler ve gerçek test çıktıları
```

## Kurulum ve Çalıştırma

### 1. Sanal Ortamı Aktifleştirme
```powershell
.\.venv\Scripts\Activate.ps1
```

### 2. Veri Setini Hazırlama (WIDER FACE -> YOLO)
```powershell
python prepare_dataset.py
```

### 3. Modeli Eğitme ve Değerlendirme
```powershell
python train_and_evaluate.py
```

### 4. Streamlit Uygulamasını Başlatma
```powershell
streamlit run app.py
```

Uygulama varsayılan olarak `http://localhost:8501` adresinde açılır.

## Gerçek Görüntülerle Test
1. Streamlit uygulamasını açın ve **"Toplu Görüntü Testi"** sekmesine geçin.
2. Test etmek istediğiniz fotoğrafları yükleyin.
3. Her fotoğraf için gerçek yüz sayısını girerek model tahminleriyle karşılaştırma tablosunu inceleyebilirsiniz.
