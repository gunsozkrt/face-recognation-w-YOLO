import io
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image, ImageOps
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "face_detector_best.pt"
METRICS_PATH = BASE_DIR / "outputs" / "metrics.json"
DATASET_STATS_PATH = BASE_DIR / "data" / "dataset_stats.json"
REAL_TEST_OUT_DIR = BASE_DIR / "outputs" / "real_test_results"

st.set_page_config(
    page_title="Yüz Tespiti (YOLOv8) — Bilgisayarlı Görü",
    page_icon="🧑‍💻",
    layout="wide",
)


@st.cache_resource
def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def fix_orientation(pil_img: Image.Image) -> Image.Image:
    """Apply EXIF orientation (important for phone camera photos) and convert to RGB."""
    return ImageOps.exif_transpose(pil_img).convert("RGB")


def run_detection(
    model: YOLO,
    pil_img: Image.Image,
    conf_thresh: float,
    iou_thresh: float,
    line_width: int = 3,
    show_conf: bool = True,
):
    img_np = np.array(pil_img)  # RGB
    start = time.perf_counter()
    results = model.predict(
        source=img_np,
        conf=conf_thresh,
        iou=iou_thresh,
        verbose=False,
    )
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    result = results[0]
    annotated_bgr = result.plot(
        line_width=line_width,
        conf=show_conf,
        labels=True,
    )
    # Ultralytics plot() returns BGR array when input is RGB/BGR; convert carefully:
    # Since we passed RGB numpy array to predict(), result.orig_img is RGB, so plot() draws on it directly.
    # Let's draw cleanly with OpenCV on a copy of img_np (RGB) so colors are guaranteed accurate.
    annotated = img_np.copy()
    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes_xyxy = result.boxes.xyxy.cpu().numpy()
        confs = result.boxes.conf.cpu().numpy()
        for idx, (box, conf) in enumerate(zip(boxes_xyxy, confs), start=1):
            x1, y1, x2, y2 = [int(round(v)) for v in box]
            detections.append(
                {
                    "Yüz #": idx,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "Genişlik (px)": x2 - x1,
                    "Yükseklik (px)": y2 - y1,
                    "Güven Skoru": round(float(conf), 4),
                }
            )
            cv2.rectangle(annotated, (x1, y1), (x2, y2), (16, 185, 129), line_width)
            label = f"face {conf:.2f}" if show_conf else "face"
            font_scale = max(0.5, min(annotated.shape[1], annotated.shape[0]) / 900.0)
            (tw, th), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2
            )
            ty = max(y1 - 6, th + 6)
            cv2.rectangle(
                annotated,
                (x1, ty - th - 6),
                (x1 + tw + 8, ty + baseline - 2),
                (16, 185, 129),
                -1,
            )
            cv2.putText(
                annotated,
                label,
                (x1 + 4, ty - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    annotated_pil = Image.fromarray(annotated)
    return annotated_pil, detections, elapsed_ms


def pil_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def main():
    st.title("🧑‍💻 Yüz Tespiti (Face Detection) — YOLOv8 Uygulaması")
    st.caption(
        "WIDER FACE veri seti üzerinde eğitilmiş YOLOv8n derin öğrenme modeli ile gerçek zamanlı yüz tespiti."
    )

    if not MODEL_PATH.exists():
        st.error(
            f"Eğitilmiş model dosyası bulunamadı: `{MODEL_PATH}`\n\n"
            "Lütfen önce `python prepare_dataset.py` ve `python train_and_evaluate.py` komutlarını çalıştırın."
        )
        st.stop()

    model = load_model(str(MODEL_PATH))

    with st.sidebar:
        st.header("⚙️ Model Ayarları")
        conf_thresh = st.slider(
            "Güven Eşiği (Confidence Threshold)",
            min_value=0.10,
            max_value=0.95,
            value=0.25,
            step=0.05,
            help="Bu değerin altındaki tespitler filtrelenir.",
        )
        iou_thresh = st.slider(
            "NMS IoU Eşiği",
            min_value=0.10,
            max_value=0.90,
            value=0.45,
            step=0.05,
            help="Çakışan kutuların birleştirilmesi için IoU eşiği.",
        )
        line_width = st.slider("Çerçeve Kalınlığı", 1, 8, 3)
        show_conf = st.checkbox("Güven skorunu göster", value=True)

        st.divider()
        st.markdown("**Model:** YOLOv8n (Fine-tuned)")
        st.markdown("**Veri Seti:** WIDER FACE")
        st.markdown("**Görev:** Yüz Tespiti (Face Detection)")

    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📷 Tekli Görüntü Tespiti",
            "🖼️ Toplu Görüntü Testi",
            "🎥 Kamera ile Test",
            "📊 Model Performansı & Grafikler",
        ]
    )

    # --- TAB 1: Single Image ---
    with tab1:
        st.subheader("Tekli Görüntü Yükle ve Yüz Tespit Et")
        uploaded = st.file_uploader(
            "Bir fotoğraf seçin (JPG, JPEG, PNG, WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="single_upload",
        )
        if uploaded is not None:
            orig_img = fix_orientation(Image.open(uploaded))
            annotated_img, detections, elapsed_ms = run_detection(
                model, orig_img, conf_thresh, iou_thresh, line_width, show_conf
            )

            avg_conf = (
                float(np.mean([d["Güven Skoru"] for d in detections]))
                if detections
                else 0.0
            )
            c1, c2, c3 = st.columns(3)
            c1.metric("Tespit Edilen Yüz", len(detections))
            c2.metric("Tahmin Süresi", f"{elapsed_ms:.1f} ms")
            c3.metric("Ortalama Güven Skoru", f"{avg_conf:.2%}")

            col_l, col_r = st.columns(2)
            with col_l:
                st.image(orig_img, caption="Orijinal Görüntü", use_container_width=True)
            with col_r:
                st.image(
                    annotated_img,
                    caption=f"Model Çıktısı ({len(detections)} yüz)",
                    use_container_width=True,
                )

            if detections:
                st.markdown("#### Tespit Detayları")
                st.dataframe(pd.DataFrame(detections), use_container_width=True)
            else:
                st.info(
                    "Bu güven eşiğinde yüz tespit edilemedi. Sol menüden güven eşiğini düşürmeyi deneyebilirsiniz."
                )

            st.download_button(
                label="💾 İşaretlenmiş Görüntüyü İndir (PNG)",
                data=pil_to_bytes(annotated_img),
                file_name=f"detected_{Path(uploaded.name).stem}.png",
                mime="image/png",
            )

    # --- TAB 2: Batch Evaluation ---
    with tab2:
        st.subheader("Toplu Görüntü Testi ve Karşılaştırma")
        st.write(
            "Birden fazla fotoğrafı topluca yükleyip modelin tahminlerini gerçek yüz sayılarıyla karşılaştırabilirsiniz."
        )
        multi_files = st.file_uploader(
            "Fotoğrafları seçin",
            type=["jpg", "jpeg", "png", "webp"],
            accept_multiple_files=True,
            key="multi_upload",
        )

        if multi_files:
            REAL_TEST_OUT_DIR.mkdir(parents=True, exist_ok=True)
            rows = []

            for idx, f in enumerate(multi_files, start=1):
                st.markdown(f"---")
                st.markdown(f"### Fotoğraf {idx}: `{f.name}`")
                orig_img = fix_orientation(Image.open(f))
                annotated_img, detections, elapsed_ms = run_detection(
                    model, orig_img, conf_thresh, iou_thresh, line_width, show_conf
                )

                out_path = REAL_TEST_OUT_DIR / f"photo_{idx}_{Path(f.name).stem}.png"
                annotated_img.save(out_path)

                c_img1, c_img2, c_meta = st.columns([2, 2, 1.5])
                with c_img1:
                    st.image(orig_img, caption="Orijinal", use_container_width=True)
                with c_img2:
                    st.image(
                        annotated_img,
                        caption=f"Tahmin ({len(detections)} yüz)",
                        use_container_width=True,
                    )
                with c_meta:
                    gt_count = st.number_input(
                        "Gerçek Yüz Sayısı",
                        min_value=0,
                        max_value=100,
                        value=max(1, len(detections)),
                        step=1,
                        key=f"gt_{idx}_{f.name}",
                    )
                    note = st.text_input(
                        "Kısa Yorum (Işık/Açı/Mesafe)",
                        value="Ön cephe, doğal ışık",
                        key=f"note_{idx}_{f.name}",
                    )
                    avg_c = (
                        float(np.mean([d["Güven Skoru"] for d in detections]))
                        if detections
                        else 0.0
                    )
                    st.metric("Tahmin Süresi", f"{elapsed_ms:.1f} ms")
                    st.metric("Ort. Güven", f"{avg_c:.2%}")

                status = (
                    "Başarılı"
                    if len(detections) == gt_count
                    else (
                        "Eksik Tespit (FN)"
                        if len(detections) < gt_count
                        else "Fazla Tespit (FP)"
                    )
                )
                rows.append(
                    {
                        "Fotoğraf": f"Fotoğraf {idx} ({f.name})",
                        "Gerçek Yüz": gt_count,
                        "Tahmin Edilen Yüz": len(detections),
                        "Ortalama Güven": f"{avg_c:.2%}",
                        "Tahmin Süresi (ms)": round(elapsed_ms, 1),
                        "Durum": status,
                        "Yorum": note,
                    }
                )

            st.markdown("---")
            st.subheader("📋 Karşılaştırma Özeti")
            summary_df = pd.DataFrame(rows)
            st.dataframe(summary_df, use_container_width=True)

    # --- TAB 3: Webcam ---
    with tab3:
        st.subheader("Bilgisayar Kamerası ile Anlık Test")
        cam_img = st.camera_input("Fotoğraf çekin")
        if cam_img is not None:
            orig_img = fix_orientation(Image.open(cam_img))
            annotated_img, detections, elapsed_ms = run_detection(
                model, orig_img, conf_thresh, iou_thresh, line_width, show_conf
            )
            c1, c2 = st.columns(2)
            c1.metric("Tespit Edilen Yüz", len(detections))
            c2.metric("Tahmin Süresi", f"{elapsed_ms:.1f} ms")
            st.image(annotated_img, caption="Kamera Tespit Sonucu", use_container_width=True)

    # --- TAB 4: Metrics & Charts ---
    with tab4:
        st.subheader("Eğitim ve Test Performans Metrikleri")
        if METRICS_PATH.exists():
            metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
            test_m = metrics.get("test_metrics", {})

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Test mAP@0.5", f"{test_m.get('mAP50', 0):.4f}")
            m2.metric("Test mAP@0.5:0.95", f"{test_m.get('mAP50_95', 0):.4f}")
            m3.metric("Ortalama IoU", f"{test_m.get('mean_iou', 0):.4f}")
            m4.metric("F1-Score", f"{test_m.get('f1_score', 0):.4f}")

            m5, m6, m7, m8 = st.columns(4)
            m5.metric("Precision", f"{test_m.get('precision', 0):.4f}")
            m6.metric("Recall", f"{test_m.get('recall', 0):.4f}")
            m7.metric(
                "Eğitim Süresi",
                f"{metrics.get('training_time_sec', 0):.1f} sn",
            )
            m8.metric(
                "Tahmin Süresi (Ort.)",
                f"{metrics.get('inference_time_ms', 0):.1f} ms/görüntü",
            )

            if DATASET_STATS_PATH.exists():
                ds = json.loads(DATASET_STATS_PATH.read_text(encoding="utf-8"))
                st.markdown("#### Veri Seti Dağılımı (Train / Val / Test)")
                ds_df = pd.DataFrame(
                    [
                        {
                            "Küme": k.upper(),
                            "Görüntü Sayısı": v["images"],
                            "Toplam Yüz Etiketi": v["faces"],
                        }
                        for k, v in ds.get("splits", {}).items()
                    ]
                )
                st.dataframe(ds_df, use_container_width=True)
        else:
            st.warning("`outputs/metrics.json` henüz oluşturulmamış.")

        curves_path = BASE_DIR / "outputs" / "training_curves.png"
        samples_path = BASE_DIR / "outputs" / "sample_predictions.png"
        if curves_path.exists():
            st.markdown("#### Eğitim ve Doğrulama Eğrileri")
            st.image(str(curves_path), use_container_width=True)
        if samples_path.exists():
            st.markdown("#### Test Kümesinden Örnek Tahminler")
            st.image(str(samples_path), use_container_width=True)


if __name__ == "__main__":
    main()
