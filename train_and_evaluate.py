"""
YOLOv8n Yüz Tespiti Model Eğitimi ve Değerlendirme Betiği
- Veri seti: data/dataset.yaml (WIDER FACE train/val/test)
- Çıktılar:
  * models/face_detector_best.pt
  * outputs/metrics.json
  * outputs/training_curves.png
  * outputs/sample_predictions.png
"""

import json
import os
import platform
import shutil
import time
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import ultralytics
from PIL import Image
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent
DATA_YAML = BASE_DIR / "data" / "dataset.yaml"
TEST_IMG_DIR = BASE_DIR / "data" / "images" / "test"
TEST_LBL_DIR = BASE_DIR / "data" / "labels" / "test"
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
RUNS_DIR = BASE_DIR / "runs"

# Hiperparametreler
CONFIG = {
    "model_arch": "YOLOv8n (yolov8n.pt transfer learning)",
    "imgsz": 320,
    "epochs": 10,
    "batch": 16,
    "optimizer": "AdamW",
    "lr0": 0.002,
    "lrf": 0.01,
    "weight_decay": 0.0005,
    "warmup_epochs": 1.0,
    "seed": 42,
    "conf_eval": 0.35,
    "iou_eval": 0.45,
}


def compute_iou_matrix(boxes1: np.ndarray, boxes2: np.ndarray) -> np.ndarray:
    """boxes1: (N, 4), boxes2: (M, 4) in xyxy format."""
    if len(boxes1) == 0 or len(boxes2) == 0:
        return np.zeros((len(boxes1), len(boxes2)), dtype=np.float32)

    x1 = np.maximum(boxes1[:, None, 0], boxes2[None, :, 0])
    y1 = np.maximum(boxes1[:, None, 1], boxes2[None, :, 1])
    x2 = np.minimum(boxes1[:, None, 2], boxes2[None, :, 2])
    y2 = np.minimum(boxes1[:, None, 3], boxes2[None, :, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = np.maximum(0.0, boxes1[:, 2] - boxes1[:, 0]) * np.maximum(
        0.0, boxes1[:, 3] - boxes1[:, 1]
    )
    area2 = np.maximum(0.0, boxes2[:, 2] - boxes2[:, 0]) * np.maximum(
        0.0, boxes2[:, 3] - boxes2[:, 1]
    )
    union = area1[:, None] + area2[None, :] - inter + 1e-6
    return inter / union


def load_gt_boxes_xyxy(lbl_path: Path, img_w: int, img_h: int) -> np.ndarray:
    if not lbl_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    boxes = []
    for line in lbl_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        _, xc, yc, w, h = map(float, parts)
        x1 = (xc - w / 2.0) * img_w
        y1 = (yc - h / 2.0) * img_h
        x2 = (xc + w / 2.0) * img_w
        y2 = (yc + h / 2.0) * img_h
        boxes.append([x1, y1, x2, y2])
    return np.array(boxes, dtype=np.float32) if boxes else np.zeros((0, 4), dtype=np.float32)


def evaluate_mean_iou_and_speed(model: YOLO, conf_thresh: float, iou_thresh: float):
    test_images = sorted(TEST_IMG_DIR.glob("*.jpg"))
    matched_ious = []
    inference_times_ms = []

    # Warmup
    if test_images:
        model.predict(
            source=str(test_images[0]),
            imgsz=CONFIG["imgsz"],
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
        )

    for img_path in test_images:
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        h, w = img_rgb.shape[:2]
        gt_boxes = load_gt_boxes_xyxy(TEST_LBL_DIR / f"{img_path.stem}.txt", w, h)

        t0 = time.perf_counter()
        res = model.predict(
            source=img_rgb,
            imgsz=CONFIG["imgsz"],
            conf=conf_thresh,
            iou=iou_thresh,
            verbose=False,
        )[0]
        dt_ms = (time.perf_counter() - t0) * 1000.0
        inference_times_ms.append(dt_ms)

        if res.boxes is not None and len(res.boxes) > 0 and len(gt_boxes) > 0:
            pred_boxes = res.boxes.xyxy.cpu().numpy()
            iou_mat = compute_iou_matrix(pred_boxes, gt_boxes)
            # Her tahmin için en iyi eşleşen GT kutusunun IoU değeri (greedy matching)
            used_gt = set()
            order = np.argsort(-res.boxes.conf.cpu().numpy())
            for pi in order:
                best_gi = int(np.argmax(iou_mat[pi]))
                best_iou = float(iou_mat[pi, best_gi])
                if best_iou >= 0.5 and best_gi not in used_gt:
                    matched_ious.append(best_iou)
                    used_gt.add(best_gi)

    mean_iou = float(np.mean(matched_ious)) if matched_ious else 0.0
    avg_inf_ms = float(np.mean(inference_times_ms)) if inference_times_ms else 0.0
    fps = 1000.0 / avg_inf_ms if avg_inf_ms > 0 else 0.0
    return mean_iou, avg_inf_ms, fps, len(test_images)


def plot_training_curves(results_csv: Path, out_path: Path):
    df = pd.read_csv(results_csv)
    df.columns = [c.strip() for c in df.columns]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Loss curves
    axes[0].plot(df["epoch"], df["train/box_loss"], label="Train Box Loss", marker="o")
    axes[0].plot(df["epoch"], df["val/box_loss"], label="Val Box Loss", marker="s")
    axes[0].plot(df["epoch"], df["train/cls_loss"], label="Train Cls Loss", linestyle="--")
    axes[0].plot(df["epoch"], df["val/cls_loss"], label="Val Cls Loss", linestyle="--")
    axes[0].set_title("Eğitim ve Doğrulama Kayıp (Loss) Eğrileri")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    # Metric curves
    axes[1].plot(df["epoch"], df["metrics/mAP50(B)"], label="mAP@0.5", marker="o")
    axes[1].plot(df["epoch"], df["metrics/mAP50-95(B)"], label="mAP@0.5:0.95", marker="s")
    axes[1].plot(df["epoch"], df["metrics/precision(B)"], label="Precision", linestyle="--")
    axes[1].plot(df["epoch"], df["metrics/recall(B)"], label="Recall", linestyle="--")
    axes[1].set_title("Doğrulama Performans Metrikleri")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Skor")
    axes[1].set_ylim(0, 1.05)
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_sample_predictions(model: YOLO, out_path: Path):
    test_images = sorted(TEST_IMG_DIR.glob("*.jpg"))[:6]
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for ax, img_path in zip(axes, test_images):
        img_rgb = np.array(Image.open(img_path).convert("RGB"))
        res = model.predict(
            source=img_rgb,
            imgsz=CONFIG["imgsz"],
            conf=CONFIG["conf_eval"],
            iou=CONFIG["iou_eval"],
            verbose=False,
        )[0]

        vis = img_rgb.copy()
        count = 0
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            count = len(boxes)
            for box, conf in zip(boxes, confs):
                x1, y1, x2, y2 = [int(round(v)) for v in box]
                cv2.rectangle(vis, (x1, y1), (x2, y2), (16, 185, 129), 3)
                cv2.putText(
                    vis,
                    f"{conf:.2f}",
                    (x1, max(y1 - 6, 18)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (16, 185, 129),
                    2,
                    cv2.LINE_AA,
                )

        ax.imshow(vis)
        ax.set_title(f"{img_path.name} — {count} yüz")
        ax.axis("off")

    plt.suptitle("Test Kümesinden Örnek Yüz Tespiti Tahminleri (YOLOv8n)", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    if not DATA_YAML.exists():
        raise FileNotFoundError(
            f"{DATA_YAML} bulunamadı. Önce `python prepare_dataset.py` çalıştırın."
        )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Eğitim cihazı: {device} | CPU çekirdek sayısı: {os.cpu_count()}")

    model = YOLO(str(BASE_DIR / "yolov8n.pt"))

    print("\n[1/4] YOLOv8n modeli eğitiliyor...")
    t_start = time.perf_counter()
    train_results = model.train(
        data=str(DATA_YAML),
        epochs=CONFIG["epochs"],
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        optimizer=CONFIG["optimizer"],
        lr0=CONFIG["lr0"],
        lrf=CONFIG["lrf"],
        weight_decay=CONFIG["weight_decay"],
        warmup_epochs=CONFIG["warmup_epochs"],
        seed=CONFIG["seed"],
        device=device,
        workers=0,
        project=str(RUNS_DIR),
        name="face_yolov8n",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    training_time_sec = time.perf_counter() - t_start
    print(f"\nEğitim tamamlandı! Toplam süre: {training_time_sec:.2f} saniye")

    save_dir = Path(train_results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    dst_best_pt = MODELS_DIR / "face_detector_best.pt"
    shutil.copy2(best_pt, dst_best_pt)
    print(f"En iyi model kaydedildi: {dst_best_pt}")

    print("\n[2/4] Test kümesi üzerinde değerlendirme yapılıyor...")
    best_model = YOLO(str(dst_best_pt))
    val_res = best_model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=CONFIG["imgsz"],
        batch=CONFIG["batch"],
        device=device,
        workers=0,
        verbose=False,
    )

    precision = float(val_res.box.mp)
    recall = float(val_res.box.mr)
    map50 = float(val_res.box.map50)
    map50_95 = float(val_res.box.map)
    f1_score = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    print("[3/4] Test kümesinde Ortalama IoU ve Tahmin Süresi ölçülüyor...")
    mean_iou, avg_inf_ms, fps, num_test_imgs = evaluate_mean_iou_and_speed(
        best_model, CONFIG["conf_eval"], CONFIG["iou_eval"]
    )

    print("[4/4] Grafikler ve metrik raporu oluşturuluyor...")
    results_csv = save_dir / "results.csv"
    if results_csv.exists():
        plot_training_curves(results_csv, OUTPUTS_DIR / "training_curves.png")
    plot_sample_predictions(best_model, OUTPUTS_DIR / "sample_predictions.png")

    metrics_payload = {
        "config": CONFIG,
        "environment": {
            "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version(),
            "pytorch": torch.__version__,
            "ultralytics": ultralytics.__version__,
            "device": "GPU (CUDA)" if torch.cuda.is_available() else f"CPU ({os.cpu_count()} threads)",
            "cpu_model": platform.processor(),
        },
        "training_time_sec": round(training_time_sec, 2),
        "inference_time_ms": round(avg_inf_ms, 2),
        "inference_fps": round(fps, 2),
        "num_test_images": num_test_imgs,
        "test_metrics": {
            "mAP50": round(map50, 4),
            "mAP50_95": round(map50_95, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1_score, 4),
            "mean_iou": round(mean_iou, 4),
        },
    }

    metrics_path = OUTPUTS_DIR / "metrics.json"
    metrics_path.write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n=== TEST SONUÇLARI ===")
    print(json.dumps(metrics_payload["test_metrics"], indent=2, ensure_ascii=False))
    print(f"Eğitim Süresi: {training_time_sec:.2f} sn")
    print(f"Tahmin Süresi: {avg_inf_ms:.2f} ms/görüntü ({fps:.1f} FPS)")


if __name__ == "__main__":
    main()
