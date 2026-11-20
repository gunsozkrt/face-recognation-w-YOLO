"""
WIDER FACE Veri Seti İndirme ve YOLO Formatına Dönüştürme Betiği
- Kaynak: CUHK-CSE/wider_face (Hugging Face Hub)
- Çıktı: data/images/{train,val,test}, data/labels/{train,val,test}, data/dataset.yaml, data/dataset_stats.json
"""

import json
import random
import shutil
import zipfile
from pathlib import Path

from huggingface_hub import hf_hub_download
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
DATA_DIR = BASE_DIR / "data"

SEED = 42
TRAIN_COUNT = 600
VAL_COUNT = 150
TEST_COUNT = 150
MIN_FACE_SIZE_PX = 16
MAX_FACES_PER_IMG = 12


def download_and_extract():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/4] Hugging Face Hub üzerinden WIDER FACE arşivleri indiriliyor...")
    val_zip_path = hf_hub_download(
        repo_id="CUHK-CSE/wider_face",
        repo_type="dataset",
        filename="data/WIDER_val.zip",
    )
    split_zip_path = hf_hub_download(
        repo_id="CUHK-CSE/wider_face",
        repo_type="dataset",
        filename="data/wider_face_split.zip",
    )

    wider_val_dir = RAW_DIR / "WIDER_val"
    split_dir = RAW_DIR / "wider_face_split"

    if not wider_val_dir.exists():
        print("[2/4] WIDER_val.zip çıkarılıyor...")
        with zipfile.ZipFile(val_zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
    else:
        print("[2/4] WIDER_val zaten mevcut, atlanıyor.")

    if not split_dir.exists():
        print("[2/4] wider_face_split.zip çıkarılıyor...")
        with zipfile.ZipFile(split_zip_path, "r") as zf:
            zf.extractall(RAW_DIR)
    else:
        print("[2/4] wider_face_split zaten mevcut, atlanıyor.")

    return wider_val_dir / "images", split_dir / "wider_face_val_bbx_gt.txt"


def parse_wider_annotations(gt_file: Path, images_root: Path):
    """WIDER FACE ground truth dosyasını okuyup geçerli yüz kutularını döner."""
    lines = [line.strip() for line in gt_file.read_text(encoding="utf-8").splitlines()]
    records = []
    i = 0
    n = len(lines)

    while i < n:
        rel_path = lines[i]
        i += 1
        if i >= n:
            break
        num_boxes = int(lines[i])
        i += 1

        raw_boxes = []
        if num_boxes == 0:
            # WIDER FACE formatı 0 yüzlü görüntülerde 1 dummy satır içerir
            i += 1
        else:
            for _ in range(num_boxes):
                if i >= n:
                    break
                parts = list(map(int, lines[i].split()))
                i += 1
                if len(parts) >= 4:
                    x, y, w, h = parts[:4]
                    invalid = parts[7] if len(parts) >= 8 else 0
                    if invalid == 0 and w >= MIN_FACE_SIZE_PX and h >= MIN_FACE_SIZE_PX:
                        raw_boxes.append((x, y, w, h))

        img_path = images_root / rel_path
        if img_path.exists() and 1 <= len(raw_boxes) <= MAX_FACES_PER_IMG:
            records.append((img_path, raw_boxes))

    return records


def write_yolo_split(split_name: str, items: list):
    img_out_dir = DATA_DIR / "images" / split_name
    lbl_out_dir = DATA_DIR / "labels" / split_name
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    total_images = 0
    total_faces = 0

    for idx, (src_img_path, boxes) in enumerate(items):
        with Image.open(src_img_path) as im:
            img_w, img_h = im.size

        yolo_lines = []
        for x, y, w, h in boxes:
            x1 = max(0.0, float(x))
            y1 = max(0.0, float(y))
            x2 = min(float(img_w), float(x + w))
            y2 = min(float(img_h), float(y + h))
            bw = x2 - x1
            bh = y2 - y1
            if bw < 4.0 or bh < 4.0:
                continue
            xc = (x1 + x2) / 2.0 / img_w
            yc = (y1 + y2) / 2.0 / img_h
            wn = bw / img_w
            hn = bh / img_h
            yolo_lines.append(f"0 {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}")

        if not yolo_lines:
            continue

        out_stem = f"{split_name}_{idx:04d}"
        dst_img = img_out_dir / f"{out_stem}.jpg"
        dst_lbl = lbl_out_dir / f"{out_stem}.txt"

        shutil.copy2(src_img_path, dst_img)
        dst_lbl.write_text("\n".join(yolo_lines) + "\n", encoding="utf-8")

        total_images += 1
        total_faces += len(yolo_lines)

    return {"images": total_images, "faces": total_faces}


def main():
    images_root, gt_file = download_and_extract()
    print("[3/4] Etiketler ayrıştırılıyor ve filtreleniyor...")
    records = parse_wider_annotations(gt_file, images_root)
    print(f"Uygun görüntü sayısı: {len(records)}")

    rng = random.Random(SEED)
    rng.shuffle(records)

    total_needed = TRAIN_COUNT + VAL_COUNT + TEST_COUNT
    if len(records) < total_needed:
        raise RuntimeError(
            f"Yeterli görüntü bulunamadı: {len(records)} < {total_needed}"
        )

    selected = records[:total_needed]
    train_items = selected[:TRAIN_COUNT]
    val_items = selected[TRAIN_COUNT : TRAIN_COUNT + VAL_COUNT]
    test_items = selected[TRAIN_COUNT + VAL_COUNT :]

    # Eski bölünmüş klasörleri temizle
    for sub in ("images", "labels"):
        p = DATA_DIR / sub
        if p.exists():
            shutil.rmtree(p)

    print("[4/4] Train / Val / Test kümeleri YOLO formatında kaydediliyor...")
    stats = {
        "dataset": "WIDER FACE (CUHK-CSE/wider_face)",
        "dataset_url": "http://shuoyang1213.me/WIDERFACE/",
        "hf_mirror": "https://huggingface.co/datasets/CUHK-CSE/wider_face",
        "seed": SEED,
        "filter": {
            "min_face_size_px": MIN_FACE_SIZE_PX,
            "max_faces_per_image": MAX_FACES_PER_IMG,
            "invalid_excluded": True,
        },
        "splits": {
            "train": write_yolo_split("train", train_items),
            "val": write_yolo_split("val", val_items),
            "test": write_yolo_split("test", test_items),
        },
    }

    yaml_content = (
        f"path: {DATA_DIR.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "test: images/test\n\n"
        "names:\n"
        "  0: face\n"
    )
    (DATA_DIR / "dataset.yaml").write_text(yaml_content, encoding="utf-8")
    (DATA_DIR / "dataset_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\nVeri seti başarıyla hazırlandı!")
    print(json.dumps(stats["splits"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
