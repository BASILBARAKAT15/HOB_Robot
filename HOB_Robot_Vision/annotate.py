
from pathlib import Path
import shutil
import cv2

BASE = Path(__file__).resolve().parent
IMAGES = BASE / "images"
DATASET = BASE / "dataset"

files = sorted(
    p for p in IMAGES.iterdir()
    if p.suffix.lower() in (".jpg", ".jpeg", ".png")
)

if not files:
    raise SystemExit("No images found in images folder")

for i, path in enumerate(files):
    image = cv2.imread(str(path))

    if image is None:
        print("Cannot read:", path.name)
        continue

    height, width = image.shape[:2]
    scale = min(1.0, 1200 / width, 800 / height)

    preview = cv2.resize(
        image, None,
        fx=scale, fy=scale
    )

    print("\nImage:", path.name)
    print("Select ONLY the room number sign")

    x, y, w, h = cv2.selectROI(
        "YOLO Annotation - " + path.name,
        preview,
        showCrosshair=True,
        fromCenter=False
    )

    cv2.destroyAllWindows()

    if w == 0 or h == 0:
        print("Skipped image")
        continue

    preview_h, preview_w = preview.shape[:2]

    # YOLO normalized coordinates
    xc = (x + w / 2) / preview_w
    yc = (y + h / 2) / preview_h
    nw = w / preview_w
    nh = h / preview_h

    # Small demo split
    if len(files) < 5:
        split = (
            "val" if i == len(files) - 1
            and len(files) > 1 else "train"
        )
    else:
        split = "val" if (i + 1) % 5 == 0 else "train"

    image_dir = DATASET / "images" / split
    label_dir = DATASET / "labels" / split

    image_dir.mkdir(parents=True, exist_ok=True)
    label_dir.mkdir(parents=True, exist_ok=True)

    name = f"room_{i + 1:03d}"

    shutil.copy2(
        path,
        image_dir / (name + path.suffix.lower())
    )

    label = f"0 {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}"

    (label_dir / (name + ".txt")).write_text(
        label + "\n",
        encoding="utf-8"
    )

    print("Saved:", name, "|", split)
    print("YOLO label:", label)

yaml_path = DATASET / "data.yaml"

yaml_path.write_text(
    f"path: '{DATASET.as_posix()}'\n"
    "train: images/train\n"
    "val: images/val\n"
    "names:\n"
    "  0: room_sign\n",
    encoding="utf-8"
)

print("\nDataset annotation completed!")
print("Configuration:", yaml_path)
