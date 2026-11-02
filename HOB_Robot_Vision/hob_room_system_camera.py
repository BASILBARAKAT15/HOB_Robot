r"""HOB Robot: room plaque detection on an image or Windows USB camera.

Uses the original 3-threshold contour search and repeated Tesseract reads.
Optionally falls back to EasyOCR when Tesseract cannot confirm a room code.

Windows examples:
  .\.venv\Scripts\python.exe hob_room_system_camera.py --image images\G0070.jpg
  .\.venv\Scripts\python.exe hob_room_system_camera.py --camera 0
  .\.venv\Scripts\python.exe hob_room_system_camera.py --camera 0 --engine easyocr

Camera keys: Q/Esc quit, P save raw frame, D show/hide candidates.
"""

import argparse
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytesseract

BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "output" / "room_system"
OUTPUT.mkdir(parents=True, exist_ok=True)
ROOM = re.compile(r"^(?:B[12]|G0|[1-4])[0-9]{3}$")

if os.name == "nt":
    tesseract_exe = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if tesseract_exe.exists():
        pytesseract.pytesseract.tesseract_cmd = str(tesseract_exe)


def room_parts(code):
    floor = code[:-3]
    return ("G" if floor == "G0" else floor), code[-3:]


def clean_room(text):
    text = re.sub(r"[^A-Z0-9]", "", str(text).upper())
    return text if ROOM.fullmatch(text) else None


def candidates_from_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    boxes = []
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Original detection approach: three thresholds, a close, and contours.
    for threshold in (95, 75, 115):
        mask = cv2.inRange(gray, 0, threshold)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if not (
                max(30, W * 0.012) <= w <= W * 0.3
                and max(12, H * 0.012) <= h <= H * 0.13
                and 1.3 <= w / max(h, 1) <= 7.5
            ):
                continue
            duplicate = any(
                abs(x - bx) <= 8 and abs(y - by) <= 8
                and abs(w - bw) <= 12 and abs(h - bh) <= 12
                for bx, by, bw, bh in boxes
            )
            if not duplicate:
                boxes.append((x, y, w, h))
    boxes.sort(key=lambda b: (b[1], -(b[2] * b[3])))
    return gray, boxes


def check_candidate(gray, box):
    H, W = gray.shape
    x, y, w, h = box
    x0 = max(0, x - max(4, round(w * 0.06)))
    y0 = max(0, y + max(1, round(h * 0.03)))
    x1 = min(W, x + w + max(4, round(w * 0.06)))
    y1 = min(H, y + h + max(2, round(h * 0.08)))
    crop = gray[y0:y1, x0:x1]
    if crop.size == 0:
        return None, [], None

    up = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    _, bw = cv2.threshold(up, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    attempts = []
    for processed in (up, bw):
        for psm in (7, 6):
            txt = pytesseract.image_to_string(
                processed,
                lang="eng",
                config=f"--psm {psm} -c tessedit_char_whitelist=BG0123456789",
            )
            clean = re.sub(r"[^A-Z0-9]", "", txt.upper())
            if clean:
                attempts.append(clean)

    # Require the SAME valid code twice, not just two different valid codes.
    counts = Counter(x for x in attempts if ROOM.fullmatch(x))
    confirmed = counts.most_common(1)[0][0] if counts and counts.most_common(1)[0][1] >= 2 else None
    return confirmed, attempts, up


class RoomScanner:
    def __init__(self, engine="hybrid", max_candidates=12, debug=False):
        self.engine = engine
        self.max_candidates = max_candidates
        self.debug = debug
        self.easy_reader = None
        if engine in ("easyocr", "hybrid"):
            import easyocr
            import torch
            gpu = torch.cuda.is_available()
            print(f"Loading EasyOCR (CUDA: {gpu}) ...", flush=True)
            self.easy_reader = easyocr.Reader(["en"], gpu=gpu)

    def scan(self, image):
        started = time.monotonic()
        gray, boxes = candidates_from_image(image)
        found = None

        if self.engine in ("hybrid", "tesseract"):
            for i, box in enumerate(boxes[:self.max_candidates], 1):
                code, attempts, crop = check_candidate(gray, box)
                if self.debug:
                    print(f"Region {i} {box}: {attempts}", flush=True)
                if code:
                    found = {"code": code, "box": (box[0], box[1], box[0]+box[2], box[1]+box[3]),
                             "method": "OpenCV + Tesseract", "crop": crop}
                    break

        if found is None and self.easy_reader is not None:
            # Full-frame text detector: helpful when a dark plaque is not found.
            results = self.easy_reader.readtext(
                image, detail=1, allowlist="BG0123456789"
            )
            for box, text, confidence in sorted(results, key=lambda item: item[2], reverse=True):
                code = clean_room(text)
                if self.debug:
                    print(f"EasyOCR: {text!r} ({confidence:.2f})", flush=True)
                if code and confidence >= 0.35:
                    points = np.asarray(box, dtype=np.int32)
                    x0, y0 = points.min(axis=0).tolist()
                    x1, y1 = points.max(axis=0).tolist()
                    found = {"code": code, "box": (x0, y0, x1, y1),
                             "method": "EasyOCR", "crop": None}
                    break

        return {"found": found, "boxes": boxes, "seconds": time.monotonic()-started}


def annotate(image, result, draw_candidates=True):
    marked = image.copy()
    if draw_candidates:
        for x, y, w, h in result["boxes"][:50]:
            cv2.rectangle(marked, (x, y), (x+w, y+h), (0, 180, 255), 2)
    found = result["found"]
    if found:
        x0, y0, x1, y1 = found["box"]
        cv2.rectangle(marked, (x0, y0), (x1, y1), (0, 255, 0), 3)
        cv2.putText(marked, found["code"], (x0, max(24, y0-8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return marked


def print_room(found, prefix="ROOM"):
    floor, number = room_parts(found["code"])
    print(f"{prefix}: {found['code']} | FLOOR: {floor} | NUMBER: {number} | METHOD: {found['method']}", flush=True)


def run_image(scanner, path):
    img = cv2.imread(str(path))
    if img is None:
        raise SystemExit(f"Cannot open image: {path}")
    print(f"Image: {img.shape[1]} x {img.shape[0]}")
    result = scanner.scan(img)
    print(f"Possible plaques: {len(result['boxes'])} | Scan: {result['seconds']:.2f}s")
    if result["found"]:
        print_room(result["found"])
        if result["found"]["crop"] is not None:
            cv2.imwrite(str(OUTPUT / "sign_crop.png"), result["found"]["crop"])
    else:
        print("No reliable room code found")
    out = OUTPUT / "image_result.jpg"
    cv2.imwrite(str(out), annotate(img, result))
    print("Saved:", out)


def open_camera(index, width, height):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return cap
        cap.release()
    raise SystemExit(f"Cannot open camera {index}; try --camera 1 or 2")


def run_camera(scanner, index, width, height, scan_interval):
    cap = open_camera(index, width, height)
    print(f"Camera {index}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
          f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("Q/Esc: quit | P: save frame | D: candidate rectangles")
    pending = None
    last_submit = 0.0
    latest = None
    latest_time = 0.0
    last_code = None
    streak = 0
    reported = None
    show_candidates = True

    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame read failed")
                    break
                now = time.monotonic()

                if pending is not None and pending.done():
                    try:
                        latest = pending.result()
                        latest_time = now
                        found = latest["found"]
                        code = found["code"] if found else None
                        streak = streak + 1 if code and code == last_code else (1 if code else 0)
                        last_code = code
                        if code and streak >= 2 and reported != code:
                            print_room(found, "CONFIRMED ROOM")
                            reported = code
                        if code is None:
                            reported = None
                    except Exception as exc:
                        print("OCR error:", exc)
                    pending = None

                if pending is None and now - last_submit >= scan_interval:
                    pending = pool.submit(scanner.scan, frame.copy())
                    last_submit = now

                display = frame.copy()
                if latest and now - latest_time < 2.5:
                    display = annotate(display, latest, draw_candidates=show_candidates)
                if last_code and streak >= 2:
                    floor, num = room_parts(last_code)
                    title = f"ROOM {last_code} | FLOOR {floor} | NO {num}"
                else:
                    title = "Searching for room signs..."
                cv2.putText(display, title, (14, 34),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 0), 2)
                ratio = min(1.0, 1100 / display.shape[1], 750 / display.shape[0])
                if ratio < 1.0:
                    display = cv2.resize(display, None, fx=ratio, fy=ratio)
                cv2.imshow("HOB Robot - Room System (Q/P/D)", display)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("d"):
                    show_candidates = not show_candidates
                if key == ord("p"):
                    saved = OUTPUT / f"capture_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(saved), frame)
                    print("Saved raw camera frame:", saved)
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="HOB Robot - Room number system")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--image", type=Path, help="Test a saved image")
    source.add_argument("--camera", type=int, help="Windows camera index (default: 0)")
    parser.add_argument("--engine", choices=("hybrid", "tesseract", "easyocr"), default="hybrid")
    parser.add_argument("--max-candidates", type=int, default=12)
    parser.add_argument("--scan-interval", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--debug", action="store_true", help="Print OCR readings for each region")
    args = parser.parse_args()
    scanner = RoomScanner(args.engine, max(1, args.max_candidates), args.debug)
    if args.image:
        run_image(scanner, args.image)
    else:
        run_camera(scanner, args.camera if args.camera is not None else 0,
                   args.width, args.height, max(0.0, args.scan_interval))

if __name__ == "__main__":
    main()
