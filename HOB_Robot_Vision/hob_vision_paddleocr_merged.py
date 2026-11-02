r"""HOB Robot Vision - merged PaddleOCR + OpenCV/Tesseract for Windows.

Examples (PowerShell in HOB_Robot_Vision):
  .\.venv\Scripts\python.exe .\hob_vision_paddleocr_merged.py --camera 0
  .\.venv\Scripts\python.exe .\hob_vision_paddleocr_merged.py --image .\images\G0070.jpg
  .\.venv\Scripts\python.exe .\hob_vision_paddleocr_merged.py --image .\dataset\images\val\room_003.jpeg --roi-label .\dataset\labels\val\room_003.txt

Live keys: Q/Esc exit | P save original full-resolution frame | S toggle light stabilization
PaddleOCR detects text first; OpenCV/Tesseract is fallback in hybrid mode.
The program recognizes room codes B2030, B1030, G0070, 1030, 2870, 3030, 4030.
Recognition from 3 m is experimental, not guaranteed.
"""

import argparse
import re
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "test_3m"
OUTPUT.mkdir(parents=True, exist_ok=True)
TESSERACT_WINDOWS = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
ROOM_PATTERN = re.compile(r"(?:B[12]|G0|[1-4])[0-9]{3}")


def normalize_code(text):
    """Keep the UNIVERSITY format, including leading zeroes in the room number."""
    chars = re.sub(r"[^A-Z0-9]", "", text.upper())
    if chars.startswith(("B", "G")):
        prefix, digits = chars[0], chars[1:]
        digits = digits.translate(str.maketrans({"O": "0", "I": "1", "L": "1"}))
        chars = prefix + digits
    else:
        chars = chars.translate(str.maketrans({"O": "0", "I": "1", "L": "1"}))
    return chars if ROOM_PATTERN.fullmatch(chars) else None


def room_parts(code):
    floor = code[:-3]
    return ("G" if floor == "G0" else floor), code[-3:]


def sharpness(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    scale = min(1.0, 640.0 / gray.shape[1])
    if scale < 1:
        gray = cv2.resize(gray, None, fx=scale, fy=scale)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def paddle_items_v3(result):
    """Extract PaddleOCR 3.x OCR detections from PaddleX-style prediction results."""
    if isinstance(result, dict):
        data = result
    else:
        data = getattr(result, "json", None)
        if callable(data):
            data = data()
        if isinstance(data, str):
            import json
            data = json.loads(data)
        if not isinstance(data, dict):
            try:
                data = dict(result)
            except (TypeError, ValueError):
                return []
    if isinstance(data.get("res"), dict):
        data = data["res"]

    texts = data.get("rec_texts", [])
    scores = data.get("rec_scores", [])
    polygons = data.get("rec_polys", None)
    if polygons is None:
        polygons = data.get("dt_polys", [])
    rectangles = data.get("rec_boxes", [])
    items = []
    for i, text in enumerate(texts):
        conf = float(scores[i]) if i < len(scores) else 0.0
        poly = polygons[i] if i < len(polygons) else None
        if poly is not None and np.asarray(poly).size:
            pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
            lo, hi = np.min(pts, axis=0), np.max(pts, axis=0)
            box = tuple(int(v) for v in (lo[0], lo[1], hi[0], hi[1]))
        elif i < len(rectangles):
            box = tuple(int(v) for v in rectangles[i][:4])
        else:
            continue
        items.append((str(text), box, conf))
    return items


def paddle_items_v2(results):
    """Extract PaddleOCR 2.x: [[[polygon, (text, score)], ...]]."""
    items = []
    if not results:
        return items
    # Typical 2.x output is nested per image; tolerate one-level output too.
    groups = results
    if isinstance(groups, list) and groups:
        first = groups[0]
        if isinstance(first, (list, tuple)) and len(first) == 2 and isinstance(first[1], (list, tuple)) and len(first[1]) == 2 and isinstance(first[1][0], str):
            groups = [groups]
    for group in groups:
        if not group:
            continue
        for entry in group:
            try:
                poly, (text, conf) = entry
                pts = np.asarray(poly, dtype=np.float32).reshape(-1, 2)
                lo, hi = np.min(pts, axis=0), np.max(pts, axis=0)
                box = tuple(int(v) for v in (lo[0], lo[1], hi[0], hi[1]))
                items.append((str(text), box, float(conf)))
            except (TypeError, ValueError, IndexError):
                continue
    return items


def paddleocr_results(reader, image, enhance=False):
    """Detect and recognize signs with PaddleOCR; supports 3.x and legacy 2.x."""
    variants = [image]
    if enhance:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(l)
        variants.append(cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR))

    candidates, raw_text = [], []
    for frame in variants:
        if hasattr(reader, "predict"):
            results = reader.predict(frame)
            tokens = []
            for result in results:
                tokens.extend(paddle_items_v3(result))
        else:
            results = reader.ocr(frame, cls=False)
            tokens = paddle_items_v2(results)

        for text, box, confidence in tokens:
            raw_text.append((text, round(confidence, 3)))
            code = normalize_code(text)
            if code:
                candidates.append((code, box, confidence, "PaddleOCR"))

        # Sometimes OCR separates a room number into two adjacent fragments.
        for i, (left_text, left, left_conf) in enumerate(tokens):
            x0, y0, x1, y1 = left
            h = max(1, y1 - y0)
            for j, (right_text, right, right_conf) in enumerate(tokens):
                if i == j:
                    continue
                rx0, ry0, rx1, ry1 = right
                gap = rx0 - x1
                overlap = min(y1, ry1) - max(y0, ry0)
                if -0.15*h <= gap <= 1.0*h and overlap >= 0.50*min(h, ry1-ry0):
                    code = normalize_code(left_text + right_text)
                    if code:
                        merged = (min(x0,rx0), min(y0,ry0), max(x1,rx1), max(y1,ry1))
                        candidates.append((code, merged, min(left_conf,right_conf), "PaddleOCR merged"))
        if candidates:
            break
    candidates.sort(key=lambda row: row[2], reverse=True)
    return candidates, raw_text

def tesseract_ready():
    return TESSERACT_WINDOWS.is_file()


def tesseract_read(crop):
    if not tesseract_ready() or crop.size == 0:
        return None
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_WINDOWS)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    zoom = min(9, max(2, round(500 / max(gray.shape[1], 1))))
    big = cv2.resize(gray, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(big)
    blurred = cv2.GaussianBlur(clahe, (0, 0), 1.0)
    sharp = cv2.addWeighted(clahe, 1.5, blurred, -0.5, 0)
    bw = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    settings = "--oem 3 --psm 7 -c tessedit_char_whitelist=BG0123456789"
    readings = []
    for variant in (big, clahe, sharp, bw, cv2.bitwise_not(bw)):
        text = pytesseract.image_to_string(variant, lang="eng", config=settings)
        readings.append(re.sub(r"[^A-Z0-9]", "", text.upper()))
        code = normalize_code(text)
        if code:
            return code
    # Preserve the original script's recovery when G/B is read separately.
    for prefix in ("B", "G"):
        if prefix in readings:
            for digits in readings:
                if digits.isascii() and digits.isdigit():
                    code = normalize_code(prefix + digits)
                    if code:
                        return code
    return None


def contour_fallback(image, max_candidates):
    """Original OpenCV sign proposals + Tesseract reading, only if EasyOCR fails."""
    if not tesseract_ready():
        return []
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY_INV)[1]
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3)),
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, bw, bh = cv2.boundingRect(contour)
        ratio = bw / max(bh, 1)
        area = bw * bh / (w * h)
        if bw >= 25 and bh >= 9 and 1.6 < ratio < 10 and 0.0001 < area < 0.12:
            boxes.append((x, y, bw, bh))
    boxes.sort(key=lambda b: (b[1], -(b[2] * b[3])))
    found = []
    for x, y, bw, bh in boxes[:max_candidates]:
        pad = max(3, round(bh * 0.08))
        box = (max(0, x - pad), max(0, y - pad), min(w, x + bw + pad), min(h, y + bh + pad))
        x0, y0, x1, y1 = box
        code = tesseract_read(image[y0:y1, x0:x1])
        if code:
            found.append((code, box, 0.0, "OpenCV + Tesseract"))
            break
    return found


def crop_from_yolo_label(image, label_path):
    if not label_path.is_file():
        raise SystemExit(f"YOLO label not found: {label_path}")
    h, w = image.shape[:2]
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5 or parts[0] != "0":
            continue
        xc, yc, bw, bh = map(float, parts[1:])
        x0 = max(0, round((xc - bw * 0.56) * w))
        y0 = max(0, round((yc - bh * 0.56) * h))
        x1 = min(w, round((xc + bw * 0.56) * w))
        y1 = min(h, round((yc + bh * 0.56) * h))
        if x1 > x0 and y1 > y0:
            return image[y0:y1, x0:x1], (x0, y0, x1, y1)
    raise SystemExit("No valid class-0 room_sign label found")


def detect_room(reader, frame, engine, max_candidates=5, enhance=False):
    """Returns (detections, raw_ocr_tokens)."""
    candidates, raw = ([], [])
    if engine in ("paddleocr", "hybrid"):
        candidates, raw = paddleocr_results(reader, frame, enhance=enhance)
    if not candidates and engine in ("tesseract", "hybrid"):
        candidates = contour_fallback(frame, max_candidates)
    return candidates, raw


def load_paddleocr(engine):
    if engine == "tesseract":
        return None
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise SystemExit(
            "PaddleOCR is missing. In your Paddle environment, install "
            "paddlepaddle and paddleocr."
        ) from exc
    print("Loading PaddleOCR models (first launch may download them)...", flush=True)
    try:
        # PaddleOCR 3.x API. CPU by default for predictable Windows setup.
        return PaddleOCR(
            lang="en", device="cpu",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except TypeError:
        # PaddleOCR 2.x has different initialization flags.
        return PaddleOCR(lang="en", use_angle_cls=False, use_gpu=False, show_log=False)

def image_test(reader, path, engine, max_candidates, enhance, roi_label):
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"Cannot load image: {path}")
    print(f"IMAGE: {path} ({image.shape[1]}x{image.shape[0]})")
    if roi_label:
        crop, original_box = crop_from_yolo_label(image, roi_label)
        path_out = OUTPUT / "image_test_roi.jpg"
        cv2.imwrite(str(path_out), crop)
        print(f"Saved annotated plaque crop: {path_out}")
        found, raw = detect_room(reader, crop, engine, max_candidates, enhance)
        if not found and engine in ("hybrid", "tesseract"):
            code = tesseract_read(crop)
            if code:
                found = [(code, (0, 0, crop.shape[1], crop.shape[0]), 0.0, "Tesseract ROI")]
        # Convert ROI coordinates back to coordinates in original image.
        xoff, yoff = original_box[:2]
        found = [(code, (b[0]+xoff, b[1]+yoff, b[2]+xoff, b[3]+yoff),
                  conf, method) for code, b, conf, method in found]
    else:
        found, raw = detect_room(reader, image, engine, max_candidates, enhance)
    if raw:
        print("PaddleOCR raw text:", raw[:25])
    if found:
        for code, box, conf, method in found[:5]:
            floor, number = room_parts(code)
            print(f"ROOM: {code} | FLOOR: {floor} | NUMBER: {number} | METHOD: {method}")
            x0, y0, x1, y1 = box
            cv2.rectangle(image, (x0, y0), (x1, y1), (0, 255, 0), 3)
            cv2.putText(image, code, (x0, max(32, y0-8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    else:
        print("No validated room code detected in this image.")
    out = OUTPUT / "image_test_result.jpg"
    cv2.imwrite(str(out), image)
    print(f"Saved: {out}")


class LightStabilizer:
    """Stabilize small translations; does not restore blurred frames."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.prev = None
        self.position = np.zeros(2, dtype=np.float64)
        self.smoothed = np.zeros(2, dtype=np.float64)

    def apply(self, frame):
        h, w = frame.shape[:2]
        size = (max(1, int(w*0.33)), max(1, int(h*0.33)))
        tiny = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size)
        if self.prev is None:
            self.prev = tiny
            return frame
        pts = cv2.goodFeaturesToTrack(self.prev, maxCorners=100,
                                      qualityLevel=0.025, minDistance=12)
        if pts is None or len(pts) < 8:
            self.prev = tiny
            return frame
        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(self.prev, tiny, pts, None)
        self.prev = tiny
        if new_pts is None or status is None:
            return frame
        valid = status.reshape(-1) == 1
        if valid.sum() < 8:
            return frame
        move = np.median((new_pts[valid] - pts[valid]).reshape(-1, 2), axis=0)
        move *= w / size[0]
        if not np.all(np.isfinite(move)) or np.linalg.norm(move) > 55:
            self.position[:] = 0
            self.smoothed[:] = 0
            return frame
        self.position += move
        self.smoothed += 0.20*(self.position - self.smoothed)
        correction = np.clip(self.smoothed - self.position, -25, 25)
        matrix = np.float32([[1, 0, correction[0]], [0, 1, correction[1]]])
        return cv2.warpAffine(frame, matrix, (w, h),
                              borderMode=cv2.BORDER_REFLECT)


def open_camera(index, width, height, fps):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            ok, _ = cap.read()
            if ok:
                return cap
        cap.release()
    raise SystemExit(f"Cannot open camera index {index}; try --camera 0, 1 or 2")


def run_camera(reader, index, width, height, fps, engine, interval, max_candidates, enhance, stable, debug=False):
    cap = open_camera(index, width, height, fps)
    print(f"Camera {index}: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print("Q quit | P save raw frame | S stabilization on/off")
    last_request = 0.0
    pending = None
    scan_started = 0.0
    last_scan_duration = None
    scan_ticks = deque(maxlen=100)
    stream_ticks = deque(maxlen=100)
    previous = None
    streak = 0
    last_reported = None
    active_detections = []
    active_since = 0.0
    confirmed_code = None
    confirmed_at = 0.0
    stabilizer = LightStabilizer()
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            while True:
                ok, raw = cap.read()
                if not ok:
                    print("Camera read failed")
                    break
                frame = stabilizer.apply(raw) if stable else raw.copy()
                now = time.monotonic()
                stream_ticks.append(now)
                if pending is not None and pending.done():
                    try:
                        detections, raw_text = pending.result()
                        if debug:
                            print("OCR RAW:", raw_text[:12])
                        last_scan_duration = now - scan_started
                        scan_ticks.append(now)
                        if detections:
                            active_detections = detections[:3]
                            active_since = now
                            code = detections[0][0]
                            streak = streak + 1 if code == previous else 1
                            previous = code
                            if streak >= 2:
                                confirmed_code = code
                                confirmed_at = now
                                if code != last_reported:
                                    floor, number = room_parts(code)
                                    print(f"CONFIRMED: {code} | FLOOR: {floor} | ROOM: {number}")
                                    last_reported = code
                        else:
                            previous = None
                            streak = 0
                            active_detections = []
                    except Exception as exc:
                        print("OCR error:", exc)
                    pending = None
                if pending is None and now - last_request >= interval:
                    # Only one job can run at once. Avoid accumulating stale frames.
                    pending = executor.submit(detect_room, reader, frame.copy(),
                                              engine, max_candidates, enhance)
                    scan_started = now
                    last_request = now
                scale = min(1.0, 1100 / frame.shape[1])
                display = cv2.resize(frame, None, fx=scale, fy=scale)
                if now - active_since < 3:
                    for code, box, conf, method in active_detections:
                        x0, y0, x1, y1 = (round(v*scale) for v in box)
                        cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)
                        cv2.putText(display, f"{code} ({method})", (x0, max(28, y0-7)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
                status = f"Room: {confirmed_code}" if confirmed_code and now-confirmed_at < 5 else "Searching for room signs..."
                cv2.putText(display, status, (14, 34), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (0, 255, 0) if confirmed_code and now-confirmed_at < 5 else (255,255,255), 2)
                count = lambda stamps: sum(t >= now - 5 for t in stamps) / 5.0
                footer = f"Video ~{count(stream_ticks):.1f} fps | OCR ~{count(scan_ticks):.2f} scans/s | Last {last_scan_duration or 0:.2f}s"
                cv2.putText(display, footer, (10, display.shape[0] - 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                cv2.imshow("HOB Robot | PaddleOCR + Tesseract", display)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    path = OUTPUT / f"capture_3m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(path), raw)
                    print("Saved full-resolution photo:", path)
                if key == ord("s"):
                    stable = not stable
                    stabilizer.reset()
                    print("Stabilization:", "ON" if stable else "OFF")
        finally:
            cap.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="HOB Robot merged PaddleOCR + OpenCV/Tesseract")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scan-interval", type=float, default=0.5)
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--engine", choices=("hybrid", "paddleocr", "tesseract"), default="hybrid")
    parser.add_argument("--enhance", action="store_true", help="If PaddleOCR fails, try CLAHE-enhanced picture")
    parser.add_argument("--stabilize", action="store_true", help="Enable light stabilization (S toggles during video)")
    parser.add_argument("--debug", action="store_true", help="Print PaddleOCR raw readings in live mode")
    parser.add_argument("--image", type=Path)
    parser.add_argument("--roi-label", type=Path)
    args = parser.parse_args()
    if args.roi_label and not args.image:
        parser.error("--roi-label requires --image")
    if args.engine == "tesseract" and not tesseract_ready():
        parser.error(f"Tesseract not found: {TESSERACT_WINDOWS}")
    if args.engine == "hybrid" and not tesseract_ready():
        print("Note: Tesseract not installed; PaddleOCR will still work.")
    reader = load_paddleocr(args.engine)
    if args.image:
        image_test(reader, args.image, args.engine, args.max_candidates, args.enhance, args.roi_label)
    else:
        run_camera(reader, args.camera, args.width, args.height, args.fps,
                   args.engine, max(0.0, args.scan_interval),
                   max(1, args.max_candidates), args.enhance, args.stabilize, args.debug)


if __name__ == "__main__":
    main()
