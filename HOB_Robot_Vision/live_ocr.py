r"""HOB Robot Vision — faster 3-meter USB-camera OCR on Windows.

Requirements: opencv-python, numpy, pytesseract; Tesseract executable installed.
Example: .\.venv\Scripts\python.exe hob_vision_3m_windows.py --camera 1
Image regression: .\.venv\Scripts\python.exe hob_vision_3m_windows.py --image images\G0070.jpg
Keys in live view: Q/Esc quit, P save current full-resolution frame, S toggle stabilizer.
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
import pytesseract

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "output" / "test_3m"
OUTPUT.mkdir(parents=True, exist_ok=True)
TESSERACT_WINDOWS = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if TESSERACT_WINDOWS.exists():
    pytesseract.pytesseract.tesseract_cmd = str(TESSERACT_WINDOWS)

ROOM_PATTERN = re.compile(r"(?:B[12]|G0|[1-4])[0-9]{3}")


def room_parts(code):
    floor = code[:-3]
    return ("G" if floor == "G0" else floor), code[-3:]


def sharpness(image):
    """Relative image sharpness; compare only at the same resolution and scene."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    scale = min(1.0, 640 / max(w, 1))
    if scale < 1:
        gray = cv2.resize(gray, (round(w * scale), round(h * scale)))
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def read_room(crop):
    if crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    zoom = min(9, max(2, round(500 / max(gray.shape[1], 1))))
    up = cv2.resize(gray, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    contrast = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(up)
    blur = cv2.GaussianBlur(contrast, (0, 0), 1.0)
    sharp = cv2.addWeighted(contrast, 1.5, blur, -0.5, 0)
    bw = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    # Upscaled gray image often preserves small letters best.
    # Other passes are attempted only if the first reading fails.
    variants = (up, contrast, sharp, bw)
    readings = []
    for variant in variants:
        text = pytesseract.image_to_string(
            variant, lang="eng",
            config="--oem 3 --psm 7 -c tessedit_char_whitelist=BG0123456789",
        )
        text = re.sub(r"[^A-Z0-9]", "", text.upper())
        readings.append(text)
        if ROOM_PATTERN.fullmatch(text):
            return text

    # Try one extra contrast polarity only if the fast passes failed.
    inverted = cv2.bitwise_not(bw)
    extra = pytesseract.image_to_string(
        inverted, lang="eng",
        config="--oem 3 --psm 7 -c tessedit_char_whitelist=BG0123456789",
    )
    extra = re.sub(r"[^A-Z0-9]", "", extra.upper())
    readings.append(extra)
    if ROOM_PATTERN.fullmatch(extra):
        return extra

    # Only combine complementary readings from the same cropped plaque.
    for prefix in ("B", "G"):
        if prefix in readings:
            for digits in readings:
                if digits.isascii() and digits.isdigit():
                    candidate = prefix + digits
                    if ROOM_PATTERN.fullmatch(candidate):
                        return candidate
    return None


def detect_room(image, max_candidates=5):
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
        area = bw * bh / max(w * h, 1)
        if bw >= 25 and bh >= 9 and 1.6 < ratio < 10 and 0.0001 < area < 0.12:
            boxes.append((x, y, bw, bh))
    boxes.sort(key=lambda b: (b[1], -(b[2] * b[3])))
    for x, y, bw, bh in boxes[:max_candidates]:
        pad = max(3, round(bh * 0.08))
        x0, y0 = max(0, x - pad), max(0, y - pad)
        x1, y1 = min(w, x + bw + pad), min(h, y + bh + pad)
        crop = image[y0:y1, x0:x1]
        code = read_room(crop)
        if code:
            return code, (x0, y0, x1, y1), sharpness(crop)
    return None


class LightStabilizer:
    """Smooth small camera translations; cannot recover true motion blur."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.prev = None
        self.position = np.zeros(2, dtype=np.float64)
        self.smoothed = np.zeros(2, dtype=np.float64)

    def apply(self, frame):
        h, w = frame.shape[:2]
        size = (max(1, round(w * 0.33)), max(1, round(h * 0.33)))
        tiny = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), size)
        if self.prev is None:
            self.prev = tiny
            return frame
        pts = cv2.goodFeaturesToTrack(self.prev, maxCorners=100, qualityLevel=0.025, minDistance=12)
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
        move = np.median((new_pts[valid] - pts[valid]).reshape(-1, 2), axis=0) * (w / size[0])
        if not np.all(np.isfinite(move)) or np.linalg.norm(move) > 55:
            self.position[:] = 0
            self.smoothed[:] = 0
            return frame
        self.position += move
        self.smoothed += 0.20 * (self.position - self.smoothed)
        correction = np.clip(self.smoothed - self.position, -25, 25)
        matrix = np.float32([[1, 0, correction[0]], [0, 1, correction[1]]])
        return cv2.warpAffine(frame, matrix, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)


def image_test(path):
    picture = cv2.imread(str(path))
    if picture is None:
        raise SystemExit(f"Could not open image: {path}")
    print(f"Image: {picture.shape[1]} x {picture.shape[0]}")
    result = detect_room(picture)
    if result:
        code, box, sharp = result
        floor, number = room_parts(code)
        print(f"ROOM: {code} | FLOOR: {floor} | NUMBER: {number} | plaque sharpness: {sharp:.1f}")
        x0, y0, x1, y1 = box
        cv2.rectangle(picture, (x0, y0), (x1, y1), (0, 255, 0), 2)
    else:
        print("No confirmed room code in this image")
    out = OUTPUT / "image_test_result.jpg"
    cv2.imwrite(str(out), picture)
    print(f"Saved: {out}")


def run_camera(index, req_w, req_h, req_fps, scan_interval, max_candidates):
    if not TESSERACT_WINDOWS.is_file():
        raise SystemExit(f"Tesseract executable not found: {TESSERACT_WINDOWS}")
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(0, cv2.CAP_MSMF)
    if not camera.isOpened():
        raise SystemExit(f"Cannot open camera index {index}. Try --camera 0 or --camera 2")

    camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, req_w)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, req_h)
    camera.set(cv2.CAP_PROP_FPS, req_fps)
    actual_w = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = camera.get(cv2.CAP_PROP_FPS)
    print(f"Requested {req_w}x{req_h}@{req_fps}; camera reported {actual_w}x{actual_h}@{actual_fps:.1f}")
    print("If the picture is the laptop camera, try another --camera index.")
    print("Place the plaque 3m away; Q quit | P save frame | S stabilization on/off")

    recent = deque(maxlen=6)
    stab = LightStabilizer()
    stabilization_on = True
    last_request = 0.0
    scan_started_at = None
    pending = None
    prev_code = None
    streak = 0
    confirmed = None
    confirmed_at = 0.0
    last_report = None
    capture_ticks = deque(maxlen=150)
    scan_finished_ticks = deque(maxlen=150)
    last_scan_seconds = None
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            while True:
                ok, raw = camera.read()
                if not ok:
                    print("Failed to read camera frame")
                    break
                frame = stab.apply(raw) if stabilization_on else raw.copy()
                now = time.monotonic()
                capture_ticks.append(now)
                score = sharpness(frame)
                recent.append((score, frame.copy()))

                if pending is not None and pending.done():
                    try:
                        detected = pending.result()
                        scan_finished_ticks.append(now)
                        if scan_started_at is not None:
                            last_scan_seconds = now - scan_started_at
                        if detected is None:
                            prev_code, streak = None, 0
                        else:
                            code, box, roi_sharp = detected
                            streak = streak + 1 if code == prev_code else 1
                            prev_code = code
                            if streak >= 2:
                                confirmed = code
                                confirmed_at = now
                                if code != last_report:
                                    floor, number = room_parts(code)
                                    print(f"CONFIRMED {code} | floor {floor} | room {number} | plaque sharpness {roi_sharp:.1f}")
                                    last_report = code
                    except Exception as error:
                        print("OCR error:", error)
                    pending = None

                # Submit at most one OCR job; never queue stale video frames.
                # Interval is only the requested rate; actual OCR rate depends on CPU.
                if pending is None and now - last_request >= scan_interval and recent:
                    best_frame = max(recent, key=lambda item: item[0])[1]
                    pending = executor.submit(detect_room, best_frame, max_candidates)
                    last_request = now
                    scan_started_at = now

                preview = frame
                if frame.shape[1] > 1100:
                    scale = 1100 / frame.shape[1]
                    preview = cv2.resize(frame, (1100, round(frame.shape[0] * scale)))
                else:
                    preview = frame.copy()
                text_color = (0, 255, 0) if confirmed and now - confirmed_at < 5 else (255, 255, 255)
                text = (f"Room: {confirmed}" if confirmed and now - confirmed_at < 5
                        else "Searching for room signs...")
                cv2.putText(preview, text, (15, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.85, text_color, 2)
                def rate(stamps):
                    # Recent events per second, not assumed camera/engine FPS.
                    count = sum(t >= now - 5.0 for t in stamps)
                    return count / min(5.0, max(now - stamps[0], 0.2)) if stamps else 0.0

                video_rate = rate(capture_ticks)
                ocr_rate = rate(scan_finished_ticks)
                ocr_time = f"{last_scan_seconds:.2f}s" if last_scan_seconds is not None else "n/a"
                footer = (f"Video: {video_rate:.1f} FPS | OCR: {ocr_rate:.2f} scans/s "
                          f"| Last OCR: {ocr_time} | Stabilization: "
                          f"{'ON' if stabilization_on else 'OFF'}")
                cv2.putText(preview, footer, (12, preview.shape[0]-16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.47, (255, 255, 255), 2)
                cv2.imshow("HOB Robot: 3-meter test (Q/P/S)", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    stabilization_on = not stabilization_on
                    stab.reset()
                    recent.clear()
                    print("Stabilization:", "ON" if stabilization_on else "OFF")
                if key == ord("p"):
                    path = OUTPUT / f"capture_3m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
                    cv2.imwrite(str(path), raw)
                    print("Saved full-resolution frame:", path)
        finally:
            camera.release()
            cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="HOB Robot Windows USB camera 3-meter OCR test")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--scan-interval", type=float, default=0.2,
                        help="Minimum gap between OCR jobs in seconds; no job backlog")
    parser.add_argument("--max-candidates", type=int, default=5,
                        help="Maximum candidate plaques examined per OCR job")
    parser.add_argument("--image", type=Path, help="Test a saved photo instead of live camera")
    args = parser.parse_args()
    if args.image:
        image_test(args.image)
    else:
        run_camera(args.camera, args.width, args.height, args.fps,
                   max(0.0, args.scan_interval), max(1, args.max_candidates))


if __name__ == "__main__":
    main()
