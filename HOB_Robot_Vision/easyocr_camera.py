r"""HOB Robot - responsive USB camera preview with asynchronous EasyOCR.

Run on Windows in the existing EasyOCR venv:
    .\.venv\Scripts\python.exe .\hob_easyocr_fast.py

Keys: Q or Esc to quit, P to save a raw frame.
"""

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import re
import time

import cv2
import easyocr
import numpy as np
import torch

CAMERA_INDEX = 1             # Change to 0 if your USB camera uses index 0
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
PREVIEW_WIDTH = 960         # Window width only; does not alter camera capture
OCR_WIDTH = 1280            # 960 = faster; 1600 = better for small/far signs
OCR_INTERVAL = 0.5          # Minimum seconds between OCR jobs
BOX_KEEP_SECONDS = 1.5
ROOM_PATTERN = re.compile(r"^(?:B[12]|G0|[1-4])[0-9]{3}$")
SAVE_DIR = Path(__file__).resolve().parent / "output" / "fast_camera"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def room_parts(code):
    prefix = code[:-3]
    return ("G" if prefix == "G0" else prefix), code[-3:]


def scan_frame(reader, frame):
    """Run OCR away from the camera/display loop.

    Returns bounding boxes expressed in ORIGINAL frame coordinates.
    """
    started = time.perf_counter()
    h, w = frame.shape[:2]
    scale = min(1.0, OCR_WIDTH / w)
    if scale < 1.0:
        small = cv2.resize(frame, (round(w * scale), round(h * scale)),
                           interpolation=cv2.INTER_AREA)
    else:
        small = frame

    readings = reader.readtext(
        small,
        detail=1,
        allowlist="BG0123456789",
        decoder="greedy",
        batch_size=1,
    )
    found = []
    for box, text, confidence in readings:
        code = re.sub(r"[^A-Z0-9]", "", text.upper())
        if not ROOM_PATTERN.fullmatch(code):
            continue
        points = (np.asarray(box, dtype=np.float32) / scale).round().astype(np.int32)
        floor, number = room_parts(code)
        found.append((points, code, floor, number, float(confidence)))

    return found, time.perf_counter() - started


def open_camera(index):
    # DirectShow is often reliable for Windows USB webcams.
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    raise SystemExit(f"Cannot open camera {index}. Try CAMERA_INDEX = 0 or 2")


def main():
    use_gpu = torch.cuda.is_available()
    print(f"Loading EasyOCR; CUDA: {use_gpu}")
    reader = easyocr.Reader(["en"], gpu=use_gpu)

    cap = open_camera(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    # This may be ignored by some Windows camera drivers.
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("Camera resolution:",
          int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
          int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("Q/Esc: quit | P: save full-resolution frame")

    future = None
    last_scan_start = 0.0
    last_result_at = 0.0
    last_boxes = []
    last_ocr_time = None
    last_reported = None
    last_reported_at = 0.0
    frame_times = deque(maxlen=45)

    with ThreadPoolExecutor(max_workers=1) as pool:
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame read failed")
                    break

                now = time.perf_counter()
                frame_times.append(now)

                # Collect OCR only when the background job finishes.
                if future is not None and future.done():
                    try:
                        last_boxes, last_ocr_time = future.result()
                        last_result_at = now
                        for _pts, code, floor, number, confidence in last_boxes:
                            if code != last_reported or now - last_reported_at >= 3.0:
                                print(f"ROOM: {code} | FLOOR: {floor} | NUMBER: {number} "
                                      f"| OCR confidence: {confidence:.2f}")
                                last_reported = code
                                last_reported_at = now
                    except Exception as exc:
                        last_boxes = []
                        print("OCR error:", exc)
                    future = None

                # No queue: the next job uses the newest camera frame.
                if future is None and now - last_scan_start >= OCR_INTERVAL:
                    future = pool.submit(scan_frame, reader, frame.copy())
                    last_scan_start = now

                # The preview runs independently of OCR speed.
                display = frame.copy()
                if now - last_result_at <= BOX_KEEP_SECONDS:
                    for points, code, floor, number, conf in last_boxes:
                        cv2.polylines(display, [points], True, (0, 255, 0), 3)
                        x, y = points[0]
                        label = f"{code} | Floor {floor} | Room {number}"
                        cv2.putText(display, label, (int(x), max(30, int(y) - 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                if len(frame_times) >= 2:
                    fps = (len(frame_times) - 1) / max(frame_times[-1] - frame_times[0], 1e-6)
                else:
                    fps = 0.0
                timing = f"{last_ocr_time:.2f}s" if last_ocr_time is not None else "..."
                cv2.putText(display, f"Video: {fps:.1f} FPS | OCR time: {timing}",
                            (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)

                h, w = display.shape[:2]
                preview_w = min(w, PREVIEW_WIDTH)
                preview_h = round(h * preview_w / w)
                preview = cv2.resize(display, (preview_w, preview_h),
                                     interpolation=cv2.INTER_AREA)
                cv2.imshow("HOB Robot - Fast EasyOCR", preview)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("p"):
                    path = SAVE_DIR / f"capture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(path), frame)
                    print("Saved:", path)
        finally:
            cap.release()
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
