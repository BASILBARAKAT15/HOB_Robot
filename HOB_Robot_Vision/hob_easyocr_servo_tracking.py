r"""HOB Robot: faster inverted vertical servo tracking for ANY visible number.

Works on Windows with the existing EasyOCR + OpenCV environment.
Arduino Mega: Servo.h sketch reading UP/DOWN/STOP newline commands at 115200.
Only the vertical servo is controlled; the stepper motor is never used.

Safe simulation (default):
    .\.venv\Scripts\python.exe .\hob_easyocr_follow_fast_inverted.py --camera 1

Live servo on a verified, mechanically safe setup:
    .\.venv\Scripts\python.exe .\hob_easyocr_follow_fast_inverted.py --camera 1 --port COM7 --enable-motors

Keys: Q/Esc exit | S toggle tracking pause | P save full-resolution image.
"""

import argparse
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

CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
PREVIEW_WIDTH = 960
OCR_WIDTH = 960              # Faster OCR; use 1280 again if distant signs are missed
OCR_INTERVAL = 0.05         # Submit the next frame promptly after OCR completes
BOX_KEEP_SECONDS = 1.6
MIN_CONFIDENCE = 0.58
SINGLE_DIGIT_CONFIDENCE = 0.75  # Less likely to chase a stray digit
CENTER_BAND = 0.12         # Dead zone: +/- 12% of screen height
MOTOR_COOLDOWN = 0.28       # Smaller gap between fresh OCR-based servo commands
MAX_OCR_AGE = 1.8          # Do not act on stale OCR frames
NUMBER_PATTERN = re.compile(r"^(?:[BG])?[0-9]{1,8}$")
OUTPUT_DIR = Path(__file__).resolve().parent / "output" / "follow_any_number"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_number(raw_text, min_digits):
    """Accept 1..8 digits optionally preceded by B or G."""
    text = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
    if not NUMBER_PATTERN.fullmatch(text):
        return None
    digits = sum(ch.isdigit() for ch in text)
    return text if digits >= min_digits else None


def scan_frame(reader, frame, taken_at, min_digits):
    start = time.perf_counter()
    h, w = frame.shape[:2]
    scale = min(1.0, OCR_WIDTH / w)
    if scale < 1:
        smaller = cv2.resize(frame, (round(w * scale), round(h * scale)),
                             interpolation=cv2.INTER_AREA)
    else:
        smaller = frame

    readings = reader.readtext(
        smaller, detail=1, allowlist="BG0123456789", decoder="greedy", batch_size=1
    )
    detections = []
    for box, text, confidence in readings:
        code = normalize_number(text, min_digits)
        if code is None:
            continue
        confidence = float(confidence)
        needed = (SINGLE_DIGIT_CONFIDENCE if sum(ch.isdigit() for ch in code) == 1
                  else MIN_CONFIDENCE)
        if confidence < needed:
            continue
        points = (np.asarray(box, dtype=np.float32) / scale).round().astype(np.int32)
        detections.append((points, code, confidence))

    return detections, time.perf_counter() - start, taken_at


def open_camera(index):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        cam = cv2.VideoCapture(index, backend)
        if cam.isOpened():
            return cam
        cam.release()
    raise SystemExit(f"Camera {index} is unavailable. Try --camera 0 or --camera 2")


def connect_mega(port):
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("Install pyserial: .\\.venv\\Scripts\\python.exe -m pip install pyserial") from exc
    try:
        device = serial.Serial(port, 115200, timeout=0, write_timeout=1)
    except serial.SerialException as exc:
        raise SystemExit(f"Cannot open {port}. Close Arduino Serial Monitor: {exc}") from exc
    print(f"Connected to Mega: {port} at 115200. Waiting for reset...")
    time.sleep(2)
    device.reset_input_buffer()
    return device


def send_command(mega, command, live):
    if live:
        mega.write((command + "\n").encode("ascii"))
        print("SENT TO MEGA:", command)
    else:
        print("SIMULATED:", command)


def choose_number(detections, previously_followed):
    """Keep following the same code while visible, to avoid jumping between signs."""
    matching = [d for d in detections if d[1] == previously_followed]
    eligible = matching if matching else detections
    return max(eligible, key=lambda d: d[2])


def main():
    parser = argparse.ArgumentParser(description="HOB live vertical servo follows any OCR number")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--port", help="COM port of Mega, e.g. COM7")
    parser.add_argument("--enable-motors", action="store_true", help="Send commands to actual servo")
    parser.add_argument("--invert", action="store_true", help="Undo this version's default reversed UP/DOWN direction")
    parser.add_argument("--min-digits", type=int, default=1,
                        help="Ignore strings shorter than this many digits (1..8)")
    args = parser.parse_args()
    if not 1 <= args.min_digits <= 8:
        parser.error("--min-digits must be between 1 and 8")
    if args.enable_motors and not args.port:
        parser.error("--enable-motors needs --port COM7 (or your current Mega port)")

    print("Loading EasyOCR; CUDA:", torch.cuda.is_available())
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    mega = connect_mega(args.port) if args.enable_motors else None
    cam = open_camera(args.camera)
    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
    cam.set(cv2.CAP_PROP_FPS, 30)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    print("Camera resolution:", int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
          int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print("MODE:", "LIVE SERVO" if args.enable_motors else "SIMULATION")
    print("Tracking ANY number. Faster OCR, reversed directions. Q quit, S pause, P snapshot")

    future = None
    last_scan_start = 0.0
    last_result_at = 0.0
    last_boxes = []
    last_ocr_time = None
    last_command_at = 0.0
    followed_number = None
    tracking_on = True
    status = "Looking for numbers..."
    frame_times = deque(maxlen=45)

    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            while True:
                ok, frame = cam.read()
                if not ok:
                    print("Camera frame read failed")
                    break

                now = time.perf_counter()
                frame_times.append(now)
                h, w = frame.shape[:2]

                # Only issue motor commands based on FRESHLY COMPLETED OCR results.
                if future is not None and future.done():
                    try:
                        boxes, duration, captured_at = future.result()
                        last_boxes = boxes
                        last_ocr_time = duration
                        last_result_at = now
                        if boxes:
                            points, code, conf = choose_number(boxes, followed_number)
                            followed_number = code
                            center_y = float(np.mean(points[:, 1]))
                            error = (center_y - h / 2) / h
                            age = now - captured_at
                            if not tracking_on:
                                status = "PAUSED (press S to resume)"
                            elif abs(error) <= CENTER_BAND:
                                status = f"CENTERED: {code} | conf {conf:.2f}"
                            elif age > MAX_OCR_AGE:
                                status = f"OCR too old ({age:.1f}s): no move"
                            elif now - last_command_at >= MOTOR_COOLDOWN:
                                command = "DOWN" if error < 0 else "UP"  # default REVERSED compared with previous script
                                if args.invert:
                                    command = "DOWN" if command == "UP" else "UP"
                                send_command(mega, command, args.enable_motors)
                                last_command_at = now
                                status = f"FOLLOW {code}: {command} | Y {error:+.2f}"
                            else:
                                status = f"FOLLOW {code}: cooldown | Y {error:+.2f}"
                        else:
                            # Do not keep moving when number leaves the frame.
                            followed_number = None
                            status = "No number: servo holds position"
                    except Exception as exc:
                        last_boxes = []
                        followed_number = None
                        status = "OCR error; no motion"
                        print("OCR error:", exc)
                    future = None

                if future is None and now - last_scan_start >= OCR_INTERVAL:
                    future = pool.submit(scan_frame, reader, frame.copy(), now, args.min_digits)
                    last_scan_start = now

                display = frame.copy()
                y1 = int(h * (0.5 - CENTER_BAND))
                y2 = int(h * (0.5 + CENTER_BAND))
                cv2.line(display, (0, y1), (w, y1), (0, 255, 255), 2)
                cv2.line(display, (0, y2), (w, y2), (0, 255, 255), 2)
                cv2.line(display, (0, h // 2), (w, h // 2), (255, 255, 255), 1)

                if now - last_result_at <= BOX_KEEP_SECONDS:
                    for pts, code, conf in last_boxes:
                        selected = code == followed_number
                        color = (0, 255, 0) if selected else (255, 160, 0)
                        cv2.polylines(display, [pts], True, color, 3)
                        x, y = pts[0]
                        cv2.putText(display, f"{code} ({conf:.2f})",
                                    (int(x), max(25, int(y) - 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                fps = ((len(frame_times)-1) / max(frame_times[-1]-frame_times[0], 1e-6)
                       if len(frame_times) > 1 else 0)
                ocr = f"{last_ocr_time:.2f}s" if last_ocr_time is not None else "..."
                mode = "LIVE" if args.enable_motors else "SIM"
                if not tracking_on:
                    mode += " PAUSED"
                cv2.putText(display, f"{mode} | VIDEO {fps:.1f}fps | OCR {ocr}",
                            (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, status[:82], (15, 78),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

                view_w = min(w, PREVIEW_WIDTH)
                view_h = round(h * view_w / w)
                preview = cv2.resize(display, (view_w, view_h), interpolation=cv2.INTER_AREA)
                cv2.imshow("HOB Robot - Follow Any Number", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    tracking_on = not tracking_on
                    print("TRACKING:", "ACTIVE" if tracking_on else "PAUSED")
                    if not tracking_on and mega is not None:
                        send_command(mega, "STOP", True)
                if key == ord("p"):
                    filename = OUTPUT_DIR / f"capture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print("Saved:", filename)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        if mega is not None:
            try:
                send_command(mega, "STOP", True)
            except Exception:
                pass
            mega.close()
        cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
