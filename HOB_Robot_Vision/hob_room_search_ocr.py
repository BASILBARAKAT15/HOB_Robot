r"""HOB Robot: one-click Run; room names are embedded in the Python code.

Open in VS Code and click Run Python File (top-right play button).
Default: live camera index 1, Mega COM7, serial at 115200.
For a test WITHOUT moving the servo: add --simulate if launching from a terminal.
Camera physically searches UP ONLY (Arduino command DOWN for inverted mount).
After 2 matching reads: send CENTER, display the room info, then automatically
release camera, close serial and finish after a brief display.

Q/Esc: quit early | S: pause/resume | P: save image.
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
OCR_WIDTH = 1280
OCR_INTERVAL = 0.15
BOX_KEEP_SECONDS = 1.5
MIN_CONFIDENCE = 0.45
ROOM_PATTERN = re.compile(r"^(?:B[12]|G0|[1-4])[0-9]{3}$")

# Mega DOWN physically tilts the camera UP for this particular mounting.
PHYSICAL_UP_COMMAND = "DOWN"
RETURN_SETTLE_SECONDS = 1.2  # Allow camera to reach home before announcing
HOME_ANGLE = 90  # Must match Arduino CENTER_ANGLE.
RESULT_SHOW_SECONDS = 3.0  # Keep result visible briefly, then auto-close.

BASE_DIR = Path(__file__).resolve().parent
SAVE_DIR = BASE_DIR / "output" / "room_search"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Room names are embedded in this Python file. No separate data file is needed.
# Add real, verified names here; unknown rooms display their code and floor.
ROOM_NAMES = {
    "G0070": {
        "ar": "مختبر العمليات الإنتاجية",
        "en": "Manufacturing Processes Lab",
    },
    # "B1030": {"ar": "اسم القاعة", "en": "Room name"},
}


def room_parts(code):
    prefix = code[:-3]
    return ("G" if prefix == "G0" else prefix), code[-3:]


def scan_frame(reader, frame):
    """Keep original EasyOCR pipeline; return boxes in camera coordinates."""
    started = time.perf_counter()
    h, w = frame.shape[:2]
    scale = min(1.0, OCR_WIDTH / w)
    if scale < 1.0:
        small = cv2.resize(
            frame, (round(w * scale), round(h * scale)),
            interpolation=cv2.INTER_AREA,
        )
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
    for box, raw_text, confidence in readings:
        code = re.sub(r"[^A-Z0-9]", "", raw_text.upper())
        if not ROOM_PATTERN.fullmatch(code) or float(confidence) < MIN_CONFIDENCE:
            continue
        points = (np.asarray(box, dtype=np.float32) / scale).round().astype(np.int32)
        floor, number = room_parts(code)
        found.append((points, code, floor, number, float(confidence)))
    return found, time.perf_counter() - started


def open_camera(index):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            return cap
        cap.release()
    raise SystemExit(f"Cannot open camera {index}. Try --camera 0 or 2")


def send_mega(mega, command):
    if mega is None:
        print(f"SIMULATED: {command}")
        return True
    try:
        mega.write((command + "\n").encode("ascii"))
        print(f"MEGA COMMAND SENT: {command}")
        return True
    except Exception as exc:
        print(f"FAILED TO SEND {command}: {exc}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="HOB Room OCR: search upward, return to center, display room name"
    )
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--port", default="COM7")
    # One-click Run enables Mega automatically. Use --simulate to keep servo off.
    parser.add_argument("--simulate", action="store_true", help="Do not move servo")
    parser.add_argument("--wait-seconds", type=float, default=2.5)
    parser.add_argument("--max-steps", type=int, default=3)
    args = parser.parse_args()
    args.enable_motors = not args.simulate
    if args.wait_seconds < 1.0:
        parser.error("--wait-seconds must be >= 1.0")
    if not 0 <= args.max_steps <= 3:
        parser.error("--max-steps must be between 0 and 3 until safe travel is verified")

    print("Loading EasyOCR; CUDA:", torch.cuda.is_available())
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    mega = None
    cap = None
    try:
        if args.enable_motors:
            try:
                import serial
                mega = serial.Serial(args.port, 115200, timeout=0, write_timeout=1)
            except (ImportError, OSError) as exc:
                raise SystemExit(f"Cannot connect to Mega on {args.port}: {exc}") from exc
            print(f"Connected to Mega on {args.port}, waiting for reset...")
            time.sleep(2.0)
            mega.reset_input_buffer()

        cap = open_camera(args.camera)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        print("Camera resolution:", int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
              int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("MODE:", "LIVE SERVO" if mega else "SIMULATION")
        print("Valid room codes: B2xxx / B1xxx / G0xxx / 1xxx / 2xxx / 3xxx / 4xxx")
        print("Search: PHYSICAL UP ONLY via Mega", PHYSICAL_UP_COMMAND)
        print("On detection: CENTER to return to home, then show code and room name")
        print("Q/Esc quit | S pause | P screenshot | AUTO CLOSE after room found")

        # Sync the search's physical starting angle with Arduino CENTER_ANGLE.
        # The Arduino sketch itself also positions the servo at CENTER on startup.
        send_mega(mega, "CENTER")
        next_scan_at = time.perf_counter() + RETURN_SETTLE_SECONDS
        last_scan_start = 0.0
        last_result_at = 0.0
        last_boxes = []
        last_ocr_time = None
        frame_times = deque(maxlen=45)
        candidate_code = None
        confirmations = 0
        missing_scans = 0
        locked = None
        announce_at = None
        announced = False
        finish_at = None
        steps = 0
        paused = False
        status = "SEARCH: checking current camera angle"
        future = None

        with ThreadPoolExecutor(max_workers=1) as pool:
            while True:
                ok, frame = cap.read()
                if not ok:
                    print("Camera frame read failed")
                    break
                now = time.perf_counter()
                frame_times.append(now)

                if future is not None and future.done():
                    try:
                        boxes, last_ocr_time = future.result()
                        last_boxes = boxes
                        last_result_at = now
                        best = max(boxes, key=lambda item: item[4], default=None)
                        if locked is None and best is not None:
                            _points, code, floor, number, confidence = best
                            missing_scans = 0
                            if code == candidate_code:
                                confirmations += 1
                            else:
                                candidate_code = code
                                confirmations = 1
                            status = f"VERIFY {code}: {confirmations}/2"
                            next_scan_at = now + OCR_INTERVAL
                            if confirmations >= 2:
                                locked = (code, floor, number, confidence)
                                announce_at = now + RETURN_SETTLE_SECONDS
                                announced = False
                                success = send_mega(mega, "CENTER")
                                status = (
                                    f"FOUND {code} - returning to {HOME_ANGLE} deg"
                                    if success else f"FOUND {code} - CENTER COMMAND FAILED"
                                )
                        elif locked is None:
                            candidate_code = None
                            confirmations = 0
                            missing_scans += 1
                            if not paused and missing_scans >= 2 and steps < args.max_steps:
                                # This command is physically UP for the user's inverted mount.
                                if send_mega(mega, PHYSICAL_UP_COMMAND):
                                    steps += 1
                                    missing_scans = 0
                                    next_scan_at = now + args.wait_seconds
                                    status = (
                                        f"SEARCH UP {steps}/{args.max_steps}; "
                                        f"wait {args.wait_seconds:.1f}s"
                                    )
                                else:
                                    paused = True
                                    status = "SERIAL ERROR - search paused"
                            elif steps >= args.max_steps:
                                status = "UP LIMIT reached - HOLD and check current view"
                                next_scan_at = now + 0.5
                            else:
                                status = "SEARCH: looking for room code"
                                next_scan_at = now + OCR_INTERVAL
                    except Exception as exc:
                        print("OCR error:", exc)
                        last_boxes = []
                        status = "OCR ERROR - no movement"
                        next_scan_at = now + 1.0
                    future = None

                if locked is not None and not announced and announce_at is not None and now >= announce_at:
                    code, floor, number, confidence = locked
                    record = ROOM_NAMES.get(code, {})
                    ar = record.get("ar", "")
                    en = record.get("en", "")
                    print("\n========== ROOM DETECTED ==========")
                    print("Room code:", code)
                    print("Floor:", floor)
                    print("Room number:", number)
                    print("اسم القاعة:", ar if ar else "اسم القاعة غير مسجل في الكود")
                    if en:
                        print("Room name:", en)
                    print("===================================\n")
                    status = f"FOUND {code} | HOME {HOME_ANGLE} | {en or 'Room name not defined'}"
                    announced = True
                    finish_at = now + RESULT_SHOW_SECONDS

                if (future is None and locked is None and not paused
                        and now >= next_scan_at
                        and now - last_scan_start >= OCR_INTERVAL):
                    future = pool.submit(scan_frame, reader, frame.copy())
                    last_scan_start = now

                display = frame.copy()
                if now - last_result_at <= BOX_KEEP_SECONDS:
                    for points, code, floor, number, confidence in last_boxes:
                        cv2.polylines(display, [points], True, (0, 255, 0), 3)
                        x, y = points[0]
                        label = f"{code} | Floor {floor} | Room {number}"
                        cv2.putText(display, label, (int(x), max(30, int(y) - 12)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
                fps = ((len(frame_times) - 1) / max(frame_times[-1] - frame_times[0], 1e-6)
                       if len(frame_times) > 1 else 0.0)
                timing = f"{last_ocr_time:.2f}s" if last_ocr_time is not None else "..."
                cv2.putText(display, f"Video: {fps:.1f} FPS | OCR: {timing}",
                            (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 2)
                cv2.putText(display, ("PAUSED | " if paused else "") + status[:100],
                            (15, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (0, 255, 255), 2)

                h, w = display.shape[:2]
                preview_w = min(w, PREVIEW_WIDTH)
                preview_h = round(h * preview_w / w)
                preview = cv2.resize(display, (preview_w, preview_h),
                                     interpolation=cv2.INTER_AREA)
                cv2.imshow("HOB Robot - Search / Return Home / Room Name", preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    paused = not paused
                    print("PAUSED" if paused else "RESUMED")
                if key == ord("p"):
                    path = SAVE_DIR / f"capture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(path), frame)
                    print("Saved:", path)
                # Close automatically once the result has been visible for a few seconds.
                if finish_at is not None and time.perf_counter() >= finish_at:
                    print("DONE: closing USB camera and serial connection")
                    break
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        if mega is not None:
            try:
                mega.write(b"STOP\n")
            except Exception:
                pass
            mega.close()
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
