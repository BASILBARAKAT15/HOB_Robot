r"""HOB Robot: search ONLY upward in small steps, waiting at each angle.

Recognizes valid university room codes only: B2xxx, B1xxx, G0xxx,
1xxx, 2xxx, 3xxx, 4xxx. Uses Arduino Mega over USB, no stepper.

Simulation:
  .\.venv\Scripts\python.exe .\hob_easyocr_search_up_wait.py --camera 1
Live, only after checking physical direction / mechanical limits:
  .\.venv\Scripts\python.exe .\hob_easyocr_search_up_wait.py --camera 1 --port COM7 --enable-motors

The user's previous servo installation had inverted wiring/mechanics: Mega
command DOWN physically tilted UP. Adjust --up-command if yours differs.

Keys: Q/Esc quit, S pause/resume, R search again from current angle,
P save a full-resolution frame. There is NO automatic downward sweep.
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

CAMERA_WIDTH = 1920
CAMERA_HEIGHT = 1080
PREVIEW_WIDTH = 960
OCR_INTERVAL = 0.07
MAX_OCR_AGE = 3.0
CONFIDENCE_MIN = 0.55
CONFIRM_FRAMES = 2
SETTLE_SECONDS = 0.45

# These MUST match the Arduino sketch currently installed on Mega.
SERVO_START = 90
SERVO_MIN = 75
SERVO_MAX = 105
SERVO_STEP = 5

ROOM_PATTERN = re.compile(r"^(?:B[12]|G0|[1-4])[0-9]{3}$")
SAVE_DIR = Path(__file__).resolve().parent / "output" / "search_up_wait"
SAVE_DIR.mkdir(parents=True, exist_ok=True)


def normalize_room(raw):
    code = re.sub(r"[^A-Z0-9]", "", raw.upper())
    return code if ROOM_PATTERN.fullmatch(code) else None


def floor_and_room(code):
    prefix = code[:-3]
    return ("G" if prefix == "G0" else prefix), code[-3:]


def scan_frame(reader, frame, captured_at, width):
    started = time.perf_counter()
    height, original_width = frame.shape[:2]
    scale = min(1.0, width / original_width)
    if scale < 1:
        smaller = cv2.resize(frame, (round(original_width * scale), round(height * scale)),
                             interpolation=cv2.INTER_AREA)
    else:
        smaller = frame
    output = []
    for box, raw, confidence in reader.readtext(
        smaller, detail=1, allowlist="BG0123456789", decoder="greedy", batch_size=1
    ):
        code = normalize_room(raw)
        if code is None or float(confidence) < CONFIDENCE_MIN:
            continue
        points = (np.asarray(box, dtype=np.float32) / scale).round().astype(np.int32)
        output.append((points, code, float(confidence)))
    return output, time.perf_counter() - started, captured_at


def open_camera(camera_index):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(camera_index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    raise SystemExit(f"Cannot open camera {camera_index}. Try --camera 0")


def open_mega(port):
    try:
        import serial
    except ImportError as exc:
        raise SystemExit("Install pyserial: .\\.venv\\Scripts\\python.exe -m pip install pyserial") from exc
    try:
        mega = serial.Serial(port, 115200, timeout=0, write_timeout=1)
    except serial.SerialException as exc:
        raise SystemExit(f"Cannot open {port}; close Arduino Serial Monitor: {exc}") from exc
    print(f"Mega connected: {port}; waiting for its automatic reset...")
    time.sleep(2)
    mega.reset_input_buffer()
    return mega


def next_up_angle(angle, up_command):
    # 'DOWN' may physically point UP if the camera is installed inverted.
    if up_command == "DOWN":
        return max(SERVO_MIN, angle - SERVO_STEP)
    return min(SERVO_MAX, angle + SERVO_STEP)


def send(mega, command, enabled):
    if enabled:
        mega.write((command + "\n").encode("ascii"))
        print(f"MEGA: {command} (physical tilt UP if direction verified)")
    else:
        print(f"SIMULATED: {command} (intended tilt UP)")


def main():
    parser = argparse.ArgumentParser(description="Look UP in steps; wait for valid room sign; HOLD")
    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--port", default=None)
    parser.add_argument("--enable-motors", action="store_true")
    parser.add_argument("--up-command", choices=["UP", "DOWN"], default="DOWN",
                        help="Command that makes the physical camera tilt upward; default DOWN for the previously inverted setup")
    parser.add_argument("--wait-seconds", type=float, default=2.5,
                        help="Minimum waiting period at each angle, in addition to waiting for OCR")
    parser.add_argument("--ocr-width", type=int, default=960)
    args = parser.parse_args()
    if args.enable_motors and not args.port:
        parser.error("Provide --port COM7 with --enable-motors")
    if args.wait_seconds < 0.5:
        parser.error("--wait-seconds must be at least 0.5")
    if not 320 <= args.ocr_width <= 1920:
        parser.error("--ocr-width must be 320..1920")

    print("Loading EasyOCR; CUDA:", torch.cuda.is_available())
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
    mega = open_mega(args.port) if args.enable_motors else None
    cam = None

    # Arduino starts at 90 degrees on boot; there is intentionally NO CENTER
    # command from Python, as it could cause a downward movement.
    angle = SERVO_START
    found_code = None
    candidate_code = None
    candidate_count = 0
    active = True
    future = None
    last_ocr_started = 0.0
    last_boxes = []
    last_result_at = 0.0
    last_ocr_time = None
    last_move_at = 0.0
    settle_until = 0.0
    need_new_reading = True
    next_move_at = time.perf_counter() + args.wait_seconds
    state = "WAIT: reading current angle before searching upward"
    frame_times = deque(maxlen=45)

    try:
        cam = open_camera(args.camera)
        print("Camera:", int(cam.get(cv2.CAP_PROP_FRAME_WIDTH)), "x",
              int(cam.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        print("MODE:", "LIVE SERVO" if args.enable_motors else "SIMULATION")
        print("ONLY ROOM CODES: B2xxx / B1xxx / G0xxx / 1xxx / 2xxx / 3xxx / 4xxx")
        print(f"SEARCH: tilt UP {SERVO_STEP} degrees, wait {args.wait_seconds:.1f}s and OCR; NEVER sweep downward")
        print(f"UP COMMAND: {args.up_command} (check physical direction before letting it run)")
        print("Q/Esc quit | S pause | R restart search from CURRENT angle | P snapshot")

        with ThreadPoolExecutor(max_workers=1) as pool:
            while True:
                ok, frame = cam.read()
                if not ok:
                    print("Camera frame read failed")
                    break
                now = time.perf_counter()
                frame_times.append(now)

                if future is not None and future.done():
                    try:
                        boxes, last_ocr_time, captured_at = future.result()
                        last_result_at = now
                        last_boxes = boxes
                        # Never process an old image captured before the last movement settled.
                        fresh = (now - captured_at <= MAX_OCR_AGE and
                                 captured_at >= settle_until)
                        if fresh:
                            need_new_reading = False
                            if found_code is None:
                                selection = max(boxes, key=lambda item: item[2]) if boxes else None
                                if selection is None:
                                    candidate_code = None
                                    candidate_count = 0
                                else:
                                    _pts, code, confidence = selection
                                    if code == candidate_code:
                                        candidate_count += 1
                                    else:
                                        candidate_code = code
                                        candidate_count = 1
                                    state = f"VERIFY {code} ({candidate_count}/{CONFIRM_FRAMES})"
                                    if candidate_count >= CONFIRM_FRAMES:
                                        found_code = code
                                        floor, room = floor_and_room(code)
                                        state = f"FOUND {code} - FLOOR {floor} ROOM {room} - HOLD POSITION"
                                        print("ROOM FOUND:", code, "| FLOOR:", floor,
                                              "| ROOM:", room, "| CAMERA HOLDING")
                        elif found_code is None:
                            state = "WAIT: ignoring stale OCR frame"
                    except Exception as exc:
                        last_boxes = []
                        need_new_reading = True
                        state = "OCR ERROR - NO MOVEMENT"
                        print("OCR error:", exc)
                    future = None

                # Only take new snapshots after settling; OCR remains async for smooth preview.
                if future is None and now >= settle_until and now - last_ocr_started >= OCR_INTERVAL:
                    future = pool.submit(scan_frame, reader, frame.copy(), now, args.ocr_width)
                    last_ocr_started = now

                # Unlike the previous sweep, this moves in ONLY ONE physical direction.
                if active and found_code is None and not need_new_reading and now >= next_move_at:
                    if candidate_count > 0:
                        # Give a possible room code another OCR read to confirm it.
                        state = f"WAIT: confirming {candidate_code}"
                    else:
                        proposed = next_up_angle(angle, args.up_command)
                        if proposed == angle:
                            state = f"UPPER SCAN LIMIT: {angle} deg - HOLD (no DOWN)"
                        else:
                            send(mega, args.up_command, args.enable_motors)
                            angle = proposed
                            last_move_at = now
                            settle_until = now + SETTLE_SECONDS
                            need_new_reading = True
                            next_move_at = now + args.wait_seconds
                            state = f"MOVE UP, then WAIT at approx {angle} deg"
                            print(state)
                elif found_code is None and active and now < next_move_at:
                    if candidate_count == 0 and not need_new_reading:
                        state = f"WAIT at angle ~{angle} deg for room code"

                display = frame.copy()
                if now - last_result_at <= 1.5:
                    for pts, code, conf in last_boxes:
                        color = (0, 255, 0) if code == found_code else (0, 190, 255)
                        cv2.polylines(display, [pts], True, color, 3)
                        x, y = pts[0]
                        cv2.putText(display, f"{code} ({conf:.2f})", (int(x), max(30, int(y)-10)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                fps = ((len(frame_times) - 1) / max(1e-6, frame_times[-1] - frame_times[0])
                       if len(frame_times) >= 2 else 0.0)
                ocr_t = f"{last_ocr_time:.2f}s" if last_ocr_time is not None else "..."
                cv2.putText(display, f"{'LIVE' if args.enable_motors else 'SIM'} | FPS {fps:.1f} | OCR {ocr_t} | angle ~{angle}",
                            (16, 42), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 255), 2)
                cv2.putText(display, ("PAUSED | " if not active else "") + state[:75], (16, 82),
                            cv2.FONT_HERSHEY_SIMPLEX, .68, (0, 255, 255), 2)
                h, w = display.shape[:2]
                preview_width = min(w, PREVIEW_WIDTH)
                view = cv2.resize(display, (preview_width, round(h * preview_width / w)),
                                  interpolation=cv2.INTER_AREA)
                cv2.imshow("HOB Robot - Room Signs: UP, WAIT, HOLD", view)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("s"):
                    active = not active
                    print("SEARCH", "RESUMED" if active else "PAUSED")
                    if active:
                        next_move_at = time.perf_counter() + args.wait_seconds
                if key == ord("r"):
                    found_code = None
                    candidate_code = None
                    candidate_count = 0
                    next_move_at = time.perf_counter() + args.wait_seconds
                    need_new_reading = True
                    state = "RESTART search from CURRENT angle (never move down)"
                    print(state)
                if key == ord("p"):
                    filename = SAVE_DIR / f"capture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print("Saved:", filename)
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        if mega is not None:
            # STOP does not command a position change on the existing Mega sketch.
            try:
                mega.write(b"STOP\n")
            except Exception:
                pass
            mega.close()
        if cam is not None:
            cam.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
