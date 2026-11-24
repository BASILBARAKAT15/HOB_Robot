"""HOB Robot — continuous mission camera + on-arrival door/room search.

Run in VS Code via Run Python File. No JSON files.

NAVIGATING: continuous video + YOLO object recognition (all COCO classes).
ARRIVED event: detect a door, prioritize OCR above it, then use full-frame OCR if needed.
After a verified room code: servo CENTER, display the name, continue video.

Test event: press A in the OpenCV window, or send UDP 'ARRIVED' to 127.0.0.1:5055.
To target a specific room: send 'ARRIVED G0070'.
Press N to return to normal object-detection mode, Q/Esc to exit.

Hardware: Arduino Mega with UP/DOWN/CENTER/STOP serial sketch, 115200.
Auto-detect the camera-servo Arduino over USB; no fixed COM port.
If Arduino is disconnected, live camera + AI still run in simulation mode.
Set ENABLE_SERVO True only AFTER fixing/verifying safe wiring and travel limits.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
import re
import socket
import time

import cv2
import easyocr
import numpy as np
import torch
from ultralytics import YOLO, YOLOWorld

# ----------------- SETTINGS -----------------
CAMERA_INDEX = 1
PREVIEW_WIDTH = 960
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080

# Arduino port is detected automatically; no COM number is hardcoded.
ENABLE_SERVO = True          # Set False to disable physical servo movement
REQUIRE_MEGA = False          # True: stop if Mega is not detected; False: video-only fallback
PHYSICAL_UP_COMMAND = "DOWN"  # Inverted mounting: Mega DOWN raises camera
MAX_UP_STEPS = 3             # Only with 5-degree Arduino sketch, 90 -> 75 deg
WAIT_AFTER_MOVE = 2.5

UDP_HOST = "127.0.0.1"        # Local demo. Change only for a trusted network.
UDP_PORT = 5055

OBJECT_CONFIDENCE = 0.35
DOOR_CONFIDENCE = 0.28
OCR_CONFIDENCE = 0.45
MIN_CONFIDENCE = OCR_CONFIDENCE  # Preserve the uploaded working EasyOCR filter
OCR_WIDTH = 1280
FULL_FRAME_OCR_FALLBACK = True  # Read room codes even if YOLO misses the door
ROOM_CONFIRMATIONS = 2
RESULT_SECONDS = 5.0
VISION_MIN_INTERVAL = 0.12
ROOM_PATTERN = re.compile(r"^(?:B[12]|G0|[1-4])[0-9]{3}$")

ROOM_NAMES = {
    "G0070": {"ar": "مختبر العمليات الإنتاجية", "en": "Manufacturing Processes Lab"},
    # "B1030": {"ar": "اسم القاعة", "en": "Verified room name"},
}

SAVE_DIR = Path(__file__).resolve().parent / "output" / "mission_camera"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
# --------------------------------------------


def parse_room(raw):
    code = re.sub(r"[^A-Z0-9]", "", raw.upper())
    return code if ROOM_PATTERN.fullmatch(code) else None


def room_parts(code):
    floor = code[:-3]
    return ("G" if floor == "G0" else floor), code[-3:]


def _is_arduino_candidate(port):
    """Select likely Mega USB serial ports; skip common ESP32 CP210x ports."""
    vid = getattr(port, "vid", None)
    description = (getattr(port, "description", "") or "").lower()
    hwid = (getattr(port, "hwid", "") or "").lower()

    if vid in (0x2341, 0x2A03, 0x1A86, 0x0403):
        return True  # Official Arduino, WCH CH340 clones, FTDI adapters
    if vid == 0x10C4 or "cp210" in description or "cp210" in hwid:
        return False  # Typically the ESP32, not our Mega
    return any(word in description for word in ("arduino", "mega 2560", "ch340", "ch341", "usb serial"))


def _read_serial_lines(ser, seconds):
    """Read a short startup/handshake response without an endless wait."""
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            line = ser.readline().decode("utf-8", errors="replace").strip()
        except (OSError, UnicodeError):
            break
        if line:
            yield line


def _is_camera_servo_ready(message):
    normalized = " ".join(message.upper().split())
    return normalized.startswith("HOB CAMERA") and "READY" in normalized


def connect_mega():
    """Find the *camera-servo* Mega, not just the first available COM port.

    Opening USB serial normally resets the Mega, which prints a distinctive
    HOB CAMERA READY banner from the existing Arduino servo sketch.
    If the banner is absent, GETANGLE provides a safe non-moving check for
    the newer HOB servo sketch. Reject unrelated devices rather than guess.
    """
    if not ENABLE_SERVO:
        print("SERVO: simulation only (ENABLE_SERVO=False)")
        return None

    try:
        import serial
        from serial.tools import list_ports
    except ImportError as exc:
        raise RuntimeError(
            "pyserial is missing: run python -m pip install pyserial"
        ) from exc

    ports = list(list_ports.comports())
    if not ports:
        raise RuntimeError("No serial ports found. Connect Arduino Mega via USB.")

    print("USB/serial ports detected:")
    for port in ports:
        print(f"  {port.device}: {port.description}")

    candidates = [p for p in ports if _is_arduino_candidate(p)]
    # Prefer the official Mega identity, then USB/serial clones (CH340 etc.).
    candidates.sort(key=lambda p: (
        0 if (getattr(p, "vid", None) in (0x2341, 0x2A03)
              or "mega" in (getattr(p, "description", "") or "").lower()) else 1,
        p.device,
    ))
    if not candidates:
        raise RuntimeError(
            "No Arduino-like serial port found. Check Mega USB cable and driver. "
            "ESP32/CP210x ports are intentionally not selected."
        )

    for port in candidates:
        print(f"Checking {port.device} for HOB camera-servo sketch...")
        try:
            mega = serial.Serial(
                port.device, 115200, timeout=0.2, write_timeout=1
            )
        except (serial.SerialException, OSError) as exc:
            print(f"  Cannot open {port.device}: {exc}")
            continue

        accepted = False
        try:
            # Standard Mega auto-reset occurs on opening serial. Wait for boot.
            for line in _read_serial_lines(mega, 3.5):
                if _is_camera_servo_ready(line):
                    accepted = True
                    print(f"  Arduino signature: {line}")
                    break

            if not accepted:
                # Newer camera sketch supports GETANGLE; harmless to the servo.
                mega.reset_input_buffer()
                mega.write(b"GETANGLE\n")
                for line in _read_serial_lines(mega, 0.9):
                    # Only the camera sketch should respond with a plain angle.
                    if line.isdigit() and 0 <= int(line) <= 180:
                        accepted = True
                        print(f"  Servo angle response: {line}")
                        break

            if accepted:
                mega.reset_input_buffer()
                print(f"Arduino Mega CAMERA SERVO detected automatically on {port.device}")
                return mega
            print(f"  {port.device} did not respond as HOB camera servo; skipping.")
        except (serial.SerialException, OSError) as exc:
            print(f"  Probe failed on {port.device}: {exc}")
        mega.close()

    raise RuntimeError(
        "Arduino camera servo was not found. Check USB connection and close "
        "Arduino Serial Monitor. The Mega must be running the HOB camera servo "
        "sketch (startup banner: HOB CAMERA READY). No other board was selected."
    )


def servo_command(mega, command):
    if mega is None:
        print("SIMULATED SERVO:", command)
        return True
    try:
        mega.write((command + "\n").encode("ascii"))
        print("MEGA:", command)
        return True
    except Exception as exc:
        print("SERVO SERIAL ERROR:", exc)
        return False


def open_camera(index):
    for backend in (cv2.CAP_DSHOW, cv2.CAP_ANY):
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAPTURE_WIDTH)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAPTURE_HEIGHT)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            return cap
        cap.release()
    raise RuntimeError(f"Unable to open USB camera {index}. Try CAMERA_INDEX=0")


def object_inference(model, frame, device):
    """YOLO COCO classes — recognized objects, not a safe obstacle avoidance system."""
    prediction = model.predict(frame, imgsz=640, conf=OBJECT_CONFIDENCE,
                               device=device, max_det=60, verbose=False)[0]
    objects = []
    for b in prediction.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        label = prediction.names[int(b.cls[0])]
        objects.append(((x1, y1, x2, y2), label, float(b.conf[0])))
    return {"kind": "objects", "objects": objects}


def best_door(model, frame, device):
    h, w = frame.shape[:2]
    prediction = model.predict(frame, imgsz=640, conf=DOOR_CONFIDENCE,
                               device=device, max_det=12, verbose=False)[0]
    choices = []
    for b in prediction.boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        x1, x2 = max(0, x1), min(w, x2)
        y1, y2 = max(0, y1), min(h, y2)
        if x2 - x1 < 40 or y2 - y1 < 65:
            continue
        confidence = float(b.conf[0])
        score = confidence * ((x2 - x1) * (y2 - y1)) ** 0.25
        choices.append((score, (x1, y1, x2, y2), confidence))
    if not choices:
        return None
    _, box, confidence = max(choices)
    return box, confidence


def above_door_roi(frame, door, last_door_x):
    h, w = frame.shape[:2]
    if door is not None:
        (x1, y1, x2, y2), _ = door
        dw, dh = x2 - x1, y2 - y1
        rect = (max(0, int(x1 - 0.30 * dw)),
                max(0, int(y1 - 0.55 * dh)),
                min(w, int(x2 + 0.30 * dw)),
                min(h, int(y1 + 0.12 * dh)))
    elif last_door_x is not None:
        rect = (max(0, last_door_x[0]), 0,
                min(w, last_door_x[1]), int(h * 0.63))
    else:
        return None
    x1, y1, x2, y2 = rect
    return rect if x2 - x1 >= 40 and y2 - y1 >= 35 else None


def scan_frame(reader, frame):
    """Original user's EasyOCR reading logic, unchanged in behavior.

    It reads any campus room code in the supplied frame and returns
    (polygon, code, floor, number, confidence), plus OCR execution time.
    """
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


def room_inference(door_model, reader, frame, device, last_door_x):
    """Use uploaded EasyOCR code after ARRIVED; prioritize above the door.

    If the overhead region is empty or YOLO doesn't see a door, optionally
    perform the ORIGINAL full-camera scan so a valid room sign isn't lost.
    """
    door = best_door(door_model, frame, device)
    roi = above_door_roi(frame, door, last_door_x)
    rooms = []
    ocr_time = 0.0
    ocr_source = "NO ROI"

    if roi is not None:
        x1, y1, x2, y2 = roi
        crop = frame[y1:y2, x1:x2]
        matches, duration = scan_frame(reader, crop)
        ocr_time += duration
        for points, code, floor, number, confidence in matches:
            # Convert the points from cropped-image to full camera coordinates.
            points = points.copy()
            points[:, 0] += x1
            points[:, 1] += y1
            rooms.append((code, confidence, points))
        if rooms:
            ocr_source = "ABOVE DOOR"

    if not rooms and FULL_FRAME_OCR_FALLBACK:
        # Reuses the exact room-recognition method from the uploaded source.
        matches, duration = scan_frame(reader, frame)
        ocr_time += duration
        for points, code, floor, number, confidence in matches:
            rooms.append((code, confidence, points))
        ocr_source = "FULL CAMERA"

    return {
        "kind": "search",
        "door": door,
        "roi": roi,
        "rooms": rooms,
        "ocr_source": ocr_source,
        "ocr_time": ocr_time,
    }



def run_job(epoch, mode, captured_at, frame, objects_model, door_model,
            reader, device, last_door_x):
    if mode == "NAVIGATING":
        data = object_inference(objects_model, frame, device)
    else:
        data = room_inference(door_model, reader, frame, device, last_door_x)
    return epoch, mode, captured_at, data


def main():
    if PHYSICAL_UP_COMMAND not in ("UP", "DOWN"):
        raise ValueError("PHYSICAL_UP_COMMAND must be UP or DOWN")
    if not 0 <= MAX_UP_STEPS <= 3:
        raise ValueError("Initial safe test requires MAX_UP_STEPS between 0 and 3")

    device = 0 if torch.cuda.is_available() else "cpu"
    print("Loading models. CUDA available:", torch.cuda.is_available())
    objects_model = YOLO("yolov8n.pt")      # 80 common COCO object categories
    door_model = YOLOWorld("yolov8s-world.pt")
    door_model.set_classes(["door", "doorway"])  # Experimental door recognition
    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    mega = None
    camera = None
    udp = None
    state = "NAVIGATING"
    generation = 0
    future = None
    last_job_at = 0.0
    last_result_at = 0.0
    overlays = {}
    status = "MISSION: live object detection; A or UDP ARRIVED to search"
    target = None
    code_candidate = None
    confirmations = 0
    last_door_x = None
    up_steps = 0
    failed_scans = 0
    next_search_at = 0.0
    found = None
    announce_at = 0.0
    finish_result_at = 0.0
    paused = False

    def begin_search(requested=None):
        nonlocal state, generation, target, code_candidate, confirmations
        nonlocal last_door_x, up_steps, failed_scans, next_search_at, found, status
        state = "SEARCH"
        generation += 1
        target = requested
        code_candidate = None
        confirmations = 0
        last_door_x = None
        up_steps = 0
        failed_scans = 0
        found = None
        status = f"ARRIVED: search above door" + (f" for {target}" if target else "")
        if not servo_command(mega, "CENTER"):
            status = "SERVO FAILED: search paused"
            return
        next_search_at = time.perf_counter() + 1.3
        print("MODE: ROOM SEARCH", "target=" + target if target else "any valid room")

    def start_navigation():
        nonlocal state, generation, status, found
        # Face the camera forward again before resuming the live mission view.
        servo_command(mega, "CENTER")
        state = "NAVIGATING"
        generation += 1
        found = None
        status = "MISSION: live object detection"
        print("MODE: NAVIGATING")

    def handle_command(message):
        nonlocal paused
        msg = message.strip().upper()
        if msg.startswith("ARRIVED"):
            parts = msg.split()
            requested = parse_room(parts[1]) if len(parts) == 2 else None
            if len(parts) > 1 and requested is None:
                print("Invalid target room code; valid example: ARRIVED G0070")
            else:
                begin_search(requested)
        elif msg in ("NAVIGATE", "MISSION_START", "RESET"):
            start_navigation()
        elif msg == "STOP":
            paused = True
            print("PAUSED: AI processing (video remains live)")
        elif msg == "RESUME":
            paused = False
            print("RESUMED")
        elif msg:
            print("Unknown external message:", msg)

    try:
        try:
            mega = connect_mega()
        except RuntimeError as exc:
            if REQUIRE_MEGA:
                raise
            print("WARNING: Arduino camera servo not detected:", exc)
            print("Continuing with live camera and AI; servo movements are SIMULATED.")
            mega = None

        camera = open_camera(CAMERA_INDEX)
        print("Camera:", int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
              "x", int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))

        udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp.bind((UDP_HOST, UDP_PORT))
        udp.setblocking(False)
        print(f"Arrival command listener: UDP {UDP_HOST}:{UDP_PORT}")
        print("Keys: A arrived | N navigation | S pause | P snapshot | Q quit")
        print("CAMERA stays open across mission, search and result states")

        with ThreadPoolExecutor(max_workers=1) as pool:
            while True:
                success, frame = camera.read()
                if not success:
                    print("Camera frame read failed")
                    break
                now = time.perf_counter()

                # External mission command can come from a navigation process.
                for _ in range(4):
                    try:
                        message, address = udp.recvfrom(256)
                    except BlockingIOError:
                        break
                    handle_command(message.decode("ascii", errors="ignore"))

                if future is not None and future.done():
                    try:
                        task_generation, task_state, captured_at, data = future.result()
                    except Exception as exc:
                        task_generation, task_state, captured_at, data = -1, "", now, None
                        status = f"VISION ERROR: {str(exc)[:75]}"
                        print(status)
                    future = None
                    if (data is not None and task_generation == generation
                            and now - captured_at <= 30.0):
                        overlays = data
                        last_result_at = now
                        if task_state == "SEARCH" and state == "SEARCH":
                            door = data["door"]
                            if door is not None:
                                x1, _y1, x2, _y2 = door[0]
                                margin = int((x2 - x1) * 0.3)
                                last_door_x = (max(0, x1 - margin),
                                               min(frame.shape[1], x2 + margin))
                            matches = [r for r in data["rooms"]
                                       if target is None or r[0] == target]
                            best = max(matches, key=lambda r: r[1], default=None)
                            if best is not None:
                                failed_scans = 0
                                if best[0] == code_candidate:
                                    confirmations += 1
                                else:
                                    code_candidate = best[0]
                                    confirmations = 1
                                status = (f"VERIFY ROOM {best[0]}: {confirmations}/{ROOM_CONFIRMATIONS} "
                                          f"[{data['ocr_source']}]")
                                if confirmations >= ROOM_CONFIRMATIONS:
                                    found = best[0]
                                    state = "RESULT"
                                    generation += 1
                                    status = f"FOUND {found}: returning camera home"
                                    servo_command(mega, "CENTER")
                                    announce_at = now + 1.2
                            else:
                                code_candidate = None
                                confirmations = 0
                                failed_scans += 1
                                status = ("SEARCH: door found; checking above and full frame" if door
                                          else "SEARCH: no door; OCR checks full camera")
                                # Step once, wait, then analyze a NEW frame.
                                if failed_scans >= 2 and now >= next_search_at:
                                    if up_steps < MAX_UP_STEPS:
                                        if servo_command(mega, PHYSICAL_UP_COMMAND):
                                            up_steps += 1
                                            failed_scans = 0
                                            next_search_at = now + WAIT_AFTER_MOVE
                                            status = f"SEARCH UP {up_steps}/{MAX_UP_STEPS}; settling"
                                    else:
                                        status = "UPPER LIMIT: keep checking; no further movement"
                # A single background job prevents AI inference from blocking preview.
                if future is None and not paused and now - last_job_at >= VISION_MIN_INTERVAL:
                    if state == "NAVIGATING" or (state == "SEARCH" and now >= next_search_at):
                        future = pool.submit(
                            run_job, generation, state, now, frame.copy(),
                            objects_model, door_model, reader, device, last_door_x
                        )
                        last_job_at = now

                if state == "RESULT" and now >= announce_at:
                    floor, number = room_parts(found)
                    room_record = ROOM_NAMES.get(found, {})
                    room_name = room_record.get("en", "Name not registered")
                    room_ar = room_record.get("ar", "اسم القاعة غير مسجل")
                    print(f"ROOM FOUND: {found} | Floor {floor} | Room {number}")
                    print("اسم القاعة:", room_ar)
                    print("Room name:", room_name)
                    status = f"ROOM {found} | {room_name} | HOME"
                    announce_at = float("inf")
                    finish_result_at = now + RESULT_SECONDS
                if state == "RESULT" and finish_result_at and now >= finish_result_at:
                    start_navigation()
                    overlays = {}
                    finish_result_at = 0.0

                display = frame.copy()
                if now - last_result_at <= 2.5:
                    if overlays.get("kind") == "objects" and state == "NAVIGATING":
                        for (x1, y1, x2, y2), label, conf in overlays["objects"]:
                            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 210, 0), 2)
                            cv2.putText(display, f"{label} {conf:.2f}",
                                        (x1, max(23, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (0, 210, 0), 2)
                    if overlays.get("kind") == "search" and state in ("SEARCH", "RESULT"):
                        door = overlays["door"]
                        if door:
                            x1, y1, x2, y2 = door[0]
                            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 160, 0), 3)
                            cv2.putText(display, f"DOOR {door[1]:.2f}",
                                        (x1, max(22, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                                        0.7, (255, 160, 0), 2)
                        if overlays["roi"]:
                            x1, y1, x2, y2 = overlays["roi"]
                            cv2.rectangle(display, (x1, y1), (x2, y2), (0, 220, 220), 2)
                        for code, conf, pts in overlays["rooms"]:
                            cv2.polylines(display, [pts], True, (0, 255, 0), 3)
                            x, y = pts[0]
                            cv2.putText(display, f"{code} ({conf:.2f})", (int(x), max(25, int(y) - 10)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

                if state == "SEARCH" and overlays.get("kind") == "search":
                    cv2.putText(display, f"OCR: {overlays.get('ocr_source', '...')} | "
                                f"{overlays.get('ocr_time', 0.0):.2f}s",
                                (15, 116), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (0, 255, 255), 2)

                cv2.putText(display, f"HOB {state} | SERVO {'ON' if mega is not None else 'SIM'}",
                            (15, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(display, status[:100], (15, 76),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.67, (0, 255, 255), 2)
                h, w = display.shape[:2]
                preview_w = min(w, PREVIEW_WIDTH)
                preview = cv2.resize(display, (preview_w, round(h * preview_w / w)),
                                     interpolation=cv2.INTER_AREA)
                cv2.imshow("HOB Robot - Mission Camera", preview)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("a"):
                    handle_command("ARRIVED")
                if key == ord("n"):
                    handle_command("NAVIGATE")
                if key == ord("s"):
                    handle_command("RESUME" if paused else "STOP")
                if key == ord("p"):
                    filename = SAVE_DIR / f"capture_{datetime.now():%Y%m%d_%H%M%S}.jpg"
                    cv2.imwrite(str(filename), frame)
                    print("Saved", filename)

    except KeyboardInterrupt:
        print("Stopped")
    finally:
        if mega is not None:
            try:
                servo_command(mega, "STOP")
            except Exception:
                pass
            mega.close()
        if udp is not None:
            udp.close()
        if camera is not None:
            camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
