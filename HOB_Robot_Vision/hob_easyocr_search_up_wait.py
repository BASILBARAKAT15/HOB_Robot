
import argparse
import re
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import easyocr
import torch

# University room number system
ROOM = re.compile(r"^(?:B[12]|G0|[1-4])[0-9]{3}$")

MIN_CONFIDENCE = 0.55
MAX_STEPS = 3  # 3 steps x 5 degrees = 15 degrees


def find_rooms(reader, image, width):
    h, w = image.shape[:2]
    scale = min(1.0, width / w)

    if scale < 1.0:
        image = cv2.resize(
            image,
            (round(w * scale), round(h * scale))
        )

    results = []

    for box, text, confidence in reader.readtext(
        image,
        detail=1,
        allowlist="BG0123456789"
    ):
        code = re.sub(
            r"[^A-Z0-9]", "", text.upper()
        )

        if ROOM.fullmatch(code) and confidence >= MIN_CONFIDENCE:
            x = int(box[0][0] / scale)
            y = int(box[0][1] / scale)
            results.append(
                (code, float(confidence), x, y)
            )

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--camera", type=int, default=1)
    parser.add_argument("--port", default="COM7")
    parser.add_argument("--enable-motors", action="store_true")
    parser.add_argument(
        "--up-command",
        choices=["UP", "DOWN"],
        default="DOWN"
    )
    parser.add_argument(
        "--wait-seconds", type=float, default=2.5
    )
    parser.add_argument(
        "--ocr-width", type=int, default=1280
    )

    args = parser.parse_args()

    if args.wait_seconds < 0.5:
        parser.error("Wait must be at least 0.5 seconds")

    if not 480 <= args.ocr_width <= 1920:
        parser.error("OCR width must be 480..1920")

    print("Loading EasyOCR; CUDA:", torch.cuda.is_available())

    reader = easyocr.Reader(
        ["en"],
        gpu=torch.cuda.is_available()
    )

    mega = None
    camera = None

    try:
        if args.enable_motors:
            import serial

            try:
                mega = serial.Serial(
                    args.port,
                    115200,
                    timeout=0,
                    write_timeout=1
                )
            except serial.SerialException as exc:
                raise SystemExit(
                    f"Cannot open {args.port}: {exc}"
                )

            time.sleep(2)
            mega.reset_input_buffer()
            print("LIVE SERVO:", args.port)
        else:
            print("SIMULATION (motor will NOT move)")

        camera = cv2.VideoCapture(
            args.camera, cv2.CAP_DSHOW
        )

        if not camera.isOpened():
            camera.release()
            camera = cv2.VideoCapture(args.camera)

        if not camera.isOpened():
            raise SystemExit(
                "Cannot open camera; try --camera 0"
            )

        camera.set(
            cv2.CAP_PROP_FOURCC,
            cv2.VideoWriter_fourcc(*"MJPG")
        )
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        camera.set(cv2.CAP_PROP_FPS, 30)

        steps = 0
        found = None
        last_candidate = None
        confirmations = 0
        paused = False
        next_scan = 0.0
        future = None
        last_boxes = []
        status = "Searching valid university room codes"

        print("Q: quit | S: pause/resume")
        print("Physical UP command:", args.up_command)

        with ThreadPoolExecutor(max_workers=1) as pool:
            while True:
                ok, frame = camera.read()

                if not ok:
                    break

                now = time.perf_counter()

                if future is not None and future.done():
                    try:
                        last_boxes = future.result()

                        match = (
                            max(last_boxes, key=lambda r: r[1])
                            if last_boxes else None
                        )

                        if match:
                            code = match[0]

                            if code == last_candidate:
                                confirmations += 1
                            else:
                                confirmations = 1

                            last_candidate = code

                            status = (
                                f"Checking {code}: "
                                f"{confirmations}/2"
                            )

                            if confirmations >= 2:
                                found = code
                                floor = (
                                    "G" if code[:-3] == "G0"
                                    else code[:-3]
                                )

                                status = (
                                    f"FOUND {code} | "
                                    f"FLOOR {floor} | "
                                    f"ROOM {code[-3:]} | HOLD"
                                )
                                print(status)

                        else:
                            confirmations = 0
                            last_candidate = None

                            if not paused and steps < MAX_STEPS:
                                if mega:
                                    mega.write(
                                        (args.up_command + "\n")
                                        .encode("ascii")
                                    )
                                    print("MEGA:", args.up_command)
                                else:
                                    print("SIMULATED:", args.up_command)

                                steps += 1

                                next_scan = (
                                    time.perf_counter()
                                    + args.wait_seconds
                                )

                                status = (
                                    f"Tilt UP {steps}/{MAX_STEPS}; "
                                    f"wait {args.wait_seconds}s"
                                )

                            elif steps >= MAX_STEPS:
                                status = (
                                    "TOP LIMIT - HOLD; "
                                    "still checking OCR"
                                )

                    except Exception as exc:
                        status = "OCR error; no motor movement"
                        print(status, exc)
                        next_scan = time.perf_counter() + 1.0

                    future = None

                if (
                    not paused
                    and found is None
                    and future is None
                    and now >= next_scan
                ):
                    future = pool.submit(
                        find_rooms,
                        reader,
                        frame.copy(),
                        args.ocr_width
                    )
                    next_scan = now + 0.1

                view = frame.copy()

                for code, conf, x, y in last_boxes:
                    cv2.putText(
                        view,
                        f"{code} ({conf:.2f})",
                        (x, max(30, y - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.85,
                        (0, 255, 0),
                        2
                    )

                cv2.putText(
                    view,
                    status,
                    (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                h, w = view.shape[:2]

                preview = cv2.resize(
                    view,
                    (960, round(960 * h / w))
                )

                cv2.imshow(
                    "HOB Robot - Room Search: UP & WAIT",
                    preview
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                if key == ord("s"):
                    paused = not paused
                    print(
                        "PAUSED" if paused else "RESUMED"
                    )

    except KeyboardInterrupt:
        print("Stopped")

    finally:
        if mega:
            try:
                mega.write(b"STOP\n")
            except Exception:
                pass
            mega.close()

        if camera:
            camera.release()

        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
