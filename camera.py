from picamera2 import Picamera2
import time, os, threading
from evdev import InputDevice, ecodes

OUT_DIR = "/home/miserasp/Desktop/projecte/shared/"
os.makedirs(OUT_DIR, exist_ok=True)

SLOTS = 10
INTERVAL_SEC = 1
CAPTURE_DURATION = 10
WARMUP = 2.0

MOUSE_DEV = "/dev/input/by-id/usb-Logitech_USB_Optical_Mouse-event-mouse"

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()
time.sleep(WARMUP)

slot = 1
capturing = False
lock = threading.Lock()

def capture_for_duration():
    global slot, capturing
    try:
        start = time.time()
        print("Capture started (10s)")

        while time.time() - start < CAPTURE_DURATION:
            # compute paths using current slot safely
            with lock:
                final_path = os.path.join(OUT_DIR, f"slot_{slot:02d}.jpg")
                tmp_path   = os.path.join(OUT_DIR, f"slot_{slot:02d}.part.jpg")

            if os.path.exists(final_path):
                time.sleep(0.05)
                continue

            # capture + publish, then advance slot
            with lock:
                picam2.capture_file(tmp_path)
                os.replace(tmp_path, final_path)
                slot = (slot % SLOTS) + 1

            time.sleep(INTERVAL_SEC)

        print("Capture finished")
    finally:
        capturing = False  # always reset

def start_capture():
    global capturing
    if not capturing:
        capturing = True
        threading.Thread(target=capture_for_duration, daemon=True).start()
    else:
        print("Already capturing, ignoring click")

print("Listening for RIGHT CLICK on:", MOUSE_DEV)
print("Right click to capture for 10 seconds...")

dev = InputDevice(MOUSE_DEV)

try:
    for event in dev.read_loop():
        if event.type == ecodes.EV_KEY and event.code == ecodes.BTN_RIGHT and event.value == 1:
            print("Right click detected")
            start_capture()
finally:
    picam2.stop()
