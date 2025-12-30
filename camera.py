from picamera2 import Picamera2
from pynput import mouse
import time, os, threading

OUT_DIR = "/home/miserasp/Desktop/projecte/shared/"
os.makedirs(OUT_DIR, exist_ok=True)

SLOTS = 10
INTERVAL_SEC = 1
WARMUP = 2.0
CAPTURE_DURATION = 10  # seconds

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()
time.sleep(WARMUP)

slot = 1
capturing = False
lock = threading.Lock()


def capture_for_duration():
    global slot, capturing

    start_time = time.time()

    while time.time() - start_time < CAPTURE_DURATION:
        with lock:
            final_path = os.path.join(OUT_DIR, f"slot_{slot:02d}.jpg")
            tmp_path = os.path.join(OUT_DIR, f"slot_{slot:02d}.part.jpg")

            if os.path.exists(final_path):
                time.sleep(0.05)
                continue

            picam2.capture_file(tmp_path)
            os.replace(tmp_path, final_path)

            slot = (slot % SLOTS) + 1

        time.sleep(INTERVAL_SEC)

    capturing = False


def on_click(x, y, button, pressed):
    global capturing

    if button == mouse.Button.right and pressed:
        if not capturing:
            capturing = True
            threading.Thread(target=capture_for_duration, daemon=True).start()
        else:
            print("Already capturing, ignoring click")



try:
    with mouse.Listener(on_click=on_click) as listener:
        listener.join()
finally:
    picam2.stop()
