from picamera2 import Picamera2
import time, os

OUT_DIR = "/home/miserasp/Desktop/projecte/shared/"
os.makedirs(OUT_DIR, exist_ok=True)

SLOTS = 10
INTERVAL_SEC = 1
WARMUP = 2.0

picam2 = Picamera2()
picam2.configure(picam2.create_still_configuration())
picam2.start()
time.sleep(WARMUP)

slot = 1
try:
    while True:
        final_path = os.path.join(OUT_DIR, f"slot_{slot:02d}.jpg")
        tmp_path   = os.path.join(OUT_DIR, f"slot_{slot:02d}.part.jpg")  # FIX

        if os.path.exists(final_path):
            time.sleep(0.05)
            continue

        picam2.capture_file(tmp_path)     # now PIL recognizes .jpg
        os.replace(tmp_path, final_path)  # atomic publish
        print("Wrote:", final_path)

        slot = (slot % SLOTS) + 1
        time.sleep(INTERVAL_SEC)

finally:
    picam2.stop()
