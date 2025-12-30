# ring_reader_worker.py (run in conda)
import os, time, cv2

OUT_DIR = "/home/miserasp/Desktop/projecte/shared/"
SLOTS = 10

def process_image(img_bgr, path):
    # TODO: run your detector + embeddings here
    # Example placeholder:
    print("Processing", path, "shape:", img_bgr.shape)

while True:
    did_work = False

    for slot in range(1, SLOTS + 1):
        path = os.path.join(OUT_DIR, f"slot_{slot:02d}.jpg")

        if not os.path.exists(path):
            # “Start reading from 1”: if slot_01 missing, stop the pass early
            if slot == 1:
                break
            continue

        img = cv2.imread(path)
        if img is None:
            # file exists but not readable (rare with atomic rename)
            # wait a bit and try next loop
            time.sleep(0.01)
            continue

        process_image(img, path)

        # delete once processed
        try:
            os.remove(path)
        except FileNotFoundError:
            pass

        did_work = True

    if not did_work:
        time.sleep(0.05)
