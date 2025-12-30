import os, time, cv2

OUT_DIR = "/home/miserasp/Desktop/projecte/shared/"
SLOTS = 10

def process_image(img_bgr, path):
    print("Processing", path, "shape:", img_bgr.shape)

while True:
    did_work = False

    for slot in range(1, SLOTS + 1):
        path = os.path.join(OUT_DIR, f"slot_{slot:02d}.jpg")

        if not os.path.exists(path):
            continue  

        img = cv2.imread(path)
        if img is None:
            time.sleep(0.01)
            continue

        process_image(img, path)

        try:
            os.remove(path)  # frees the slot for camera
        except FileNotFoundError:
            pass

        did_work = True

    if not did_work:
        time.sleep(0.05)
