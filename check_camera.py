from picamera2 import Picamera2
import time
import os

# Create output directory
output_dir = "test_photos"
os.makedirs(output_dir, exist_ok=True)

picam2 = Picamera2()

# Configure camera for still capture
config = picam2.create_still_configuration()
picam2.configure(config)

picam2.start()
time.sleep(2)  # warm-up time (important)

for i in range(15):
    filename = os.path.join(output_dir, f"image_{i+1}.jpg")
    picam2.capture_file(filename)
    print(f"Captured {filename}")
    time.sleep(1)

picam2.stop()
