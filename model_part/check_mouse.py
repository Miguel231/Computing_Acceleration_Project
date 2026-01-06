from evdev import InputDevice, ecodes

dev = InputDevice("/dev/input/by-id/usb-Compx_2.4G_Receiver-if01-event-mouse")
print("Listening for right click...")

for event in dev.read_loop():
    if event.type == ecodes.EV_KEY and event.code == ecodes.BTN_RIGHT:
        print("Right button:", event.value)
