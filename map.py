import time

from evdev import ecodes as e

from virtual_controller import VirtualController


def map_controller_buttons():
    vc = VirtualController()
    print("🚀 Heartbeat started. Go to PPSSPP -> Settings -> Controls -> Mapping")
    print("Assign 'Next Save State Slot' to the button that flickers now.")
    print("Press Ctrl+C to stop once mapped.")

    try:
        while True:
            print("Tapping L3 (BTN_THUMBL)...")
            vc._tap(e.BTN_THUMBL)
            time.sleep(2.0)
    except KeyboardInterrupt:
        print("\nMapping finished.")
        vc.close()


if __name__ == "__main__":
    map_controller_buttons()
