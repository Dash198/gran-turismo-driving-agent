import time
from typing import Dict, Sequence

from evdev import AbsInfo, UInput
from evdev import ecodes as e


class VirtualController:
    def __init__(self):
        # 1. Define the Capabilities (Standard Gamepad Layout)
        capabilities: Dict[int, Sequence[int]] = {
            e.EV_KEY: [
                e.BTN_A,  # Cross
                e.BTN_B,  # Circle
                e.BTN_X,  # Square
                e.BTN_Y,  # Triangle
                e.BTN_MODE,  # Reset (Guide Button)
            ],
            e.EV_ABS: {
                # Left Analog Stick
                e.ABS_X: AbsInfo(
                    value=0, min=-32767, max=32767, fuzz=0, flat=0, resolution=0
                ),
                e.ABS_Y: AbsInfo(
                    value=0, min=-32767, max=32767, fuzz=0, flat=0, resolution=0
                ),
            },
        }

        # 2. THE SPOOF: Identity Theft
        # Vendor: 0x045e (Microsoft)
        # Product: 0x028e (Xbox 360 Controller)
        # PPSSPP will see this and say "Ah, a valid controller."
        self.ui = UInput(
            events=capabilities,
            name="Microsoft X-Box 360 pad",
            vendor=0x045E,
            product=0x028E,
            version=0x1,
        )
        time.sleep(1)

    def step(self, steering, gas_brake, reset=False):
        # 1. HANDLE RESET (Guide Button)
        if reset:
            self._tap(e.BTN_MODE)
            return

        # 2. HANDLE STEERING (Analog X)
        steer_val = int(steering * 32767)
        self.ui.write(e.EV_ABS, e.ABS_X, steer_val)

        # 3. HANDLE GAS/BRAKE (Buttons)
        # Gas = A (Cross), Brake = X (Square) - Standard Layout
        if gas_brake > 0.1:
            self.ui.write(e.EV_KEY, e.BTN_A, 1)  # Gas
            self.ui.write(e.EV_KEY, e.BTN_X, 0)
        elif gas_brake < -0.1:
            self.ui.write(e.EV_KEY, e.BTN_A, 0)
            self.ui.write(e.EV_KEY, e.BTN_X, 1)  # Brake
        else:
            self.ui.write(e.EV_KEY, e.BTN_A, 0)
            self.ui.write(e.EV_KEY, e.BTN_X, 0)

        self.ui.syn()

    def _tap(self, button):
        self.ui.write(e.EV_KEY, button, 1)
        self.ui.syn()
        time.sleep(0.2)
        self.ui.write(e.EV_KEY, button, 0)
        self.ui.syn()

    def close(self):
        self.ui.close()
