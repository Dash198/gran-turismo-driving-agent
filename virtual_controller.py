import time
from typing import Dict, Sequence

import numpy as np
from evdev import AbsInfo, UInput
from evdev import ecodes as e


class VirtualController:
    def __init__(self):
        capabilities: Dict[int, Sequence[int]] = {
            e.EV_KEY: [
                e.BTN_A,  # Gas (Cross)
                e.BTN_B,  # Circle
                e.BTN_X,  # Brake (Square)
                e.BTN_Y,  # Triangle
                e.BTN_MODE,  # PPSSPP: Load State
                e.BTN_THUMBL,  # PPSSPP: Next Save State Slot
            ],
            e.EV_ABS: {
                e.ABS_X: AbsInfo(
                    value=0, min=-32767, max=32767, fuzz=0, flat=0, resolution=0
                ),
                e.ABS_Y: AbsInfo(
                    value=0, min=-32767, max=32767, fuzz=0, flat=0, resolution=0
                ),
            },
        }

        self.ui = UInput(
            events=capabilities,
            name="Microsoft X-Box 360 pad",
            vendor=0x045E,
            product=0x028E,
            version=0x1,
        )
        self.current_slot = 0  # Track which slot we are theoretically on
        time.sleep(1)

    def load_save_state(self, target_slot: int):
        """
        Precise cycling logic for PPSSPP with 5 total slots (0-4).
        """
        # Safety check: We only want to use your 3 prepared slots
        if not (0 <= target_slot <= 2):
            target_slot = 0

        # 1. Calculate steps in a 5-slot universe
        # If current=2, target=0: (0 - 2) % 5 = 3 taps (to skip 3 and 4)
        steps_to_tap = (target_slot - self.current_slot) % 5

        if steps_to_tap > 0:
            print(
                f"🔄 Cycling slots: {self.current_slot} -> {target_slot} via {steps_to_tap} taps"
            )
            for _ in range(steps_to_tap):
                self._tap(e.BTN_THUMBL)
                time.sleep(0.15)

        self.current_slot = target_slot

        # 2. Execute Load
        time.sleep(0.4)  # Slightly longer delay to ensure UI has switched
        self._tap(e.BTN_MODE)
        print(f"🏁 Slot {target_slot} Loaded successfully.")

    def step(self, steering, gas_brake, reset=False):
        if reset:
            # You can call load_random_state(5) from the Env reset instead
            return

        # Steering and Gas/Brake logic remains the same
        steer_val = int(steering * 32767)
        self.ui.write(e.EV_ABS, e.ABS_X, steer_val)

        if gas_brake > 0.1:
            self.ui.write(e.EV_KEY, e.BTN_A, 1)
            self.ui.write(e.EV_KEY, e.BTN_X, 0)
        elif gas_brake < -0.1:
            self.ui.write(e.EV_KEY, e.BTN_A, 0)
            self.ui.write(e.EV_KEY, e.BTN_X, 1)
        else:
            self.ui.write(e.EV_KEY, e.BTN_A, 0)
            self.ui.write(e.EV_KEY, e.BTN_X, 0)

        self.ui.syn()

    def _tap(self, button):
        self.ui.write(e.EV_KEY, button, 1)
        self.ui.syn()
        time.sleep(0.1)
        self.ui.write(e.EV_KEY, button, 0)
        self.ui.syn()

    def close(self):
        self.ui.close()
