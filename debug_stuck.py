import cv2
import numpy as np

from gt_env import VisionInterface


def run_offset_test():
    vision = VisionInterface(camera_index=2)

    # --- THE TEST OFFSET --- [cite: 11-02-2026]
    # We are replacing the hardcoded 80 with your discovered 51.
    CALIBRATED_ZERO = 51

    print(f">>> TESTING OFFSET: {CALIBRATED_ZERO}")
    print(">>> If the car is centered, 'Adjusted Pos' should now be ~0.00")

    while True:
        frame = vision.get_frame()
        if frame is None:
            continue

        # 1. Get the raw CX from your current vision logic [cite: 11-02-2026]
        # Peak into the logic: resize -> hsv -> mask -> contours -> moments
        small = cv2.resize(frame, (160, 120))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, vision.lower_blue, vision.upper_blue)
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        cx_val = 0
        adj_pos = 0.0

        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) > 30:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx_val = int(M["m10"] / M["m00"])

                    # --- THE 51-OFFSET FORMULA --- [cite: 11-02-2026]
                    # This ensures that if cx_val is 51, adj_pos is 0.00.
                    adj_pos = (cx_val - CALIBRATED_ZERO) / 80.0

        # --- Visual Annotations (320x240 for clarity) --- [cite: 11-02-2026]
        display_frame = cv2.resize(frame, (320, 240))

        # Blue line: Your New Zero (51 * 2 = 102 pixels in display width)
        cv2.line(display_frame, (102, 0), (102, 240), (255, 0, 0), 2)

        # Green line: The old broken 80 center (160 pixels in display width)
        cv2.line(display_frame, (160, 0), (160, 240), (0, 255, 0), 1)

        cv2.putText(
            display_frame, f"Raw CX: {cx_val}", (10, 30), 1, 1, (255, 255, 255), 1
        )
        cv2.putText(
            display_frame,
            f"Adjusted Pos: {adj_pos:.2f}",
            (10, 60),
            1,
            1,
            (255, 255, 0),
            2,
        )

        cv2.imshow("Offset Verification", display_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_offset_test()
