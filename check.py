import cv2
import numpy as np


def nothing(x):
    pass


def calibrate_steering_center():
    cap = cv2.VideoCapture(10)
    if not cap.isOpened() or not cap.read()[0]:
        cap.release()
        cap = cv2.VideoCapture(2)

    cv2.namedWindow("Steering Center Calibrator")
    # The ROI is 160 pixels wide. Start the slider in the mathematical middle (80).
    cv2.createTrackbar("Center Pixel", "Steering Center Calibrator", 80, 160, nothing)

    print("--- 🎯 STEERING CENTER CALIBRATION ---")
    print("1. Park the car straight on the track.")
    print("2. Move the slider until the green line splits the car perfectly in half.")
    print("3. Press 'Q' to quit and get your number.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))

        # 1. Recreate your exact 20% to 50% line detection ROI
        h, w = frame.shape[:2]
        # Shift the vision down to the actual road
        roi_top = int(h * 0.5)  # Start at 50% (Y=240, just below the minimap)
        roi_bot = int(h * 0.8)  # End at 80% (Y=384, right around the car hood)
        roi_frame = frame[roi_top:roi_bot, :]

        # 2. Resize to 160x60 exactly like vision.py does
        small_roi = cv2.resize(roi_frame, (160, 60))

        # 3. Scale it up by 4x so we don't have to squint at a tiny box
        display = cv2.resize(small_roi, (640, 240), interpolation=cv2.INTER_NEAREST)

        # 4. Get trackbar value (0 to 160)
        center_val = cv2.getTrackbarPos("Center Pixel", "Steering Center Calibrator")

        # 5. Draw the vertical line
        # We multiply by 4 because we scaled the 160px image up to 640px
        draw_x = center_val * 4
        cv2.line(display, (draw_x, 0), (draw_x, 240), (0, 255, 0), 2)
        cv2.putText(
            display,
            f"Current Offset: {center_val}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )

        cv2.imshow("Steering Center Calibrator", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("\n" + "=" * 45)
            print(f"✅ YOUR NEW OFFSET IS: {center_val}")
            print(f"Update vision.py to: normalized_x = (cx - {center_val}) / 80.0")
            print("=" * 45)
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    calibrate_steering_center()
