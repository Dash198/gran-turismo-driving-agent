"""
HSV Threshold Tuner — adjust color detection thresholds interactively.

Usage: uv run tune_hsv.py --camera 2
  - Use trackbars to adjust H/S/V min/max
  - Toggle between LINE ROI and ROAD ROI with keys 1/2
  - Press P to print current values
  - Press Q to quit
"""
import argparse
import cv2
import numpy as np


def nothing(x):
    pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=2)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    cv2.namedWindow("HSV Tuner")
    cv2.createTrackbar("H min", "HSV Tuner", 108, 180, nothing)
    cv2.createTrackbar("H max", "HSV Tuner", 135, 180, nothing)
    cv2.createTrackbar("S min", "HSV Tuner", 170, 255, nothing)
    cv2.createTrackbar("S max", "HSV Tuner", 255, 255, nothing)
    cv2.createTrackbar("V min", "HSV Tuner", 40, 255, nothing)
    cv2.createTrackbar("V max", "HSV Tuner", 255, 255, nothing)

    mode = "LINE"  # or "ROAD"

    print("=" * 50)
    print("🎨 HSV TUNER")
    print("  1 = LINE mode (blue line thresholds)")
    print("  2 = ROAD mode (road surface thresholds)")
    print("  P = Print current values")
    print("  Q = Quit")
    print("=" * 50)

    while True:
        # Drain buffer
        for _ in range(2):
            cap.grab()
        ret, raw = cap.read()
        if not ret:
            break
        frame = cv2.resize(raw, (640, 480))
        h, w = frame.shape[:2]

        # Apply ROI based on mode
        if mode == "LINE":
            roi = frame[int(h * 0.3):int(h * 0.6), :]
        else:
            roi = frame[int(h * 0.3):int(h * 0.85), :]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        h_min = cv2.getTrackbarPos("H min", "HSV Tuner")
        h_max = cv2.getTrackbarPos("H max", "HSV Tuner")
        s_min = cv2.getTrackbarPos("S min", "HSV Tuner")
        s_max = cv2.getTrackbarPos("S max", "HSV Tuner")
        v_min = cv2.getTrackbarPos("V min", "HSV Tuner")
        v_max = cv2.getTrackbarPos("V max", "HSV Tuner")

        lower = np.array([h_min, s_min, v_min])
        upper = np.array([h_max, s_max, v_max])
        mask = cv2.inRange(hsv, lower, upper)

        # Count matched pixels
        total = mask.size
        matched = np.count_nonzero(mask)
        pct = (matched / total) * 100

        # Display
        masked_roi = cv2.bitwise_and(roi, roi, mask=mask)

        # Add info text
        cv2.putText(masked_roi, f"Mode: {mode} | Match: {pct:.1f}%",
                     (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(masked_roi, f"[{h_min},{s_min},{v_min}] - [{h_max},{s_max},{v_max}]",
                     (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Side by side: ROI + mask + masked
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        combined = np.hstack([roi, mask_bgr, masked_roi])
        combined = cv2.resize(combined, (960, 240))

        cv2.imshow("HSV Tuner", combined)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("1"):
            mode = "LINE"
            cv2.setTrackbarPos("H min", "HSV Tuner", 108)
            cv2.setTrackbarPos("H max", "HSV Tuner", 135)
            cv2.setTrackbarPos("S min", "HSV Tuner", 170)
            cv2.setTrackbarPos("S max", "HSV Tuner", 255)
            cv2.setTrackbarPos("V min", "HSV Tuner", 40)
            cv2.setTrackbarPos("V max", "HSV Tuner", 255)
            print("  🔵 Switched to LINE mode")

        elif key == ord("2"):
            mode = "ROAD"
            cv2.setTrackbarPos("H min", "HSV Tuner", 0)
            cv2.setTrackbarPos("H max", "HSV Tuner", 180)
            cv2.setTrackbarPos("S min", "HSV Tuner", 0)
            cv2.setTrackbarPos("S max", "HSV Tuner", 60)
            cv2.setTrackbarPos("V min", "HSV Tuner", 80)
            cv2.setTrackbarPos("V max", "HSV Tuner", 220)
            print("  🛣️  Switched to ROAD mode")

        elif key == ord("p"):
            print(f"\n  📋 {mode} thresholds:")
            print(f"     lower = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"     upper = np.array([{h_max}, {s_max}, {v_max}])")
            print(f"     Match %: {pct:.1f}%\n")

        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
