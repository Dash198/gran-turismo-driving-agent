"""
Red Dot Tuner — Adjust HSV thresholds with trackbars until only the car dot shows.
Press S to save thresholds, Q to quit.
"""
import cv2
import numpy as np
from vision import VisionInterface

vision = VisionInterface()

cv2.namedWindow("Tuner", cv2.WINDOW_NORMAL)

# Current thresholds
cv2.createTrackbar("H Lo1", "Tuner", 0, 180, lambda x: None)
cv2.createTrackbar("H Hi1", "Tuner", 10, 180, lambda x: None)
cv2.createTrackbar("H Lo2", "Tuner", 170, 180, lambda x: None)
cv2.createTrackbar("H Hi2", "Tuner", 180, 180, lambda x: None)
cv2.createTrackbar("S Min", "Tuner", 140, 255, lambda x: None)
cv2.createTrackbar("V Min", "Tuner", 100, 255, lambda x: None)
cv2.createTrackbar("Area Min", "Tuner", 1, 50, lambda x: None)
cv2.createTrackbar("Area Max", "Tuner", 80, 500, lambda x: None)

print("Adjust trackbars until ONLY the red car dot is visible in the mask.")
print("Press S to save, Q to quit.")

while True:
    raw = vision.get_frame()
    if raw is None:
        continue

    h_lo1 = cv2.getTrackbarPos("H Lo1", "Tuner")
    h_hi1 = cv2.getTrackbarPos("H Hi1", "Tuner")
    h_lo2 = cv2.getTrackbarPos("H Lo2", "Tuner")
    h_hi2 = cv2.getTrackbarPos("H Hi2", "Tuner")
    s_min = cv2.getTrackbarPos("S Min", "Tuner")
    v_min = cv2.getTrackbarPos("V Min", "Tuner")
    a_min = cv2.getTrackbarPos("Area Min", "Tuner")
    a_max = cv2.getTrackbarPos("Area Max", "Tuner")

    y, x, h, w = vision.MAP_ROI
    map_roi = raw[y:y+h, x:x+w]
    hsv = cv2.cvtColor(map_roi, cv2.COLOR_BGR2HSV)

    # Red mask with trackbar values
    mask1 = cv2.inRange(hsv, np.array([h_lo1, s_min, v_min]), np.array([h_hi1, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([h_lo2, s_min, v_min]), np.array([h_hi2, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [(c, cv2.contourArea(c)) for c in contours if a_min <= cv2.contourArea(c) <= a_max]

    # Draw on map
    map_draw = map_roi.copy()
    if valid:
        best = max(valid, key=lambda x: x[1])[0]
        cv2.drawContours(map_draw, [best], -1, (0, 255, 0), 1)
        M = cv2.moments(best)
        if M["m00"] > 0:
            cx, cy = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
            cv2.circle(map_draw, (cx, cy), 3, (0, 255, 255), -1)

    # Scale up for display
    scale = 4
    map_big = cv2.resize(map_draw, (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
    mask_big = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)

    # Info bar
    info = np.zeros((40, w*scale*2, 3), dtype=np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    n_contours = len(contours)
    n_valid = len(valid)
    best_area = max(valid, key=lambda x: x[1])[1] if valid else 0
    cv2.putText(info, f"Contours: {n_contours}  Valid: {n_valid}  Best area: {best_area:.0f}", (10, 25), f, 0.4, (200,200,200), 1)

    display = np.hstack([map_big, mask_big])
    display = np.vstack([display, info])
    cv2.imshow("Tuner", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        print(f"\n{'='*50}")
        print(f"SAVED THRESHOLDS:")
        print(f"  lower_red1 = np.array([{h_lo1}, {s_min}, {v_min}])")
        print(f"  upper_red1 = np.array([{h_hi1}, 255, 255])")
        print(f"  lower_red2 = np.array([{h_lo2}, {s_min}, {v_min}])")
        print(f"  upper_red2 = np.array([{h_hi2}, 255, 255])")
        print(f"  Area filter: {a_min} - {a_max}")
        print(f"{'='*50}")

        # Auto-update vision.py thresholds
        with open("vision.py", "r") as vf:
            code = vf.read()
        code = code.replace(
            f"self.lower_red1 = np.array([{vision.lower_red1[0]}, {vision.lower_red1[1]}, {vision.lower_red1[2]}])",
            f"self.lower_red1 = np.array([{h_lo1}, {s_min}, {v_min}])"
        )
        code = code.replace(
            f"self.upper_red1 = np.array([{vision.upper_red1[0]}, {vision.upper_red1[1]}, {vision.upper_red1[2]}])",
            f"self.upper_red1 = np.array([{h_hi1}, 255, 255])"
        )
        code = code.replace(
            f"self.lower_red2 = np.array([{vision.lower_red2[0]}, {vision.lower_red2[1]}, {vision.lower_red2[2]}])",
            f"self.lower_red2 = np.array([{h_lo2}, {s_min}, {v_min}])"
        )
        code = code.replace(
            f"self.upper_red2 = np.array([{vision.upper_red2[0]}, {vision.upper_red2[1]}, {vision.upper_red2[2]}])",
            f"self.upper_red2 = np.array([{h_hi2}, 255, 255])"
        )
        with open("vision.py", "w") as vf:
            vf.write(code)
        print("✅ vision.py updated!")

cv2.destroyAllWindows()
