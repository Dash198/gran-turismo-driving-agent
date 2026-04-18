"""
Track Calibration — Tune red dot detection + record one lap.
1. Adjust sliders until ONLY the car dot is visible in the mask
2. Press S to start recording
3. Drive one lap
4. Press Q to save
"""
import cv2
import numpy as np
import time
from vision import VisionInterface

vision = VisionInterface()

WIN = "Track Calibration"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)

# Tuning trackbars
cv2.createTrackbar("S Min", WIN, 140, 255, lambda x: None)
cv2.createTrackbar("V Min", WIN, 100, 255, lambda x: None)
cv2.createTrackbar("Area Min", WIN, 1, 50, lambda x: None)
cv2.createTrackbar("Area Max", WIN, 80, 500, lambda x: None)

positions = []
recording = False
step = 0

print("=" * 50)
print("TRACK CALIBRATION")
print("1. Adjust S Min / V Min sliders until only car dot shows")
print("2. Press S to start recording, then drive one lap")
print("3. Press Q to stop and save")
print("=" * 50)

def detect_dot(map_roi, s_min, v_min, a_min, a_max):
    """Detect red dot with current trackbar settings."""
    hsv = cv2.cvtColor(map_roi, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(hsv, np.array([0, s_min, v_min]), np.array([10, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([170, s_min, v_min]), np.array([180, 255, 255]))
    mask = cv2.bitwise_or(mask1, mask2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [(c, cv2.contourArea(c)) for c in contours if a_min <= cv2.contourArea(c) <= a_max]

    pos = None
    if valid:
        best = max(valid, key=lambda x: x[1])[0]
        M = cv2.moments(best)
        if M["m00"] > 0:
            pos = (M["m10"] / M["m00"], M["m01"] / M["m00"])

    return pos, mask, len(contours), len(valid)

while True:
    raw = vision.get_frame()
    if raw is None:
        time.sleep(0.05)
        continue

    s_min = cv2.getTrackbarPos("S Min", WIN)
    v_min = cv2.getTrackbarPos("V Min", WIN)
    a_min = cv2.getTrackbarPos("Area Min", WIN)
    a_max = cv2.getTrackbarPos("Area Max", WIN)

    y, x, h, w = vision.MAP_ROI
    map_roi = raw[y:y+h, x:x+w]
    pos, mask, n_all, n_valid = detect_dot(map_roi, s_min, v_min, a_min, a_max)

    if recording:
        step += 1
        if pos is not None:
            if len(positions) == 0 or np.linalg.norm(np.array(pos) - np.array(positions[-1])) > 0.3:
                positions.append(pos)

    # Build display
    scale = 4
    map_draw = map_roi.copy()
    if pos:
        cv2.circle(map_draw, (int(pos[0]), int(pos[1])), 2, (0, 255, 255), -1)

    map_big = cv2.resize(map_draw, (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)
    mask_big = cv2.resize(cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), (w*scale, h*scale), interpolation=cv2.INTER_NEAREST)

    # Draw recorded path on map
    if len(positions) > 1:
        pts = np.array(positions) * scale
        for i in range(1, len(pts)):
            p1 = (int(pts[i-1][0]), int(pts[i-1][1]))
            p2 = (int(pts[i][0]), int(pts[i][1]))
            cv2.line(map_big, p1, p2, (0, 255, 0), 1)

    top = np.hstack([map_big, mask_big])

    # Info bar
    info = np.zeros((50, top.shape[1], 3), dtype=np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX

    if recording:
        cv2.putText(info, f"RECORDING  Points: {len(positions)}  Step: {step}", (10, 18), f, 0.4, (0, 0, 255), 1)
    else:
        cv2.putText(info, "Tune sliders, then press S to start", (10, 18), f, 0.4, (0, 255, 0), 1)

    pos_str = f"({pos[0]:.1f}, {pos[1]:.1f})" if pos else "NONE"
    cv2.putText(info, f"Pos: {pos_str}  Contours: {n_all}  Valid: {n_valid}", (10, 38), f, 0.35, (200, 200, 200), 1)

    display = np.vstack([top, info])
    cv2.imshow(WIN, display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not recording:
        recording = True
        positions = []
        step = 0
        print(f"🔴 Recording! S_min={s_min}, V_min={v_min}, Area={a_min}-{a_max}")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

if len(positions) > 10:
    path = np.array(positions, dtype=np.float32)

    # Compute cumulative distance
    dists = np.zeros(len(path))
    for i in range(1, len(path)):
        dists[i] = dists[i-1] + np.linalg.norm(path[i] - path[i-1])
    total_dist = dists[-1]

    # 50 equally-spaced waypoints
    NUM_WP = 50
    wp_dists = np.linspace(0, total_dist, NUM_WP, endpoint=False)
    waypoints = np.zeros((NUM_WP, 2), dtype=np.float32)
    for i, d in enumerate(wp_dists):
        idx = np.searchsorted(dists, d)
        idx = min(idx, len(path) - 1)
        waypoints[i] = path[idx]

    np.save("track_path.npy", path)
    np.save("track_waypoints.npy", waypoints)

    # Save tuned thresholds to vision.py
    import re
    with open("vision.py", "r") as vf:
        code = vf.read()
    code = re.sub(r"self\.lower_red1 = np\.array\(\[.*?\]\)", f"self.lower_red1 = np.array([0, {s_min}, {v_min}])", code)
    code = re.sub(r"self\.upper_red1 = np\.array\(\[.*?\]\)", f"self.upper_red1 = np.array([10, 255, 255])", code)
    code = re.sub(r"self\.lower_red2 = np\.array\(\[.*?\]\)", f"self.lower_red2 = np.array([170, {s_min}, {v_min}])", code)
    code = re.sub(r"self\.upper_red2 = np\.array\(\[.*?\]\)", f"self.upper_red2 = np.array([180, 255, 255])", code)
    with open("vision.py", "w") as vf:
        vf.write(code)

    print(f"\n✅ Saved!")
    print(f"   Path points: {len(path)}")
    print(f"   Total distance: {total_dist:.1f} px")
    print(f"   Waypoints: {NUM_WP}")
    print(f"   Thresholds: S>={s_min}, V>={v_min}, Area {a_min}-{a_max}")
    print(f"   Files: track_path.npy, track_waypoints.npy, vision.py updated")
else:
    print(f"\n❌ Not enough points ({len(positions)}). Drive longer!")
