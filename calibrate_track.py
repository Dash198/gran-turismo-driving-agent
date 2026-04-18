"""
Track Calibration — Drive one lap to record the track path.
Saves waypoints to track_path.npy for use in training.
Press S to start recording, Q to stop and save.
"""
import cv2
import numpy as np
import time
from vision import VisionInterface

vision = VisionInterface()
positions = []
recording = False
step = 0

print("=" * 50)
print("TRACK CALIBRATION")
print("1. Get to the starting position")
print("2. Press S to start recording")
print("3. Drive one full lap at normal speed")
print("4. Press Q to stop and save")
print("=" * 50)

while True:
    raw = vision.get_frame()
    if raw is None:
        time.sleep(0.05)
        continue

    pos = vision.get_map_position(raw)

    # Dashboard
    db = np.zeros((120, 400, 3), dtype=np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX

    if pos:
        cv2.putText(db, f"Pos: ({pos[0]:.1f}, {pos[1]:.1f})", (10, 25), f, 0.4, (200, 200, 200), 1)

    if recording:
        step += 1
        if pos is not None:
            # Only add if moved enough from last point (filter jitter)
            if len(positions) == 0 or np.linalg.norm(np.array(pos) - np.array(positions[-1])) > 0.3:
                positions.append(pos)

        cv2.putText(db, "RECORDING", (10, 55), f, 0.5, (0, 0, 255), 2)
        cv2.putText(db, f"Points: {len(positions)}", (10, 80), f, 0.4, (0, 255, 255), 1)
        cv2.putText(db, f"Step: {step}", (200, 80), f, 0.4, (100, 100, 100), 1)

        # Draw path so far
        if len(positions) > 1:
            path_img = np.zeros((66, 80, 3), dtype=np.uint8)
            pts = np.array(positions, dtype=np.int32)
            for i in range(1, len(pts)):
                cv2.line(path_img, tuple(pts[i-1]), tuple(pts[i]), (0, 255, 0), 1)
            cv2.circle(path_img, tuple(pts[0]), 3, (255, 0, 0), -1)  # Start = blue
            cv2.circle(path_img, tuple(pts[-1]), 3, (0, 0, 255), -1)  # Current = red
            path_big = cv2.resize(path_img, (160, 132))
            db[0:120, 240:400] = path_big[:120, :]
    else:
        cv2.putText(db, "Press S to start", (10, 55), f, 0.5, (0, 255, 0), 1)

    cv2.putText(db, "Q=save+quit  S=start", (10, 110), f, 0.3, (100, 100, 100), 1)
    cv2.imshow("Track Calibration", db)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('s') and not recording:
        recording = True
        positions = []
        step = 0
        print("🔴 Recording started! Drive one lap.")
    elif key == ord('q'):
        break

cv2.destroyAllWindows()

if len(positions) > 10:
    path = np.array(positions, dtype=np.float32)

    # Compute cumulative distance along path
    dists = np.zeros(len(path))
    for i in range(1, len(path)):
        dists[i] = dists[i-1] + np.linalg.norm(path[i] - path[i-1])
    total_dist = dists[-1]

    # Create 50 equally-spaced waypoints along the path
    NUM_WP = 50
    wp_dists = np.linspace(0, total_dist, NUM_WP, endpoint=False)
    waypoints = np.zeros((NUM_WP, 2), dtype=np.float32)
    for i, d in enumerate(wp_dists):
        idx = np.searchsorted(dists, d)
        idx = min(idx, len(path) - 1)
        waypoints[i] = path[idx]

    np.save("track_path.npy", path)
    np.save("track_waypoints.npy", waypoints)
    print(f"\n✅ Saved!")
    print(f"   Path points: {len(path)}")
    print(f"   Total distance: {total_dist:.1f} px")
    print(f"   Waypoints: {NUM_WP}")
    print(f"   Files: track_path.npy, track_waypoints.npy")
else:
    print(f"\n❌ Not enough points ({len(positions)}). Drive longer!")
