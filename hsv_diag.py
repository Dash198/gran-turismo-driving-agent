"""
HSV Diagnostics — prints actual HSV distribution in the vision ROI.
Run this ONCE and read the output to understand what thresholds to use.

Usage: uv run hsv_diag.py --camera 2
"""
import argparse
import time
import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=2)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("❌ Cannot open camera")
        return

    print("⏳ Capturing 10 frames for diagnostics...\n")
    frames = []
    for _ in range(15):
        cap.grab()  # drain buffer
    for _ in range(10):
        ret, raw = cap.read()
        if ret:
            frames.append(raw)
        time.sleep(0.1)
    cap.release()

    if not frames:
        print("❌ No frames captured")
        return

    # Use the last frame
    frame = frames[-1]
    h, w = frame.shape[:2]
    print(f"Frame size: {w}x{h}")

    # LINE ROI (30-60% height)
    line_roi = frame[int(h * 0.3):int(h * 0.6), :]
    line_small = cv2.resize(line_roi, (160, 60))
    line_hsv = cv2.cvtColor(line_small, cv2.COLOR_BGR2HSV)

    print("\n=== LINE ROI (30-60% height) ===")
    print(f"  H: min={line_hsv[:,:,0].min()}, max={line_hsv[:,:,0].max()}, mean={line_hsv[:,:,0].mean():.1f}")
    print(f"  S: min={line_hsv[:,:,1].min()}, max={line_hsv[:,:,1].max()}, mean={line_hsv[:,:,1].mean():.1f}")
    print(f"  V: min={line_hsv[:,:,2].min()}, max={line_hsv[:,:,2].max()}, mean={line_hsv[:,:,2].mean():.1f}")

    # Test current blue thresholds
    lower_blue = np.array([108, 170, 40])
    upper_blue = np.array([135, 255, 255])
    mask_blue = cv2.inRange(line_hsv, lower_blue, upper_blue)
    pct_blue = (mask_blue.sum() / 255) / mask_blue.size * 100
    print(f"  Blue line match: {pct_blue:.2f}%  (target: 0.5-5%)")

    # ROAD ROI (30-60% height)
    road_roi = frame[int(h * 0.3):int(h * 0.6), :]
    road_small = cv2.resize(road_roi, (160, 60))
    road_hsv = cv2.cvtColor(road_small, cv2.COLOR_BGR2HSV)

    print("\n=== ROAD ROI (30-60% height) ===")
    print(f"  H: min={road_hsv[:,:,0].min()}, max={road_hsv[:,:,0].max()}, mean={road_hsv[:,:,0].mean():.1f}")
    print(f"  S: min={road_hsv[:,:,1].min()}, max={road_hsv[:,:,1].max()}, mean={road_hsv[:,:,1].mean():.1f}")
    print(f"  V: min={road_hsv[:,:,2].min()}, max={road_hsv[:,:,2].max()}, mean={road_hsv[:,:,2].mean():.1f}")

    # Current road thresholds
    lower_road = np.array([0, 0, 50])
    upper_road = np.array([180, 40, 150])
    road_mask = cv2.inRange(road_hsv, lower_road, upper_road)
    pct_road = (road_mask.sum() / 255) / road_mask.size * 100
    print(f"  Road match with [0,0,50]-[180,40,150]: {pct_road:.1f}%  (target: 20-50%)")

    # Percentile breakdown of S channel (key for road detection)
    s_vals = road_hsv[:,:,1].flatten()
    print(f"\n  S channel percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"    {p}th: {np.percentile(s_vals, p):.0f}")

    v_vals = road_hsv[:,:,2].flatten()
    print(f"\n  V channel percentiles:")
    for p in [10, 25, 50, 75, 90]:
        print(f"    {p}th: {np.percentile(v_vals, p):.0f}")

    # Suggest better thresholds
    # Road = pixels where S is low (gray) - find the typical S/V of road pixels
    print("\n=== SUGGESTED ROAD THRESHOLDS ===")
    s_med = np.median(s_vals)
    v_10 = np.percentile(v_vals, 10)
    v_90 = np.percentile(v_vals, 90)
    print(f"  Based on median S={s_med:.0f}, V range={v_10:.0f}-{v_90:.0f}")
    print(f"  Try:  lower = [0, 0, {max(30, int(v_10))}]")
    print(f"        upper = [180, {min(60, int(s_med*2))}, {min(200, int(v_90))}]")

    # Save the ROI for inspection
    cv2.imwrite("/tmp/line_roi.png", line_roi)
    cv2.imwrite("/tmp/road_roi.png", road_roi)
    cv2.imwrite("/tmp/road_mask.png", road_mask)
    cv2.imwrite("/tmp/line_hsv_h.png", line_hsv[:,:,0])
    cv2.imwrite("/tmp/line_hsv_s.png", line_hsv[:,:,1])
    cv2.imwrite("/tmp/line_hsv_v.png", line_hsv[:,:,2])
    print("\n✅ Saved ROI images to /tmp/ for inspection")
    print("   line_roi.png, road_roi.png, road_mask.png")
    print("   line_hsv_h.png, line_hsv_s.png, line_hsv_v.png")


if __name__ == "__main__":
    main()
