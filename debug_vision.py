"""
Live Vision Debug — runs all vision pipeline functions against the camera
and displays results in real-time.

Usage: uv run debug_vision.py --camera 2
"""
import argparse
import time

import cv2
import numpy as np

from vision import VisionInterface


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=2)
    args = parser.parse_args()

    vision = VisionInterface(args.camera)
    prev_progress = None
    prev_pos = None

    print("🔍 Live Vision Debug — press Q to quit")

    while True:
        frame = vision.get_frame()
        if frame is None:
            print("❌ No frame")
            time.sleep(0.5)
            continue

        h, w = frame.shape[:2]
        display = frame.copy()

        # 1. Line detection (single-pass)
        line_channel, line_pos = vision.detect_line(frame)
        line_status = f"LINE: {line_pos:.2f}" if line_pos is not None else "LINE: NONE"
        col = (0, 255, 0) if line_pos is not None else (0, 0, 255)
        cv2.putText(display, line_status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        # 2. Collision
        is_col = vision.check_collision(frame)
        col_text = "COLLISION: YES" if is_col else "COLLISION: no"
        col_c = (0, 0, 255) if is_col else (0, 255, 0)
        cv2.putText(display, col_text, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_c, 2)

        # 3. Speed OCR
        speed = vision.get_speed(frame)
        cv2.putText(display, f"OCR SPEED: {speed}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 4. Progress
        prog = vision.get_progress_percent(frame)
        prog_val = prog[0] if prog and prog[0] is not None else None
        delta = 0.0
        if prog_val is not None and prev_progress is not None:
            delta = prog_val - prev_progress
        prev_progress = prog_val
        cv2.putText(display, f"PROGRESS: {prog_val:.3f} (d={delta:+.4f})" if prog_val else "PROGRESS: NONE",
                     (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        # 5. Map position
        pos = vision.get_map_position(frame)
        disp = 0.0
        if pos is not None and prev_pos is not None:
            disp = np.linalg.norm(np.array(pos) - np.array(prev_pos))
        prev_pos = pos
        cv2.putText(display, f"MAP POS: {pos}  disp={disp:.2f}" if pos else "MAP POS: NONE",
                     (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # 6. Lap
        lap_text = f"LAP: {vision.last_detected_lap}"
        cv2.putText(display, lap_text, (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw ROI boxes on the frame
        rois = {
            "MAP": vision.MAP_ROI,
            "LAP": vision.LAP_ROI,
            "COL": vision.COL_ROI,
            "SPD": vision.SPEED_ROI,
        }
        colors = {"MAP": (0, 255, 0), "LAP": (0, 255, 255), "COL": (0, 0, 255), "SPD": (255, 100, 0)}
        for name, (ry, rx, rh, rw) in rois.items():
            if rh > 0 and rw > 0:
                cv2.rectangle(display, (rx, ry), (rx + rw, ry + rh), colors[name], 2)
                cv2.putText(display, name, (rx, ry - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colors[name], 1)

        # ─── Masks panel ───
        mask_panel = np.zeros((200, 420, 3), dtype=np.uint8)

        # Line mask (distance transform)
        ch0 = cv2.resize(cv2.cvtColor(line_channel, cv2.COLOR_GRAY2BGR), (100, 100))
        mask_panel[10:110, 10:110] = ch0
        cv2.putText(mask_panel, "LINE", (30, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 200, 0), 1)

        # Road mask
        road = vision.get_road_mask(frame)
        ch1 = cv2.resize(cv2.cvtColor(road, cv2.COLOR_GRAY2BGR), (100, 100))
        mask_panel[10:110, 120:220] = ch1
        cv2.putText(mask_panel, "ROAD", (140, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

        # Brake mask
        brake = vision.get_brake_overlay(frame)
        ch2 = cv2.resize(cv2.cvtColor(brake, cv2.COLOR_GRAY2BGR), (100, 100))
        mask_panel[10:110, 230:330] = ch2
        cv2.putText(mask_panel, "BRAKE", (250, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # Map mask
        m_mask = vision.get_map_mask_only(frame)
        ch3 = cv2.resize(cv2.cvtColor(m_mask, cv2.COLOR_GRAY2BGR), (80, 80))
        mask_panel[10:90, 340:420] = ch3
        cv2.putText(mask_panel, "MAP MASK", (345, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 200, 0), 1)

        # COL ROI preview
        y_c, x_c, h_c, w_c = vision.COL_ROI
        if h_c > 0 and w_c > 0:
            col_roi = frame[y_c:y_c+h_c, x_c:x_c+w_c]
            col_thumb = cv2.resize(col_roi, (100, 60))
            mask_panel[135:195, 10:110] = col_thumb
            cv2.putText(mask_panel, "COL ROI", (25, 132), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)

        cv2.imshow("Vision Debug - Feed", display)
        cv2.imshow("Vision Debug - Masks", mask_panel)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vision.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
