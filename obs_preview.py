"""
obs_preview.py — Live diagnostic: shows exactly what the agent sees.
Press 'Q' to quit.
"""

import cv2
import numpy as np
from vision import VisionInterface

vis = VisionInterface(camera_index=2)

print("🔍 Live Agent Observation Preview — Press Q to quit")

while True:
    raw = vis.get_frame()
    if raw is None:
        continue

    # Exactly what the agent receives
    line_ch, line_pos = vis.get_line_channel(raw)
    road_ch = vis.get_road_channel(raw)
    brake_ch = vis.get_brake_channel(raw)

    # Telemetry (what aux vector uses)
    progress, dist = vis.get_progress_percent(raw)
    speed = vis.get_speed(raw)
    collision = vis.check_collision(raw)
    map_pos = vis.get_map_position(raw)

    # === BUILD DISPLAY ===
    W, H = 800, 500
    disp = np.zeros((H, W, 3), dtype=np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    WHITE = (220, 220, 220)
    DIM = (120, 120, 120)
    CYAN = (0, 255, 255)

    # Row 1: Raw frame (resized) + 3 channels
    raw_small = cv2.resize(raw, (200, 150))
    disp[30:180, 10:210] = raw_small
    cv2.putText(disp, "RAW FRAME (640x480)", (10, 22), f, 0.4, WHITE, 1)

    labels = ["CH0: LINE", "CH1: ROAD", "CH2: BRAKE"]
    colors = [(0, 200, 255), (0, 255, 0), (0, 0, 255)]
    channels = [line_ch, road_ch, brake_ch]

    for i, (ch, label, col) in enumerate(zip(channels, labels, colors)):
        x0 = 230 + i * 180
        ch_bgr = cv2.cvtColor(cv2.resize(ch, (150, 150)), cv2.COLOR_GRAY2BGR)
        # Tint the channel with its color
        tinted = ch_bgr.copy()
        mask = ch > 10
        mask_resized = cv2.resize(mask.astype(np.uint8) * 255, (150, 150))
        for c_idx in range(3):
            tinted[:, :, c_idx] = np.where(mask_resized > 0, 
                np.clip(ch_bgr[:, :, 0].astype(int) * col[c_idx] // 255, 0, 255).astype(np.uint8),
                ch_bgr[:, :, c_idx])
        disp[30:180, x0:x0+150] = tinted
        cv2.putText(disp, label, (x0, 22), f, 0.38, col, 1)

    # Row 2: Stacked observation (what CNN actually gets)
    obs_stack = np.stack([line_ch, road_ch, brake_ch], axis=-1)
    obs_big = cv2.resize(obs_stack, (150, 150))
    disp[210:360, 10:160] = obs_big
    cv2.putText(disp, "STACKED OBS (64x64x3)", (10, 202), f, 0.38, CYAN, 1)

    # Pixel stats for each channel
    cv2.putText(disp, "CHANNEL STATS:", (180, 215), f, 0.38, CYAN, 1)
    for i, (name, ch) in enumerate([("LINE", line_ch), ("ROAD", road_ch), ("BRAKE", brake_ch)]):
        nonzero = np.count_nonzero(ch)
        pct = nonzero / (64 * 64) * 100
        mean_val = ch.mean()
        y = 235 + i * 22
        cv2.putText(disp, f"{name}:  {pct:.1f}% active  mean={mean_val:.0f}  max={ch.max()}", 
                    (180, y), f, 0.35, colors[i], 1)

    # Row 2 right: Telemetry
    tx = 500
    cv2.putText(disp, "TELEMETRY (aux vector):", (tx, 215), f, 0.38, CYAN, 1)
    
    spd_str = f"{speed} km/h" if speed is not None else "None"
    prog_str = f"{progress:.3f}" if progress is not None else "None"
    dist_str = f"{dist:.1f}" if dist is not None else "None"
    pos_str = f"({map_pos[0]:.1f}, {map_pos[1]:.1f})" if map_pos else "None"
    line_str = f"{line_pos:.3f}" if line_pos is not None else "None"

    telemetry = [
        ("OCR Speed:", spd_str),
        ("Progress:", prog_str),
        ("Map Pos:", pos_str),
        ("Map Dist:", dist_str),
        ("Line Pos:", line_str),
        ("Collision:", str(collision)),
    ]
    for i, (label, val) in enumerate(telemetry):
        y = 240 + i * 20
        cv2.putText(disp, label, (tx, y), f, 0.33, DIM, 1)
        col = (0, 255, 0) if val != "None" and val != "False" else (0, 0, 255) if val == "None" else WHITE
        cv2.putText(disp, val, (tx + 90, y), f, 0.33, col, 1)

    # Row 3: ROI overlays on raw frame
    cv2.putText(disp, "ROI OVERLAY:", (10, 380), f, 0.38, CYAN, 1)
    roi_frame = raw.copy()
    # Draw all ROIs
    rois = [
        ("MAP", vis.MAP_ROI, (0, 0, 255)),
        ("SPD", vis.SPEED_ROI, (0, 255, 0)),
        ("LAP", vis.LAP_ROI, (255, 0, 0)),
        ("COL", vis.COL_ROI, (0, 255, 255)),
    ]
    for name, (ry, rx, rh, rw), color in rois:
        cv2.rectangle(roi_frame, (rx, ry), (rx+rw, ry+rh), color, 2)
        cv2.putText(roi_frame, name, (rx, ry-5), f, 0.35, color, 1)
    
    # Vision processing ROIs
    h, w = raw.shape[:2]
    # Line detection area (30-60%)
    cv2.rectangle(roi_frame, (0, int(h*0.3)), (w, int(h*0.6)), (255, 200, 0), 1)
    cv2.putText(roi_frame, "LINE ROI", (5, int(h*0.3)+12), f, 0.3, (255, 200, 0), 1)
    # Brake detection area (50-70%)
    cv2.rectangle(roi_frame, (0, int(h*0.5)), (w, int(h*0.7)), (0, 100, 255), 1)
    cv2.putText(roi_frame, "BRAKE ROI", (5, int(h*0.5)+12), f, 0.3, (0, 100, 255), 1)

    roi_small = cv2.resize(roi_frame, (350, 110))
    disp[388:498, 10:360] = roi_small

    # FPS indicator
    cv2.putText(disp, "Press Q to quit", (W-140, H-10), f, 0.35, DIM, 1)

    cv2.imshow("Agent Observation Preview", disp)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

vis.cap.release()
cv2.destroyAllWindows()
print("✅ Preview closed.")
