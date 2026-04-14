"""
ocr_diag.py — Live OCR diagnostic for speed reading.
Shows the raw ROI, thresholded image, and parsed result.
Press Q to quit.
"""

import cv2
import numpy as np
import pytesseract
from vision import VisionInterface

vis = VisionInterface(camera_index=2)
print("🔍 OCR Speed Diagnostic — Press Q to quit")

read_count = 0
fail_count = 0

while True:
    raw = vis.get_frame()
    if raw is None:
        continue

    y, x, h, w = vis.SPEED_ROI
    speed_roi = raw[y:y+h, x:x+w]

    # Processing pipeline (same as get_speed)
    gray = cv2.cvtColor(speed_roi, cv2.COLOR_BGR2GRAY)
    _, thresh_180 = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
    _, thresh_150 = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    _, thresh_200 = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

    # Try multiple configs
    configs = {
        "PSM7 (line)":    "--psm 7 -c tessedit_char_whitelist=0123456789",
        "PSM8 (word)":    "--psm 8 -c tessedit_char_whitelist=0123456789",
        "PSM13 (raw)":    "--psm 13 -c tessedit_char_whitelist=0123456789",
        "PSM6 (block)":   "--psm 6 -c tessedit_char_whitelist=0123456789",
    }

    thresholds = {
        "Thresh 150": thresh_150,
        "Thresh 180": thresh_180,
        "Thresh 200": thresh_200,
    }

    # === BUILD DISPLAY ===
    W, H = 700, 500
    disp = np.zeros((H, W, 3), dtype=np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    WHITE = (220, 220, 220)
    DIM = (120, 120, 120)
    CYAN = (0, 255, 255)

    # Show where ROI is on the full frame
    frame_preview = raw.copy()
    cv2.rectangle(frame_preview, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(frame_preview, "SPEED ROI", (x, y-5), f, 0.4, (0, 255, 0), 1)
    frame_small = cv2.resize(frame_preview, (280, 210))
    disp[10:220, 10:290] = frame_small
    cv2.putText(disp, f"ROI: y={y} x={x} h={h} w={w}", (10, 240), f, 0.33, DIM, 1)

    # Show ROI zoomed + all thresholds
    roi_big = cv2.resize(speed_roi, (120, 60))
    disp[10:70, 310:430] = roi_big
    cv2.putText(disp, "RAW ROI (zoomed)", (310, 8), f, 0.33, WHITE, 1)

    gray_big = cv2.resize(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), (120, 60))
    disp[10:70, 450:570] = gray_big
    cv2.putText(disp, "GRAYSCALE", (450, 8), f, 0.33, WHITE, 1)

    for i, (name, thresh) in enumerate(thresholds.items()):
        t_big = cv2.resize(cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR), (120, 60))
        x0 = 310 + i * 130
        disp[80:140, x0:x0+120] = t_big
        cv2.putText(disp, name, (x0, 78), f, 0.3, CYAN, 1)

        # White pixel percentage
        pct = np.count_nonzero(thresh) / max(1, thresh.size) * 100
        cv2.putText(disp, f"{pct:.0f}% white", (x0, 155), f, 0.28, DIM, 1)

    # OCR results grid
    cv2.putText(disp, "OCR RESULTS:", (10, 275), f, 0.4, CYAN, 1)
    
    row = 0
    for t_name, thresh in thresholds.items():
        for c_name, config in configs.items():
            text = pytesseract.image_to_string(thresh, config=config).strip()
            try:
                val = int(text)
                color = (0, 255, 0)
                val_str = f"{val} km/h"
            except ValueError:
                color = (0, 0, 255)
                val_str = f"'{text}'" if text else "(empty)"

            y_pos = 295 + row * 16
            if y_pos < H - 20:
                cv2.putText(disp, f"{t_name} + {c_name}:", (10, y_pos), f, 0.28, DIM, 1)
                cv2.putText(disp, val_str, (250, y_pos), f, 0.28, color, 1)
            row += 1

    # Current get_speed() result
    official = vis.get_speed(raw)
    read_count += 1
    if official is None:
        fail_count += 1

    rate = (1 - fail_count / max(1, read_count)) * 100
    off_col = (0, 255, 0) if official is not None else (0, 0, 255)
    off_str = f"{official} km/h" if official is not None else "None"
    cv2.putText(disp, f"get_speed() = {off_str}", (400, 275), f, 0.45, off_col, 2)
    cv2.putText(disp, f"Success rate: {rate:.0f}% ({read_count - fail_count}/{read_count})", 
                (400, 300), f, 0.33, WHITE, 1)

    cv2.putText(disp, "Press Q to quit", (W-150, H-10), f, 0.33, DIM, 1)

    cv2.imshow("OCR Speed Diagnostic", disp)
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

vis.cap.release()
cv2.destroyAllWindows()
print(f"\n📊 Final: {read_count - fail_count}/{read_count} reads successful ({rate:.0f}%)")
