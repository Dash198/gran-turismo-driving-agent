"""
frame_inspect.py — Check raw frame for black bars / aspect ratio issues.
Press Q to quit.
"""

import cv2
import numpy as np
from vision import VisionInterface

vis = VisionInterface(camera_index=2)
print("🔍 Frame Inspector — Press Q to quit, S to save a frame")

while True:
    raw = vis.get_frame()  # 640×480
    if raw is None:
        continue

    h, w = raw.shape[:2]
    disp = np.zeros((550, 700, 3), dtype=np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    WHITE = (220, 220, 220)
    CYAN = (0, 255, 255)
    DIM = (120, 120, 120)

    # Show raw frame
    raw_small = cv2.resize(raw, (400, 300))
    disp[30:330, 10:410] = raw_small
    cv2.putText(disp, f"RAW FRAME ({w}x{h})", (10, 22), f, 0.4, WHITE, 1)

    # Analyze columns: find where content starts and ends
    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)

    # Column brightness profile (average brightness per column)
    col_means = gray.mean(axis=0)  # shape: (640,)
    # Row brightness profile  
    row_means = gray.mean(axis=1)  # shape: (480,)

    # Find content bounds (where mean brightness > 10)
    threshold = 10
    active_cols = np.where(col_means > threshold)[0]
    active_rows = np.where(row_means > threshold)[0]

    if len(active_cols) > 0:
        content_left = active_cols[0]
        content_right = active_cols[-1]
    else:
        content_left, content_right = 0, w - 1

    if len(active_rows) > 0:
        content_top = active_rows[0]
        content_bottom = active_rows[-1]
    else:
        content_top, content_bottom = 0, h - 1

    content_w = content_right - content_left + 1
    content_h = content_bottom - content_top + 1
    aspect = content_w / max(1, content_h)

    # Draw content bounds on raw frame overlay
    scale_x = 400 / w
    scale_y = 300 / h
    cl = int(content_left * scale_x) + 10
    cr = int(content_right * scale_x) + 10
    ct = int(content_top * scale_y) + 30
    cb = int(content_bottom * scale_y) + 30
    cv2.rectangle(disp, (cl, ct), (cr, cb), (0, 255, 0), 2)
    cv2.putText(disp, "CONTENT", (cl, ct - 5), f, 0.3, (0, 255, 0), 1)

    # Info panel
    ix = 430
    cv2.putText(disp, "FRAME ANALYSIS:", (ix, 40), f, 0.4, CYAN, 1)
    info = [
        f"Frame size:    {w} x {h}",
        f"Content area:  x={content_left}-{content_right}  y={content_top}-{content_bottom}",
        f"Content size:  {content_w} x {content_h}",
        f"Aspect ratio:  {aspect:.2f}:1",
        f"",
        f"Left black:    {content_left} px",
        f"Right black:   {w - content_right - 1} px",
        f"Top black:     {content_top} px",
        f"Bottom black:  {h - content_bottom - 1} px",
    ]
    for i, line in enumerate(info):
        color = (0, 0, 255) if ("black" in line.lower() and not line.endswith("0 px")) else WHITE
        cv2.putText(disp, line, (ix, 65 + i * 20), f, 0.35, color, 1)

    # PSP native = 480x272 = 1.76:1
    cv2.putText(disp, f"PSP native aspect: 1.76:1", (ix, 260), f, 0.33, DIM, 1)
    cv2.putText(disp, f"Detected aspect:   {aspect:.2f}:1", (ix, 280), f, 0.33, 
                (0, 255, 0) if abs(aspect - 1.76) < 0.1 else (0, 0, 255), 1)

    # Column brightness graph
    gy = 350
    cv2.putText(disp, "Column Brightness (left to right):", (10, gy), f, 0.35, CYAN, 1)
    graph_w, graph_h = 680, 80
    for i in range(0, w, 2):
        bx = 10 + int(i / w * graph_w)
        by = gy + 15 + graph_h - int(col_means[i] / 255 * graph_h)
        cv2.circle(disp, (bx, by), 1, (0, 255, 0), -1)
    # Draw content bounds
    cv2.line(disp, (10 + int(content_left / w * graph_w), gy + 15), 
             (10 + int(content_left / w * graph_w), gy + 15 + graph_h), (0, 0, 255), 1)
    cv2.line(disp, (10 + int(content_right / w * graph_w), gy + 15), 
             (10 + int(content_right / w * graph_w), gy + 15 + graph_h), (0, 0, 255), 1)

    # Row brightness graph
    ry = 460
    cv2.putText(disp, "Row Brightness (top to bottom):", (10, ry), f, 0.35, CYAN, 1)
    for i in range(0, h, 2):
        bx = 10 + int(i / h * graph_w)
        by = ry + 15 + graph_h - int(row_means[i] / 255 * graph_h)
        cv2.circle(disp, (bx, by), 1, (0, 200, 255), -1)

    cv2.putText(disp, "Press Q=quit, S=save frame", (430, 540), f, 0.33, DIM, 1)

    cv2.imshow("Frame Inspector", disp)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        cv2.imwrite("/home/devansh/repos/gran-turismo-driving-agent/raw_frame_dump.png", raw)
        print(f"💾 Saved raw frame: {w}x{h}, content={content_left}-{content_right} x {content_top}-{content_bottom}")

vis.cap.release()
cv2.destroyAllWindows()
