"""
ROI Finder for Gran Turismo Vision Pipeline.

Usage:
  uv run find_rois.py [--camera 2]

Controls:
  Click + Drag   → Draw a rectangle on the live feed
  1              → Assign current rectangle as LAP_ROI
  2              → Assign current rectangle as MAP_ROI
  3              → Assign current rectangle as COL_ROI
  4              → Assign current rectangle as SPEED_ROI
  P              → Print all ROIs in (y, x, h, w) format (paste into vision.py)
  R              → Reset current rectangle
  Q / ESC        → Quit
"""

import argparse
import cv2
import numpy as np

# --- State ---
drawing = False
ix, iy = 0, 0
rect = None  # (x1, y1, x2, y2) in pixel coords
rois = {
    "LAP_ROI": None,
    "MAP_ROI": None,
    "COL_ROI": None,
    "SPEED_ROI": None,
}
KEY_MAP = {
    ord("1"): "LAP_ROI",
    ord("2"): "MAP_ROI",
    ord("3"): "COL_ROI",
    ord("4"): "SPEED_ROI",
}
COLORS = {
    "LAP_ROI": (0, 255, 255),    # Yellow
    "MAP_ROI": (0, 255, 0),      # Green
    "COL_ROI": (0, 0, 255),      # Red
    "SPEED_ROI": (255, 100, 0),  # Blue
}


def mouse_cb(event, x, y, flags, param):
    global drawing, ix, iy, rect

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
        rect = None

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        rect = (ix, iy, x, y)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        rect = (ix, iy, x, y)


def to_yxhw(r):
    """Convert (x1, y1, x2, y2) pixel rect to (y, x, h, w) format used by vision.py."""
    x1, y1, x2, y2 = r
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    return (y1, x1, y2 - y1, x2 - x1)


def main():
    global rect

    parser = argparse.ArgumentParser(description="ROI Finder")
    parser.add_argument("--camera", type=int, default=2, help="Camera index")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"❌ Cannot open camera {args.camera}")
        return

    cv2.namedWindow("ROI Finder")
    cv2.setMouseCallback("ROI Finder", mouse_cb)

    print("=" * 50)
    print("🎯 ROI FINDER — Click & drag to draw rectangles")
    print("=" * 50)
    print("  1 = LAP    2 = MAP    3 = COL    4 = SPEED")
    print("  P = Print  R = Reset  Q = Quit")
    print("=" * 50)

    while True:
        ret, raw = cap.read()
        if not ret:
            break
        frame = cv2.resize(raw, (640, 480))
        display = frame.copy()

        # Draw saved ROIs
        for name, r in rois.items():
            if r is not None:
                y, x, h, w = r
                color = COLORS[name]
                cv2.rectangle(display, (x, y), (x + w, y + h), color, 2)
                cv2.putText(display, name, (x, y - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Draw active selection
        if rect is not None:
            x1, y1, x2, y2 = rect
            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 255, 255), 1)

        # HUD
        cv2.putText(display, "1=LAP 2=MAP 3=COL 4=SPEED | P=Print Q=Quit",
                    (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("ROI Finder", display)
        key = cv2.waitKey(1) & 0xFF

        if key in KEY_MAP and rect is not None:
            name = KEY_MAP[key]
            rois[name] = to_yxhw(rect)
            print(f"  ✅ {name} = {rois[name]}")
            rect = None

        elif key == ord("p"):
            print("\n" + "=" * 50)
            print("📋 PASTE INTO vision.py __init__:")
            print("=" * 50)
            for name, r in rois.items():
                val = r if r is not None else "(0, 0, 0, 0)"
                print(f"        self.{name} = {val}")
            print("=" * 50 + "\n")

        elif key == ord("r"):
            rect = None
            print("  🔄 Selection cleared")

        elif key in (ord("q"), 27):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
