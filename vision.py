"""
VisionInterface v3 — Hybrid
────────────────────────────
Mask-based observation (v1 quality) + streamlined telemetry (v2 efficiency).
Outputs 64×64 masks (line/road/brake) — no duplicate calls, no frame drain.
"""

import cv2
import numpy as np
import pytesseract


class VisionInterface:
    def __init__(self, camera_index=10):
        # ROI coordinates (y, x, h, w) — calibrated for 640×480
        self.LAP_ROI = (133, 70, 16, 14)
        self.MAP_ROI = (160, 2, 66, 80)
        self.COL_ROI = (317, 95, 66, 154)
        self.SPEED_ROI = (348, 141, 30, 65)

        self.last_detected_lap = 1
        self.lap_read_history = []

        # Map center for progress tracking
        self.map_center = (39.0, 201.0)

        # ── COLOR TUNING ──
        # Blue racing line
        self.lower_blue = np.array([108, 120, 40])
        self.upper_blue = np.array([135, 255, 255])

        # Red braking zones (wraparound hue)
        self.lower_red1 = np.array([0, 140, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 140, 50])
        self.upper_red2 = np.array([180, 255, 255])

        # Road surface (gray asphalt)
        self.lower_road = np.array([0, 0, 50])
        self.upper_road = np.array([180, 40, 150])

        # Reusable morphology kernels
        self._k3 = np.ones((3, 3), np.uint8)

        # Camera setup
        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened() or not self.cap.read()[0]:
            print("⚠️ Camera 10 failed. Switching to Camera 2...")
            self.cap.release()
            self.cap = cv2.VideoCapture(2)
        if not self.cap.isOpened():
            raise RuntimeError("❌ CRITICAL: Could not open Camera 10 OR Camera 2.")
        print("✅ Camera Active!")

    # ═══════════════════════════════════════
    # FRAME CAPTURE
    # ═══════════════════════════════════════

    def get_frame(self):
        """Read single frame, resized to 640×480."""
        if not self.cap.isOpened():
            print("❌ Error: Camera disconnected!")
            return None
        ret, frame = self.cap.read()
        if not ret:
            print("❌ Error: Empty frame!")
            return None
        return cv2.resize(frame, (640, 480))

    # ═══════════════════════════════════════
    # OBSERVATION CHANNELS (all output 64×64)
    # ═══════════════════════════════════════

    def get_line_channel(self, frame):
        """
        Returns (64×64 uint8 distance-gradient, center_pos float or None).
        Single-pass: HSV → blue/red mask → distance transform → gradient.
        """
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.3):int(h * 0.6), :]
        small = cv2.resize(roi, (160, 60))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        mask_blue = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        mask_r1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_r2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask_blue, cv2.bitwise_or(mask_r1, mask_r2))
        mask = cv2.dilate(mask, self._k3, iterations=1)

        # Resize to 64×64
        mask_64 = cv2.resize(mask, (64, 64))

        # Distance transform gradient (bright = close to line)
        if mask_64.max() == 0:
            return np.zeros((64, 64), dtype=np.uint8), None

        dist = cv2.distanceTransform(255 - mask_64, cv2.DIST_L2, 5)
        d_max = dist.max()
        if d_max > 0:
            channel = np.clip((1.0 - dist / d_max) * 255, 0, 255).astype(np.uint8)
        else:
            channel = (mask_64 > 0).astype(np.uint8) * 255

        # Center position from original mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_pos = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= 10:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    center_pos = (cx - 42) / 80.0

        return channel, center_pos

    def get_road_channel(self, frame):
        """Returns 64×64 binary mask of drivable surface."""
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.3):int(h * 0.6), :]
        small = cv2.resize(roi, (160, 60))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        road_mask = cv2.inRange(hsv, self.lower_road, self.upper_road)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, self._k3)
        return cv2.resize(road_mask, (64, 64))

    def get_brake_channel(self, frame):
        """Returns 64×64 brake zone overlay (near-field red detection)."""
        h, w = frame.shape[:2]
        roi = frame[int(h * 0.5):int(h * 0.7), :]
        if roi.size == 0:
            return np.zeros((64, 64), dtype=np.uint8)
        small = cv2.resize(roi, (160, 60))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, np.array([0, 140, 100]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 140, 100]), np.array([180, 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        red_mask = cv2.dilate(red_mask, self._k3, iterations=1)
        return cv2.resize(red_mask, (64, 64))

    # ═══════════════════════════════════════
    # TELEMETRY (for reward/termination)
    # ═══════════════════════════════════════

    def get_speed(self, frame):
        """OCR speed from HUD. Returns int km/h or None."""
        y, x, h, w = self.SPEED_ROI
        if h == 0 or w == 0:
            return None
        speed_roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(speed_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        config = "--psm 7 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(thresh, config=config).strip()
        try:
            return int(text)
        except ValueError:
            return None

    def check_lap_change(self, frame, current_step, fps, speed):
        """Detects lap increment via OCR with 3-read consistency."""
        y, x, h, w = self.LAP_ROI
        if h == 0 or w == 0:
            return False
        if current_step < (15.0 * fps):
            return False
        if speed is None or speed < 10.0:
            return False

        lap_roi = frame[y:y+h, x:x+w]
        gray = cv2.cvtColor(lap_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        config = "--psm 10 -c tessedit_char_whitelist=123"
        text = pytesseract.image_to_string(thresh, config=config).strip()
        try:
            current_lap = int(text)
        except ValueError:
            return False

        self.lap_read_history.append(current_lap)
        self.lap_read_history = self.lap_read_history[-5:]
        if len(self.lap_read_history) >= 3:
            last_three = self.lap_read_history[-3:]
            if (last_three[0] == last_three[1] == last_three[2] == current_lap
                    and current_lap == self.last_detected_lap + 1):
                print(f"🏁 VALID LAP CHANGE: {self.last_detected_lap} -> {current_lap}")
                self.last_detected_lap = current_lap
                self.lap_read_history = []
                return True
        return False

    def get_map_position(self, frame):
        """Returns (x, y) centroid of red dot on minimap, or None."""
        y, x, h, w = self.MAP_ROI
        if h == 0 or w == 0:
            return None
        map_roi = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(map_roi, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        points = cv2.findNonZero(mask)
        if points is not None:
            avg_pos = np.mean(points, axis=0)[0]
            return (float(avg_pos[0]), float(avg_pos[1]))
        return None

    def get_progress_percent(self, frame):
        """Returns (progress 0-1, dist_from_center) via polar angle on minimap."""
        pos = self.get_map_position(frame)
        if pos is None:
            return None, None
        dx = pos[0] - self.map_center[0]
        dy = pos[1] - self.map_center[1]
        angle = np.arctan2(dy, dx)
        progress = (angle + np.pi) / (2 * np.pi)
        dist = np.sqrt(dx * dx + dy * dy)
        return progress, dist

    def check_collision(self, frame):
        """Detects sparks (yellow/orange pixels in collision ROI)."""
        y, x, h, w = self.COL_ROI
        if w == 0 or h == 0:
            return False
        roi = frame[y:y+h, x:x+w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array([15, 100, 200]), np.array([40, 255, 255]))
        return cv2.countNonZero(mask) > 20

    def reset_lap(self):
        self.last_detected_lap = 1
        self.lap_read_history = []
