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
        self._last_map_pos = None  # Temporal filter for red dot noise
        self._map_miss_count = 0   # Reset lock after consecutive misses

        # Map center for progress tracking (polar angle fallback)
        self.map_center = (39.0, 41.0)

        # Load calibrated track path (if exists)
        import os
        wp_path = os.path.join(os.path.dirname(__file__), 'track_waypoints.npy')
        if os.path.exists(wp_path):
            self.track_waypoints = np.load(wp_path)
            print(f"\u2705 Loaded {len(self.track_waypoints)} track waypoints")
        else:
            self.track_waypoints = None
            print("\u26a0\ufe0f  No track_waypoints.npy — using polar angle fallback")

        # ── COLOR TUNING ──
        # Blue racing line
        self.lower_blue = np.array([108, 120, 40])
        self.upper_blue = np.array([135, 255, 255])

        # Red braking zones (wraparound hue)
        self.lower_red1 = np.array([0, 140, 94])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 140, 94])
        self.upper_red2 = np.array([180, 255, 255])

        # Road surface (gray asphalt)
        self.lower_road = np.array([0, 0, 50])
        self.upper_road = np.array([180, 40, 150])

        # Reusable morphology kernels
        self._k3 = np.ones((3, 3), np.uint8)
        self._k5 = np.ones((5, 5), np.uint8)

        # Game content bounds (capture card black bar crop)
        # Detected: game occupies x=0-349, y=120-383 in 640×480 frame
        self.GAME_X0, self.GAME_X1 = 0, 350
        self.GAME_Y0, self.GAME_Y1 = 120, 384

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

    def _get_game_content(self, frame):
        """Crop to actual game area (remove capture card black bars)."""
        return frame[self.GAME_Y0:self.GAME_Y1, self.GAME_X0:self.GAME_X1]

    # ═══════════════════════════════════════
    # OBSERVATION CHANNELS (all output 64×64)
    # ═══════════════════════════════════════

    def get_line_channel(self, frame):
        """
        Returns (64×64 uint8 dilated mask, center_pos float or None).
        Uses cropped game area → HSV → blue/red mask → dilate.
        """
        game = self._get_game_content(frame)
        h, w = game.shape[:2]
        roi = game[int(h * 0.15):int(h * 0.55), :]
        small = cv2.resize(roi, (160, 80))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        mask_blue = cv2.inRange(hsv, self.lower_blue, self.upper_blue)
        mask_r1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask_r2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask_blue, cv2.bitwise_or(mask_r1, mask_r2))
        mask = cv2.dilate(mask, self._k5, iterations=2)  # Thick dilation for clear signal

        # Resize to 64×64
        channel = cv2.resize(mask, (64, 64))

        if channel.max() == 0:
            return np.zeros((64, 64), dtype=np.uint8), None

        # Center position from processing mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        center_pos = None
        if contours:
            largest = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest) >= 10:
                M = cv2.moments(largest)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    center_pos = (cx - 80) / 80.0  # Centered for 160-wide

        return channel, center_pos

    def get_road_channel(self, frame):
        """Returns 64×64 binary mask of drivable surface."""
        game = self._get_game_content(frame)
        h, w = game.shape[:2]
        roi = game[int(h * 0.15):int(h * 0.55), :]
        small = cv2.resize(roi, (160, 80))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        road_mask = cv2.inRange(hsv, self.lower_road, self.upper_road)
        road_mask = cv2.morphologyEx(road_mask, cv2.MORPH_CLOSE, self._k3)
        return cv2.resize(road_mask, (64, 64))

    def get_brake_channel(self, frame):
        """Returns 64×64 brake zone overlay (near-field red detection)."""
        game = self._get_game_content(frame)
        h, w = game.shape[:2]
        roi = game[int(h * 0.45):int(h * 0.75), :]
        if roi.size == 0:
            return np.zeros((64, 64), dtype=np.uint8)
        small = cv2.resize(roi, (160, 80))
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
        # Upscale 3× — Tesseract struggles with tiny images
        big = cv2.resize(thresh, (w * 3, h * 3), interpolation=cv2.INTER_NEAREST)
        # Add white border padding for Tesseract context
        big = cv2.copyMakeBorder(big, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
        config = "--psm 8 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(big, config=config).strip()
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

        # Tighten: require decent saturation + value (car dot is bright red, noise is dim)
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        bright_mask = cv2.bitwise_and(mask, cv2.inRange(sat, 100, 255))
        bright_mask = cv2.bitwise_and(bright_mask, cv2.inRange(val, 80, 255))

        # Find contours — pick the one with area closest to car dot size
        contours, _ = cv2.findContours(bright_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        # Filter by area: car dot is ~1-80 px²
        valid = [(c, cv2.contourArea(c)) for c in contours if 1 <= cv2.contourArea(c) <= 80]
        if not valid:
            # Fallback: use any contour
            valid = [(c, max(cv2.contourArea(c), 0.1)) for c in contours]

        # Pick the largest valid contour
        best = max(valid, key=lambda x: x[1])[0]
        M = cv2.moments(best)
        if M["m00"] == 0:
            return None
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        candidate = (float(cx), float(cy))

        # Temporal filter: reject teleports (noise jumps randomly, car moves ~1px/frame)
        if self._last_map_pos is not None:
            jump = np.linalg.norm(np.array(candidate) - np.array(self._last_map_pos))
            if jump > 8.0:
                self._map_miss_count += 1
                if self._map_miss_count > 20:  # Lost tracking, re-acquire
                    self._last_map_pos = None
                    self._map_miss_count = 0
                return None  # Don't return stale position
        self._map_miss_count = 0
        self._last_map_pos = candidate
        return candidate

    def get_progress_percent(self, frame):
        """Returns (progress 0-1, dist) via calibrated path or polar angle fallback."""
        pos = self.get_map_position(frame)
        if pos is None:
            return None, None

        if self.track_waypoints is not None:
            # Path-based: find nearest waypoint
            pos_arr = np.array(pos)
            dists = np.linalg.norm(self.track_waypoints - pos_arr, axis=1)
            nearest_idx = np.argmin(dists)
            progress = nearest_idx / len(self.track_waypoints)
            return progress, float(dists[nearest_idx])
        else:
            # Polar angle fallback
            dx = pos[0] - self.map_center[0]
            dy = pos[1] - self.map_center[1]
            angle = np.arctan2(dy, dx)
            progress = (-angle + np.pi) / (2 * np.pi)
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
        self._last_map_pos = None
        self._map_miss_count = 0
