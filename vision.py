import cv2
import numpy as np
import pytesseract


class VisionInterface:
    def __init__(self, camera_index=10):
        # --- 1. PASTE YOUR ROI NUMBERS HERE ---
        # Run find_rois_fixed.py, press 'P', and copy the values here.
        self.LAP_ROI = (15, 90, 18, 18)
        self.MAP_ROI = (44, 20, 70, 91)
        self.COL_ROI = (219, 111, 65, 185)
        self.SPEED_ROI = (247, 171, 35, 59)
        # Inside your Vision class __init__
        self.last_detected_lap = 1

        # State variables for new sensors
        self.prev_map_pos = None
        self.lap_ref_image = None
        self.cap = cv2.VideoCapture(camera_index)  # Or 2, depending on your setup

        if not self.cap.isOpened() or not self.cap.read()[0]:
            print("⚠️ Camera 10 failed. Switching to Camera 2...")
            self.cap.release()
            self.cap = cv2.VideoCapture(2)

        if not self.cap.isOpened():
            raise RuntimeError("❌ CRITICAL: Could not open Camera 10 OR Camera 2.")

        print("✅ Camera Active!")

        # --- 2. COLOR TUNING (Do not touch if working) ---
        # Blue Line (Your Tuned Values)
        self.lower_blue = np.array([108, 170, 40])
        self.upper_blue = np.array([135, 255, 255])

        # Red Line (For Braking Zones - Wraparound)
        self.lower_red1 = np.array([0, 140, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 140, 50])
        self.upper_red2 = np.array([180, 255, 255])

    def get_speed(self, frame):
        y, x, h, w = self.SPEED_ROI
        if h == 0 or w == 0:
            return None

        speed_roi = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(speed_roi, cv2.COLOR_BGR2GRAY)

        # Binary threshold (Adjust 180 if digits are dim) [cite: 11-02-2026]
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # config optimized for single line digits [cite: 11-02-2026]
        config = "--psm 7 -c tessedit_char_whitelist=0123456789"
        text = pytesseract.image_to_string(thresh, config=config).strip()

        try:
            return int(text)
        except ValueError:
            return None

    def get_map_mask_only(self, frame):
        y, x, h, w = self.MAP_ROI
        if h == 0 or w == 0:
            return np.zeros((100, 100), dtype=np.uint8)

        map_roi = frame[y : y + h, x : x + w]
        hsv = cv2.cvtColor(map_roi, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        return cv2.bitwise_or(mask1, mask2)

    def check_lap_change(self, frame, current_step, fps, speed):
        """
        Uses OCR to detect if the lap number has incremented,
        gated by physical constraints to prevent false positives.
        """
        y, x, h, w = self.LAP_ROI
        if h == 0 or w == 0:
            return False

        # --- THE REVIVAL GATES ---
        # 1. Time Gate: Physically impossible to finish a lap in < 15s
        # (Adjust this based on the track, 15s is a safe "anti-glitch" floor)
        if current_step < (15.0 * fps):
            return False

        # 2. Speed Gate: You must be moving to cross the finish line
        if speed is None or speed < 10.0:
            return False

        # --- OCR PROCESSING ---
        lap_roi = frame[y : y + h, x : x + w]
        gray = cv2.cvtColor(lap_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        config = "--psm 10 -c tessedit_char_whitelist=123"
        text = pytesseract.image_to_string(thresh, config=config).strip()

        try:
            current_lap = int(text)
            if current_lap > self.last_detected_lap:
                # 3. Validation Gate: Prevent jumps from 1 -> 3
                # This ensures we only count incremental progress
                if current_lap == self.last_detected_lap + 1:
                    print(
                        f"🏁 VALID LAP CHANGE: {self.last_detected_lap} -> {current_lap}"
                    )
                    self.last_detected_lap = current_lap
                    return True
        except ValueError:
            pass

        return False

    def reset_lap(self):
        self.last_detected_lap = 1

    def get_frame(self):
        if not self.cap.isOpened():
            print("❌ Error: Camera disconnected!")
            return None

        ret, frame = self.cap.read()
        if not ret:
            print("❌ Error: Empty frame!")
            return None
        # Resize to standard size (optional, but good for consistency)
        return cv2.resize(frame, (640, 480))

    def process_obs(self, frame):
        # Returns the 84x84 Grayscale Mask for the Agent
        line_pos = self.detect_line_center(frame, return_mask=True)
        return cv2.resize(line_pos, (84, 84))

    def detect_line_center(self, frame, return_mask=False):
        # 1. Resize & Color Threshold
        small = cv2.resize(frame, (160, 120))
        hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)

        # 2. Blue Mask ONLY (No Red)
        mask_combined = cv2.inRange(hsv, self.lower_blue, self.upper_blue)

        # 3. CONNECT THE DOTS (Dilation)
        # Inflate the blue dots so they touch and become a "Line"
        kernel = np.ones((5, 5), np.uint8)
        mask_combined = cv2.dilate(mask_combined, kernel, iterations=2)

        if return_mask:
            return mask_combined

        # 4. Find Center
        contours, _ = cv2.findContours(
            mask_combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)

        # Threshold: 30 pixels (Safe now because we dilated)
        if cv2.contourArea(largest) < 30:
            return None

        M = cv2.moments(largest)
        if M["m00"] == 0:
            return None

        cx = int(M["m10"] / M["m00"])
        normalized_x = (cx - 51) / 80.0
        return normalized_x

    # --- NEW: BRAKE LINE DETECTOR (Tunnel Vision) ---
    def detect_brake_line(self, frame):
        # 1. Define the "Tunnel" (ROI)
        # We only look in the center-bottom of the screen where the road is.
        # Format: frame[y_start:y_end, x_start:x_end]
        # Assuming 640x480 resolution:
        # y: 240 to 400 (The road surface)
        # x: 200 to 440 (The center lane)
        roi = frame[240:400, 200:440]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 2. Red Masks (Wraparound 0-180)
        # Tuned for the bright red driving line
        lower_red1 = np.array([0, 140, 100])
        upper_red1 = np.array([10, 255, 255])
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)

        lower_red2 = np.array([170, 140, 100])
        upper_red2 = np.array([180, 255, 255])
        mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

        mask = cv2.bitwise_or(mask1, mask2)

        # 3. Filter Noise (Dilation + Area Check)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

        # Count red pixels
        red_pixels = cv2.countNonZero(mask)

        # If we see a significant blob of red (e.g. > 50 pixels), it's a braking zone
        return red_pixels > 50

    def get_map_position(self, frame):
        # 1. Get the raw mask [cite: 11-02-2026]
        mask = self.get_map_mask_only(frame)

        # 2. Find all white pixels (no contour filtering, no area checks)
        points = cv2.findNonZero(mask)

        if points is not None:
            # 3. Average everything. If there's red noise, it'll still shift when the car moves. [cite: 11-02-2026]
            avg_pos = np.mean(points, axis=0)[0]
            # MUST RETURN FLOATS for sub-pixel movement [cite: 11-02-2026]
            return (float(avg_pos[0]), float(avg_pos[1]))

        return None

    # --- NEW SENSOR 3: COLLISION (Sparks) ---
    def check_collision(self, frame):
        x, y, w, h = self.COL_ROI
        if w == 0 or h == 0:
            return False

        roi = frame[y : y + h, x : x + w]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Look for Yellow/Orange Sparks (Hue 15-40)
        # High Sat/Val to ignore white text
        mask = cv2.inRange(hsv, np.array([15, 100, 200]), np.array([40, 255, 255]))
        count = cv2.countNonZero(mask)

        return count > 20  # Hit!
