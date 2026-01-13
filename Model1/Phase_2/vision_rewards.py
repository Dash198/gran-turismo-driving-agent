
import cv2
import numpy as np

class VisionRewardSystem:
    def __init__(self,
                 w_sym=1.0,
                 w_edge=0.5,
                 w_vp=0.5,
                 w_corner=1.0,
                 w_wrong_way=2.0,
                 w_stuck=0.5,
                 w_line=1.0,
                 crash_threshold=100.0,
                 min_symmetry=0.1,
                 roi_vertical=(0.30, 0.85),
                 roi_horizontal=(0.15, 0.85)):
        
        self.w_sym = w_sym
        self.w_edge = w_edge
        self.w_vp = w_vp
        self.w_corner = w_corner
        self.w_wrong_way = w_wrong_way
        self.w_stuck = w_stuck
        self.w_line = w_line
        
        self.crash_threshold = crash_threshold
        self.min_symmetry = min_symmetry
        
        self.roi_v_min = roi_vertical[0]
        self.roi_v_max = roi_vertical[1]
        self.roi_h_min = roi_horizontal[0]
        self.roi_h_max = roi_horizontal[1]
        
        # State for Road Visibility Gate
        self.consecutive_invisible_steps = 0
        self.road_invisible_threshold = 180  # ~3s at 60fps (relaxed from 30)
        # Thresholds for "Roadness"
        self.max_road_texture = 35.0
        self.min_vp_lines = 1
        self.max_bottom_texture = 25.0
        
        # Blue Driving Line Detection (HSV)
        self.blue_h_low = 100
        self.blue_h_high = 130
        self.blue_s_min = 80
        self.blue_v_min = 50
        self.min_blue_pixels = 100
        
        # Wrong Way Detection (HSV for Red Stop Sign)
        self.red_h1 = (0, 10)
        self.red_h2 = (170, 180)
        self.red_s_min = 150
        self.red_v_min = 100
        self.wrong_way_pixel_threshold = 500
        self.consecutive_wrong_way = 0
        
        # Stuck State tracking
        self.consecutive_stuck_frames = 0
        
    def detect_wrong_way(self, frame_bgr):
        """Detects the red stop sign in the center of the screen."""
        if frame_bgr is None: return False, 0
        
        h, w = frame_bgr.shape[:2]
        # ROI: Lower-center of screen (stop sign flashes below center)
        ry1, ry2 = int(0.4 * h), int(0.7 * h) 
        rx1, rx2 = int(0.3 * w), int(0.7 * w)
        center_roi = frame_bgr[ry1:ry2, rx1:rx2]
        
        hsv = cv2.cvtColor(center_roi, cv2.COLOR_BGR2HSV)
        
        # Red mask
        mask1 = cv2.inRange(hsv, np.array([self.red_h1[0], self.red_s_min, self.red_v_min]), np.array([self.red_h1[1], 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([self.red_h2[0], self.red_s_min, self.red_v_min]), np.array([self.red_h2[1], 255, 255]))
        red_mask = cv2.bitwise_or(mask1, mask2)
        
        red_pixel_count = np.sum(red_mask > 0)
        return red_pixel_count > self.wrong_way_pixel_threshold, red_pixel_count

    def compute_reward(self, frame_bgr, speed=0):
        """
        Computes the vision-based reward from a single BGR frame.
        """
        # 1. Preprocessing
        if len(frame_bgr.shape) == 3:
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame_bgr
            
        h, w = gray.shape
        y1, y2 = int(self.roi_v_min * h), int(self.roi_v_max * h)
        x1, x2 = int(self.roi_h_min * w), int(self.roi_h_max * w)
        roi = gray[y1:y2, x1:x2]
        
        if roi.size == 0:
            return 0.0, {}, False
            
        roi_h, roi_w = roi.shape

        # 2. Road Visibility Check
        blur = cv2.GaussianBlur(roi, (7, 7), 0)
        texture_residual = cv2.absdiff(roi, blur)
        texture_energy = np.mean(texture_residual)
        
        # Bottom-Center Road Check
        bottom_h = int(0.25 * roi_h)
        center_w = int(0.4 * roi_w)
        cx1 = (roi_w - center_w) // 2
        cx2 = cx1 + center_w
        bottom_center = texture_residual[roi_h - bottom_h:, cx1:cx2]
        bottom_texture = np.mean(bottom_center) if bottom_center.size > 0 else 999.0
        
        # VP / Line Structure
        edges = cv2.Canny(roi, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
        
        valid_vp_lines = 0
        vp_lines_list = []
        if lines is not None:
            for line in lines:
                x_start, y_start, x_end, y_end = line[0]
                dx, dy = x_end - x_start, y_end - y_start
                angle = abs(np.arctan2(dy, dx)) if dx != 0 else np.pi/2
                if angle > np.pi/4:
                    valid_vp_lines += 1
                    vp_lines_list.append((x_start, y_start, x_end, y_end, dx, dy))

        # Blue Line Detection & Alignment
        blue_pixels = 0
        r_line_alignment = 0.0
        line_ahead_angle = 0.0
        blue_mask = None
        
        if len(frame_bgr.shape) == 3:
            roi_bgr = frame_bgr[y1:y2, x1:x2]
            hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
            blue_mask = cv2.inRange(hsv, np.array([self.blue_h_low, self.blue_s_min, self.blue_v_min]), np.array([self.blue_h_high, 255, 255]))
            blue_pixels = np.sum(blue_mask > 0)
            
            # --- Line Ahead Alignment (Upper ROI) ---
            if blue_pixels >= self.min_blue_pixels:
                # Upper half of ROI (line ahead)
                upper_blue_mask = blue_mask[:int(0.5 * roi_h), :]
                coords = np.argwhere(upper_blue_mask > 0)
                
                if len(coords) >= 20:
                    # 1. Lateral Offset
                    centroid_x = np.mean(coords[:, 1])
                    center_x = roi_w / 2
                    offset = (centroid_x - center_x) / center_x # [-1, 1]
                    r_line_alignment = max(0, 1.0 - abs(offset))
                    
                    # 2. Heading Angle (polyfit)
                    ys, xs = coords[:, 0], coords[:, 1]
                    if np.std(ys) > 1.0:
                        try:
                            # x = f(y) -> slope is dx/dy
                            p = np.polyfit(ys, xs, 1)
                            slope = p[0]
                            line_ahead_angle = np.arctan(slope) # radians
                        except:
                            line_ahead_angle = 0.0
        
        # Gate Logic
        has_blue = blue_pixels >= self.min_blue_pixels
        has_low_texture = texture_energy <= self.max_road_texture
        has_vp = valid_vp_lines >= self.min_vp_lines
        has_road_bottom = bottom_texture <= self.max_bottom_texture
        
        is_road_visible = has_blue or has_low_texture or has_vp or has_road_bottom
        if not is_road_visible:
            self.consecutive_invisible_steps += 1
        else:
            self.consecutive_invisible_steps = 0
            
        # 3. Standard Components
        gx = cv2.Scharr(roi, cv2.CV_32F, 1, 0)
        gy = cv2.Scharr(roi, cv2.CV_32F, 0, 1)
        grad_mag = np.sqrt(gx**2 + gy**2) / 255.0 

        # 3.1 Edge Density
        edge_density = grad_mag.mean()
        r_edge = -edge_density

        # 3.2 Symmetry
        mid = roi_w // 2
        min_w = min(mid, roi_w - mid)
        left_density = grad_mag[:, mid-min_w:mid].mean()
        right_density = grad_mag[:, mid:mid+min_w].mean()
        symmetry = 1.0 - abs(left_density - right_density) / (left_density + right_density + 1e-6)
        r_sym = symmetry

        # 3.3 VP Alignment
        sum_x_top, count_vp = 0.0, 0
        for (x_start, y_start, x_end, y_end, dx, dy) in vp_lines_list:
            if abs(dy) > 1e-3:
                x_top = x_start + (0 - y_start) * (dx / dy)
                if 0 <= x_top <= roi_w:
                    sum_x_top += x_top
                    count_vp += 1
        vp_metric = abs((sum_x_top / count_vp) - roi_w/2) / roi_w if count_vp > 0 else 0.5
        r_vp = -vp_metric

        # 3.4 Corner Danger
        corner_w, corner_h = int(0.15 * roi_w), int(0.15 * roi_h)
        lc_mean = grad_mag[roi_h-corner_h:, :corner_w].mean()
        rc_mean = grad_mag[roi_h-corner_h:, roi_w-corner_w:].mean()
        # Large corner energy = negative reward
        r_corner = -(lc_mean + rc_mean)

        # 4. Wrong Way & Stuck Logic
        is_wrong_way, red_pixels = self.detect_wrong_way(frame_bgr)
        r_wrong_way = -self.w_wrong_way if is_wrong_way else 0.0
        
        if is_wrong_way:
            self.consecutive_wrong_way += 1
        else:
            self.consecutive_wrong_way = 0
        
        # Stuck Logic: speed < 5 and high corner energy
        # Relaxed: Removed OR with visibility, increased threshold to 0.6
        is_stuck_momentary = speed < 5 and abs(r_corner) > 0.6
        
        if is_stuck_momentary:
            self.consecutive_stuck_frames += 1
        else:
            self.consecutive_stuck_frames = 0
        
        # Only apply penalty and flag as stuck after a grace period (0.5s)
        is_wall_stuck = self.consecutive_stuck_frames > 30
        r_stuck = -self.w_stuck if is_wall_stuck else 0.0

        # 5. Composition
        total_reward = (
            self.w_sym * r_sym +
            self.w_edge * r_edge +
            self.w_vp * r_vp +
            self.w_corner * r_corner +
            self.w_line * r_line_alignment +
            r_wrong_way +
            r_stuck
        )
        
        # 6. Hard Masks & Termination (DISABLED FOR EXPLORATION)
        # All terminations disabled to allow agent to explore and learn recovery
        done = False
        term_reason = ""
        
        # Logging only - no termination
        # if self.consecutive_invisible_steps >= self.road_invisible_threshold:
        #     done, term_reason = True, "road_invisible_gate"
        # elif self.consecutive_wrong_way >= 180:
        #     done, term_reason = True, "wrong_way_prolonged"
        
        # Symmetry mask still applies (zeroes reward, no termination)
        if symmetry < self.min_symmetry:
            total_reward = 0.0
            
        metrics = {
            "vis_edge_density": edge_density,
            "vis_symmetry": symmetry,
            "vis_vp_offset": vp_metric,
            "vis_corner_energy": abs(r_corner),
            "vis_line_offset": 1.0 - r_line_alignment, # 0 = centered, 1 = max off
            "vis_line_angle": float(line_ahead_angle),
            "vis_total_reward": total_reward,
            "vis_texture_energy": texture_energy,
            "vis_red_pixels": red_pixels,
            "vis_is_wrong_way": 1.0 if is_wrong_way else 0.0,
            "vis_is_wall_stuck": 1.0 if is_wall_stuck else 0.0,
            "vis_invisible_steps": self.consecutive_invisible_steps,
            "vis_term_reason": 1.0 if done else 0.0,
            "vis_term_reason_str": term_reason
        }
        
        return total_reward, metrics, done
