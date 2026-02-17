import time
from collections import deque

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from virtual_controller import VirtualController
from vision import VisionInterface


class GranTurismoEnv(gym.Env):
    def __init__(self, camera_index=0):
        super().__init__()
        self.controller = VirtualController()
        self.vision = VisionInterface(camera_index)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 1), dtype=np.uint8
        )

        self.max_steps = 10000
        self.current_step = 0
        self.missed_line_frames = 0
        self.stuck_frames = 0
        self.loiter_frames = 0
        # --- NEW: Reward Tracker for terminal stats ---
        self.episode_reward = 0.0
        self.pos_buffer = deque(maxlen=int(100))
        self.pcurrent_speed_val = 0.5
        self.current_speed = 0

    def _print_episode_stats(self, reason):
        """Standardized terminal output for Arch Linux CLI."""
        print("\n" + "=" * 45)
        print(f"🏁  EPISODE SUMMARY | {reason}")
        print("-" * 45)
        print(f"   Steps:       {self.current_step}")
        print(f"   Total Rew:   {self.episode_reward:.2f}")
        print(
            f"   Avg/Step:    {(self.episode_reward / max(1, self.current_step)):.4f}"
        )
        print(f"   Laps Done:   {self.vision.last_detected_lap - 1}")
        print("=" * 45 + "\n")

    def step(self, action):
        step_start_time = time.time()

        # 1. ACT
        self.controller.step(steering=action[0], gas_brake=action[1])

        # 2. OBSERVE
        raw_frame = self.vision.get_frame()
        if raw_frame is None:
            return np.zeros((84, 84, 1), dtype=np.uint8), 0, True, False, {}

        # 3. CALCULATE SENSORS & FPS
        line_pos = self.vision.detect_line_center(raw_frame)
        is_collision = self.vision.check_collision(raw_frame)
        m_mask = self.vision.get_map_mask_only(raw_frame)

        duration = time.time() - step_start_time
        fps = 1.0 / duration if duration > 0 else 300.0
        dt = 1.0 / fps

        # Displacement Update [cite: 11-02-2026]
        current_pos = self.vision.get_map_position(raw_frame)
        if current_pos is not None:
            self.pos_buffer.append(current_pos)
        elif len(self.pos_buffer) > 0:
            self.pos_buffer.append(self.pos_buffer[-1])

        displacement = 0.0
        if len(self.pos_buffer) >= 2:
            displacement = np.linalg.norm(
                np.array(self.pos_buffer[-1]) - np.array(self.pos_buffer[0])
            )

        # --- 4. CONSOLIDATED EVOLUTIONARY REWARD ---

        # --- 4. LINE-DEPENDENT REWARD (ANTI-PIT FARMING) --- [cite: 11-02-2026]

        has_line = line_pos is not None

        # A. Centering (Delta-based when line is present)
        current_center_error = abs(line_pos) if has_line else 1.0
        r_centering = (
            getattr(self, "prev_center_error", 0.5) - current_center_error
        ) * 10.0
        self.prev_center_error = current_center_error

        # B. Progress Logic (Noise-Safe) [cite: 11-02-2026, 12-02-2026]
        is_moving = displacement > 2.0
        last_speed = getattr(self, "current_speed", 0) or 0

        if has_line and is_moving:
            # Profit Zone: Gas input + Velocity breadcrumb [cite: 11-02-2026]
            r_progress = (10.0 * action[1] * dt) + ((last_speed / 5.0) * dt)
        else:
            # Existence Tax: Slashed to near-zero [cite: 12-02-2026]
            r_progress = -50.0 * dt

        # C. Instant Penalties [cite: 11-02-2026]
        r_collision = -50.0 if is_collision else 0.0
        r_stuck_drain = (
            0.0  # Placeholder for telemetry compatibility [cite: 12-02-2026]
        )

        r_loiter = 0.0
        if last_speed < 2.0:  # If going slower than a crawl
            r_loiter = -10.0 * dt

        reward = r_centering + r_progress + r_collision + r_loiter

        terminated = False

        # --- LAP & TERMINATIONS ---
        if self.current_step % 30 == 0:
            self.current_speed = self.vision.get_speed(raw_frame)

            if self.vision.check_lap_change(
                raw_frame, self.current_step, fps, self.current_speed
            ):
                reward = 1000.0  # Rescaled: Lap > several seconds of driving [cite: 11-02-2026]
                terminated = True
                self._print_episode_stats("LAP COMPLETED! 🏁")

            # Update Loiter/Stuck frames normally...
            if self.current_speed is not None and self.current_speed < 3:
                self.loiter_frames += 30
            else:
                self.loiter_frames = 0

        reason = ""
        if not terminated:
            if line_pos is None:
                self.missed_line_frames += 1
                if self.missed_line_frames > (1.0 * fps):  # 1.5s grace
                    reward, terminated = (
                        -150.0,
                        True,
                    )  # Rescaled terminal hit [cite: 11-02-2026]
                    reason = "LINE LOST"
            else:
                self.missed_line_frames = 0

            # Displacement check for Stuck logic
            if displacement < 5.0:
                self.stuck_frames += 1
            else:
                self.stuck_frames = 0

            if self.stuck_frames > (4.0 * fps):  # 4s grace
                reward, terminated = -150.0, True
                reason = "STUCK POSITION"

            if self.loiter_frames > (3.0 * fps):
                reward, terminated = -200.0, True
                reason = "LOITERING DETECTED"

        # 6. RENDER (Aligned with 1.5s/4.0s logic) [cite: 11-02-2026]
        self._render_dynamic_telemetry(
            raw_frame,
            m_mask,
            line_pos,
            reward,
            action,
            fps,
            is_collision,
            self.stuck_frames,
            int(4.0 * fps),  # Updated to match Stuck termination [cite: 11-02-2026]
            self.missed_line_frames,
            int(1.5 * fps),  # Updated to match Line Lost termination [cite: 11-02-2026]
            int(2.0 * fps),  # Warning threshold for Stuck
            r_centering,
            r_progress,  # Mapping r_progress to r_gas for visual check [cite: 11-02-2026]
            r_collision,
            r_stuck_drain,
            displacement,
            getattr(self, "current_speed", 0),
            self.loiter_frames,
        )

        self.current_step += 1
        self.episode_reward += reward
        obs = np.expand_dims(self.vision.process_obs(raw_frame), axis=-1)

        if terminated:
            self._print_episode_stats(reason)

        return obs, reward, terminated, False, {}

    def _render_dynamic_telemetry(
        self,
        frame,
        m_mask,
        l_pos,
        rew,
        action,
        fps,
        crash,
        s_frames,
        s_max,  # Should be 5.0 * fps [cite: 11-02-2026]
        m_frames,
        m_max,  # Should be 1.0 * fps [cite: 11-02-2026]
        s_warn,
        r_cent,
        r_gas,
        r_coll,
        r_drain,
        displacement,
        current_speed,
        l_frames,
    ):
        try:
            db = np.zeros((550, 650, 3), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            c_white, c_gray = (220, 220, 220), (100, 100, 100)

            # --- ZONE 1: SENSORS (Map & Lap) ---
            db[30:130, 20:120] = cv2.cvtColor(
                cv2.resize(m_mask, (100, 100)), cv2.COLOR_GRAY2BGR
            )

            y_l, x_l, h_l, w_l = self.vision.LAP_ROI
            lap_roi = cv2.resize(frame[y_l : y_l + h_l, x_l : x_l + w_l], (100, 50))
            db[160:210, 20:120] = lap_roi

            # Speedometer ROI Preview [cite: 11-02-2026]
            y_s, x_s, h_s, w_s = self.vision.SPEED_ROI
            if h_s > 0 and w_s > 0:
                speed_roi = cv2.resize(
                    frame[y_s : y_s + h_s, x_s : x_s + w_s], (100, 50)
                )
                db[230:280, 20:120] = speed_roi
                cv2.putText(db, "SPEED OCR", (20, 225), font, 0.4, (255, 255, 0), 1)

            # --- ZONE 2: REWARD BREAKDOWN (INCENTIVE CHECK) --- [cite: 11-02-2026]
            rx = 150
            cv2.putText(db, "LIVE REWARDS", (rx, 25), font, 0.5, (0, 255, 255), 1)
            cv2.putText(db, f"Centering: {r_cent:.2f}", (rx, 50), font, 0.4, c_white, 1)
            cv2.putText(
                db, f"Gas Bonus: {r_gas:.2f}", (rx, 70), font, 0.4, (100, 255, 100), 1
            )
            cv2.putText(
                db, f"Collision: {r_coll:.2f}", (rx, 90), font, 0.4, (0, 0, 255), 1
            )
            cv2.putText(
                db, f"Stuck Drn: {r_drain:.2f}", (rx, 110), font, 0.4, (0, 165, 255), 1
            )

            # --- ZONE 3: DYNAMIC TIMERS (Matching Revival Thresholds) --- [cite: 11-02-2026]
            # Stuck Bar (5.0s Max)
            s_w = int((min(s_frames, s_max) / max(1, s_max)) * 200)
            cv2.rectangle(db, (rx, 140), (rx + 200, 150), (40, 40, 40), -1)
            cv2.rectangle(db, (rx, 140), (rx + s_w, 150), (0, 165, 255), -1)
            cv2.putText(db, "STUCK TIMER", (rx, 135), font, 0.35, (0, 165, 255), 1)

            # Line-Loss Bar (1.0s Max)
            m_w = int((min(m_frames, m_max) / max(1, m_max)) * 200)
            cv2.rectangle(db, (rx, 170), (rx + 200, 180), (40, 40, 40), -1)
            cv2.rectangle(db, (rx, 170), (rx + m_w, 180), (0, 0, 255), -1)
            cv2.putText(db, "LINE LOSS TIMER", (rx, 165), font, 0.35, (0, 0, 255), 1)

            # LOITER BAR (3.0s Max) [cite: 12-02-2026]
            l_limit = int(3.0 * fps)
            l_w = int((min(l_frames, l_limit) / max(1, l_limit)) * 200)
            cv2.rectangle(db, (rx, 200), (rx + 200, 210), (40, 40, 40), -1)
            cv2.rectangle(db, (rx, 200), (rx + l_w, 210), (0, 255, 255), -1)
            cv2.putText(
                db, "LOITER CRIME METER", (rx, 195), font, 0.35, (0, 255, 255), 1
            )

            # --- ZONE 4: INPUTS ---
            # Visualizing Steering (Fixed at 51-offset center) [cite: 11-02-2026]
            cv2.line(
                db,
                (rx + 100, 250),
                (rx + 100 + int(action[0] * 50), 250),
                (255, 255, 255),
                10,
            )
            cv2.putText(db, "STEERING", (rx + 75, 240), font, 0.4, c_white, 1)

            # --- ZONE 5: SYSTEM STATS ---
            sx = 400
            cv2.putText(db, f"FPS: {fps:.1f}", (sx, 50), font, 0.8, (255, 255, 0), 2)

            speed_val = current_speed if current_speed is not None else "???"
            cv2.putText(
                db, f"SPEED: {speed_val} KM/H", (sx, 90), font, 0.7, (255, 255, 255), 2
            )

            # Highlight Profitability (Green if positive!) [cite: 11-02-2026, 12-02-2026]
            r_col = (0, 255, 0) if rew >= 0 else (0, 0, 255)
            cv2.putText(db, f"REW: {rew:.4f}", (sx, 175), font, 1.0, r_col, 3)

            # RACING STATUS [cite: 12-02-2026]
            if l_frames > 5:
                status, st_col = "LOITERING", (0, 255, 255)
            elif s_frames > s_warn:
                status, st_col = "STUCK", (0, 165, 255)
            elif m_frames > 0:
                status, st_col = "SEARCHING", (0, 0, 255)
            else:
                status, st_col = "PROFITING" if rew > 0 else "RACING", (0, 255, 0)

            cv2.putText(db, status, (sx, 230), font, 1.0, st_col, 3)
            cv2.putText(
                db,
                f"DISPLACEMENT: {displacement:.2f}",
                (sx, 300),
                font,
                0.5,
                (255, 255, 255),
                1,
            )

            cv2.imshow("GT Dynamic Telemetry", db)
            cv2.waitKey(1)
        except Exception as e:
            print(f"⚠️ Dash Error: {e}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)
        self.controller.step(0.0, 0.0, reset=True)
        self.current_step = 0
        self.missed_line_frames = 0
        self.stuck_frames = 0
        self.episode_reward = 0.0  # Reset for new ep [cite: 11-02-2026]
        self.vision.reset_lap()

        time.sleep(1.5)
        raw_frame = self.vision.get_frame()
        if raw_frame is None:
            return np.zeros((84, 84, 1), dtype=np.uint8), {}

        obs = self.vision.process_obs(raw_frame)
        obs = np.expand_dims(obs, axis=-1)
        return obs, {}

    def close(self):
        self.controller.close()
