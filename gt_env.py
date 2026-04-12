"""
GranTurismoEnv v2 — Paper-Aligned
──────────────────────────────────
- Raw 64×64 RGB observation (CNN learns features)
- Progress-dominated reward (papers' approach)
- Steering/gas history in aux vector (proprioception)
- Simplified terminations
"""

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

        # Observation: 64×64 RGB frame + proprioceptive aux vector
        self.observation_space = spaces.Dict(
            {
                "frame": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                "aux": spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
            }
        )

        # Episode state
        self.max_steps = 15000
        self.current_step = 0
        self.stuck_frames = 0
        self.loiter_frames = 0
        self.episode_reward = 0.0
        self.current_speed = 0
        self.prev_progress = 0.0
        self.estimated_speed = 0.0

        # Position tracking
        self.pos_buffer = deque(maxlen=100)
        self.progress_buffer = deque(maxlen=5)

        # Action history (for proprioceptive aux — Paper 1 uses 3-step history)
        self.steer_history = deque([0.0, 0.0, 0.0], maxlen=3)
        self.gas_history = deque([0.0, 0.0, 0.0], maxlen=3)

        # Temporal
        self.last_step_time = time.time()

        # Curriculum tracking
        self.curriculum_stage = 1
        self.lap_completions = 0
        self.episode_count = 0

        # Dashboard
        self.render_interval = 5

        # Termination thresholds (seconds)
        self.GRACE_PERIOD = 10.0
        self.STUCK_TOLERANCE = 10.0
        self.LOITER_TOLERANCE = 15.0

    def _print_episode_stats(self, reason):
        print("\n" + "=" * 45)
        print(f"🏁  EPISODE SUMMARY | {reason}")
        print("-" * 45)
        print(f"   Steps:       {self.current_step}")
        print(f"   Total Rew:   {self.episode_reward:.2f}")
        print(f"   Avg/Step:    {(self.episode_reward / max(1, self.current_step)):.4f}")
        print(f"   Laps Done:   {self.vision.last_detected_lap - 1}")
        print("=" * 45 + "\n")

    def step(self, action):
        now = time.time()
        dt = now - self.last_step_time
        self.last_step_time = now
        dt = np.clip(dt, 1.0 / 60.0, 1.0)
        fps = 1.0 / dt

        self.current_step += 1

        # 1. ACT
        steer, gas = float(action[0]), float(action[1])
        self.controller.step(steering=steer, gas_brake=gas)

        # Record action history AFTER acting
        self.steer_history.append(steer)
        self.gas_history.append(gas)

        # 2. OBSERVE — single frame
        raw_frame = self.vision.get_frame()
        if raw_frame is None:
            return self._get_empty_obs(), 0.0, True, False, {"reason": "CAMERA_LOST"}

        # 3. FRAME FOR CNN — 64×64 RGB crop
        obs_frame = self.vision.get_obs_frame(raw_frame)

        # 4. TELEMETRY
        is_collision = self.vision.check_collision(raw_frame)

        # Progress (smoothed)
        progress_data = self.vision.get_progress_percent(raw_frame)
        raw_progress = (
            progress_data[0]
            if progress_data and progress_data[0] is not None
            else self.prev_progress
        )
        self.progress_buffer.append(raw_progress)
        current_progress = float(np.median(list(self.progress_buffer)))

        # Displacement (for stuck detection)
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

        # Estimated speed (for aux vector)
        if len(self.pos_buffer) >= 2:
            instant_disp = np.linalg.norm(
                np.array(self.pos_buffer[-1]) - np.array(self.pos_buffer[-2])
            )
        else:
            instant_disp = 0.0
        self.estimated_speed = instant_disp / dt

        # ═══════════════════════════════════════════
        # 5. REWARD — Progress-dominated (Paper-aligned)
        # ═══════════════════════════════════════════

        # A. Progress — THE primary signal
        progress_delta = current_progress - self.prev_progress
        if progress_delta < -0.5:  # Lap wraparound
            progress_delta = (1.0 - self.prev_progress) + current_progress
        progress_delta = max(progress_delta, -0.05)  # Clamp backward
        r_progress = progress_delta * 500.0
        self.prev_progress = current_progress

        # B. Collision penalty — proportional to speed² (Paper 2)
        r_collision = 0.0
        if is_collision:
            spd = self.estimated_speed
            r_collision = -max(10.0, spd * spd * 0.01)

        # C. Steering change penalty (Paper 1: r_s = -|θ_t - θ_{t-1}|)
        steer_delta = abs(steer - self.steer_history[-2])  # -2 because -1 is current
        r_steer = -steer_delta * 1.0

        reward = r_progress + r_collision + r_steer

        # 6. TERMINATIONS
        terminated = False
        reason = ""

        # OCR speed (every 30 steps)
        if self.current_step % 30 == 0:
            self.current_speed = self.vision.get_speed(raw_frame)
            if self.vision.check_lap_change(
                raw_frame, self.current_step, fps, self.current_speed
            ):
                reward += 1000.0
                terminated = True
                reason = "LAP COMPLETED"
                self._update_curriculum(lap_completed=True)

            # Loiter check (OCR speed, after grace)
            ocr_speed = self.current_speed if self.current_speed is not None else 0
            if self.current_step > int(self.GRACE_PERIOD * fps):
                if ocr_speed < 5:
                    self.loiter_frames += 30
                else:
                    self.loiter_frames = 0
            else:
                self.loiter_frames = 0

        if not terminated:
            # Stuck: low displacement over long window
            if displacement < 1.0:
                self.stuck_frames += 1
                if self.stuck_frames > (self.STUCK_TOLERANCE * fps):
                    reward += -500.0
                    terminated = True
                    reason = "STUCK POSITION"
            else:
                self.stuck_frames = 0

            # Loiter: OCR says car isn't moving
            if self.loiter_frames > (self.LOITER_TOLERANCE * fps):
                reward += -500.0
                terminated = True
                reason = "LOITERING"

            # Max steps
            if self.current_step >= self.max_steps:
                terminated = True
                reason = "MAX_STEPS"

        # 7. BUILD OBS
        aux = self._build_aux()
        obs = {"frame": obs_frame, "aux": aux}

        # 8. RENDER
        if self.current_step % self.render_interval == 0:
            self._render_dashboard(
                obs_frame=obs_frame,
                rew=reward,
                action=action,
                fps=fps,
                crash=is_collision,
                r_prog=r_progress,
                r_coll=r_collision,
                r_steer=r_steer,
                displacement=displacement,
                s_frames=self.stuck_frames,
                s_max=int(self.STUCK_TOLERANCE * fps),
                l_frames=self.loiter_frames,
                l_max=int(self.LOITER_TOLERANCE * fps),
            )

        self.episode_reward += reward

        if terminated:
            self._print_episode_stats(reason)
            self._check_curriculum_advance()

        return obs, float(reward), terminated, False, {"reason": reason}

    def _build_aux(self):
        """
        8-dim proprioceptive vector (Paper 1 inspired):
          [speed_norm, progress,
           steer_t-1, steer_t-2, steer_t-3,
           gas_t-1, gas_t-2, gas_t-3]
        """
        speed_norm = min(1.0, self.estimated_speed / 50.0)
        progress = self.prev_progress

        return np.array([
            speed_norm,
            progress,
            self.steer_history[-1],  # most recent
            self.steer_history[-2],
            self.steer_history[-3],
            self.gas_history[-1],
            self.gas_history[-2],
            self.gas_history[-3],
        ], dtype=np.float32)

    def _render_dashboard(
        self, obs_frame, rew, action, fps, crash,
        r_prog, r_coll, r_steer,
        displacement, s_frames, s_max, l_frames, l_max,
    ):
        try:
            W, H = 420, 340
            db = np.zeros((H, W, 3), dtype=np.uint8)
            f = cv2.FONT_HERSHEY_SIMPLEX
            WHITE = (220, 220, 220)
            DIM = (100, 100, 100)
            CYAN = (0, 255, 255)

            # ── AGENT VISION (what CNN sees) ──
            cv2.putText(db, "CNN INPUT (64x64 RGB)", (10, 14), f, 0.35, CYAN, 1)
            preview = cv2.resize(obs_frame, (120, 120))
            db[20:140, 10:130] = preview

            # ── STATS ──
            sx = 150
            fps_col = (0, 255, 0) if fps > 10 else (0, 165, 255) if fps > 6 else (0, 0, 255)
            cv2.putText(db, f"FPS: {fps:.0f}", (sx, 30), f, 0.5, fps_col, 2)

            ocr_spd = self.current_speed if self.current_speed else 0
            cv2.putText(db, f"SPD: {ocr_spd} km/h", (sx, 52), f, 0.35, WHITE, 1)
            cv2.putText(db, f"STEP: {self.current_step}", (sx, 70), f, 0.33, DIM, 1)
            cv2.putText(db, f"DISP: {displacement:.1f}", (sx, 88), f, 0.33, DIM, 1)

            # Reward
            r_col = (0, 255, 0) if rew >= 0 else (0, 0, 255)
            cv2.putText(db, f"REW: {rew:+.1f}", (sx, 120), f, 0.6, r_col, 2)
            cv2.putText(db, f"EP: {self.episode_reward:+.0f}", (sx, 140), f, 0.33, DIM, 1)

            # Status
            if crash:
                status, st_col = "CRASH", (0, 0, 255)
            elif rew > 0.1:
                status, st_col = "PROGRESSING", (0, 255, 0)
            else:
                status, st_col = "EXPLORING", (200, 200, 200)
            cv2.putText(db, status, (sx + 120, 30), f, 0.5, st_col, 2)

            # ── REWARDS ──
            ry = 160
            cv2.putText(db, "REWARDS", (10, ry), f, 0.35, CYAN, 1)
            def _rc(v):
                return (0, 255, 0) if v > 0.01 else (0, 0, 255) if v < -0.01 else DIM
            for i, (name, val) in enumerate([("Progress", r_prog), ("Collision", r_coll), ("Steer", r_steer)]):
                y = ry + 15 + i * 16
                cv2.putText(db, f"{name}:", (10, y), f, 0.3, DIM, 1)
                cv2.putText(db, f"{val:+.2f}", (85, y), f, 0.3, _rc(val), 1)

            tot_y = ry + 15 + 3 * 16
            cv2.line(db, (10, tot_y - 3), (150, tot_y - 3), (50, 50, 50), 1)
            cv2.putText(db, f"TOTAL: {rew:+.2f}", (10, tot_y + 10), f, 0.35, r_col, 1)

            # ── TIMERS ──
            ty = tot_y + 25
            cv2.putText(db, "TIMERS", (10, ty), f, 0.35, CYAN, 1)
            bar_w = 150
            for i, (name, frames, limit, color) in enumerate([
                ("STUCK", s_frames, s_max, (0, 165, 255)),
                ("LOITER", l_frames, l_max, (0, 255, 255)),
            ]):
                y = ty + 15 + i * 20
                cv2.putText(db, name, (10, y + 3), f, 0.28, DIM, 1)
                cv2.rectangle(db, (60, y - 4), (60 + bar_w, y + 6), (30, 30, 30), -1)
                fill = int((min(frames, limit) / max(1, limit)) * bar_w)
                cv2.rectangle(db, (60, y - 4), (60 + fill, y + 6), color, -1)

            # ── STEERING ──
            sy = ty + 15 + 2 * 20 + 10
            cv2.putText(db, "STEER", (10, sy), f, 0.28, DIM, 1)
            cx = 140
            cv2.line(db, (60, sy - 3), (220, sy - 3), (40, 40, 40), 5)
            cv2.line(db, (cx, sy - 7), (cx, sy + 1), DIM, 1)
            steer_x = cx + int(action[0] * 80)
            cv2.line(db, (cx, sy - 3), (steer_x, sy - 3), WHITE, 5)

            cv2.putText(db, f"STAGE {self.curriculum_stage}", (sx + 120, H - 10), f, 0.3, (200, 200, 100), 1)

            cv2.imshow("GT Dynamic Telemetry", db)
            cv2.waitKey(1)
        except Exception as e:
            print(f"⚠️ Dash Error: {e}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        allowed_slots = self._get_curriculum_slots()
        target_slot = np.random.choice(allowed_slots)
        self.controller.load_save_state(target_slot)
        time.sleep(1.5)

        # Reset state
        self.current_step = 0
        self.stuck_frames = 0
        self.loiter_frames = 0
        self.episode_reward = 0.0
        self.pos_buffer.clear()
        self.current_speed = 0
        self.prev_progress = 0.0
        self.estimated_speed = 0.0
        self.last_step_time = time.time()
        self.progress_buffer.clear()
        self.steer_history = deque([0.0, 0.0, 0.0], maxlen=3)
        self.gas_history = deque([0.0, 0.0, 0.0], maxlen=3)

        self.vision.reset_lap()

        raw_frame = self.vision.get_frame()
        if raw_frame is None:
            return self._get_empty_obs(), {}

        obs_frame = self.vision.get_obs_frame(raw_frame)
        aux = self._build_aux()
        obs = {"frame": obs_frame, "aux": aux}

        print(f"🔄 Episode Started - Slot: {target_slot}, Stage: {self.curriculum_stage}")
        return obs, {}

    def _get_curriculum_slots(self):
        if self.curriculum_stage == 1:
            return [0]
        elif self.curriculum_stage == 2:
            return [0, 1]
        return [0, 1, 2]

    def _get_empty_obs(self):
        return {
            "frame": np.zeros((64, 64, 3), dtype=np.uint8),
            "aux": np.zeros((8,), dtype=np.float32),
        }

    def _update_curriculum(self, lap_completed):
        if lap_completed:
            self.lap_completions += 1

    def _check_curriculum_advance(self):
        self.episode_count += 1
        if self.episode_count >= 50:
            rate = self.lap_completions / self.episode_count
            if rate >= 0.2 and self.curriculum_stage < 3:
                self.curriculum_stage += 1
                print(f"📚 Curriculum advanced to Stage {self.curriculum_stage}")
                self.lap_completions = 0
                self.episode_count = 0

    def close(self):
        self.vision.cap.release()
        cv2.destroyAllWindows()
        self.controller.close()
