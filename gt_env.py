"""
GranTurismoEnv v3 — Hybrid
───────────────────────────
v1 observations (masks) + v2 reward/hyperparams + wrong-direction termination.
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

        # Dict observation: mask channels + proprioceptive aux
        self.observation_space = spaces.Dict(
            {
                "frame": spaces.Box(low=0, high=255, shape=(64, 64, 3), dtype=np.uint8),
                "aux": spaces.Box(low=-1.0, high=1.0, shape=(8,), dtype=np.float32),
            }
        )

        # Episode state
        self.max_steps = 5000  # ~3min at 27 FPS (lap takes ~2min)
        self.current_step = 0
        self.stuck_frames = 0
        self.loiter_frames = 0
        self.wrong_dir_frames = 0
        self.episode_reward = 0.0
        self.current_speed = 0
        self.prev_progress = 0.0
        self.estimated_speed = 0.0
        self.progress_at_gate = None  # Snapshot for progress gate check

        # Buffers
        self.pos_buffer = deque(maxlen=100)
        self.progress_buffer = deque(maxlen=5)
        self.progress_reward_buffer = deque(maxlen=10)  # Smooth progress over 10 frames

        # Action history (Paper 1: 3-step steering + gas history)
        self.steer_history = deque([0.0, 0.0, 0.0], maxlen=3)
        self.gas_history = deque([0.0, 0.0, 0.0], maxlen=3)

        # Temporal
        self.last_step_time = time.time()

        # Curriculum
        self.curriculum_stage = 1
        self.lap_completions = 0
        self.episode_count = 0

        # Dashboard
        self.render_interval = 5

        # Termination thresholds (seconds)
        self.GRACE_PERIOD = 10.0
        self.STUCK_TOLERANCE = 12.0    # Give agent time to recover
        self.LOITER_TOLERANCE = 15.0   # Only kill true standstills
        self.WRONG_DIR_TOLERANCE = 5.0
        self.PROGRESS_GATE_TIME = 15.0  # Was 30s — agent was dying net-positive before gate fired
        self.PROGRESS_GATE_MIN = 0.03   # Minimum progress delta required

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
        self.steer_history.append(steer)
        self.gas_history.append(gas)

        # 2. OBSERVE — single frame
        raw_frame = self.vision.get_frame()
        if raw_frame is None:
            return self._get_empty_obs(), 0.0, True, False, {"reason": "CAMERA_LOST"}

        # 3. VISION — each channel computed once
        line_channel, line_pos = self.vision.get_line_channel(raw_frame)
        road_channel = self.vision.get_road_channel(raw_frame)
        brake_channel = self.vision.get_brake_channel(raw_frame)
        is_collision = self.vision.check_collision(raw_frame)

        # 4. PROGRESS (smoothed)
        progress_data = self.vision.get_progress_percent(raw_frame)
        raw_progress = (
            progress_data[0]
            if progress_data and progress_data[0] is not None
            else self.prev_progress
        )
        self.progress_buffer.append(raw_progress)
        current_progress = float(np.median(list(self.progress_buffer)))

        # 5. DISPLACEMENT
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

        if len(self.pos_buffer) >= 2:
            instant_disp = np.linalg.norm(
                np.array(self.pos_buffer[-1]) - np.array(self.pos_buffer[-2])
            )
        else:
            instant_disp = 0.0
        self.estimated_speed = instant_disp / dt

        # ═══════════════════════════════════
        # 6. REWARD — progress-dominated
        # ═══════════════════════════════════

        # A. Progress — primary signal (smoothed, gated on movement)
        progress_delta = current_progress - self.prev_progress
        if progress_delta < -0.5:  # Lap wraparound
            progress_delta = (1.0 - self.prev_progress) + current_progress
        progress_delta = max(progress_delta, -0.1)
        self.progress_reward_buffer.append(progress_delta * 1500.0)

        # GATE: only reward progress if car is actually moving
        # Threshold 5.0 filters minimap jitter (±1-2px → endpoint drift up to ~3px)
        if displacement > 5.0:
            r_progress = float(np.mean(self.progress_reward_buffer))
        else:
            r_progress = -0.05  # Active penalty: standing still is worse than zero
            self.progress_reward_buffer.clear()  # Flush so old positives can't leak
        self.prev_progress = current_progress

        # B. Speed — directly reward locomotion (independent of minimap progress)
        # Jitter floor at 1.5 filters stationary noise, caps at 0.5/step
        r_speed = max(0.0, (self.estimated_speed - 1.5)) * 0.1
        r_speed = min(r_speed, 0.5)

        # C. Steering change penalty (Paper 1: r_s = -|Δθ|)
        steer_delta = abs(steer - self.steer_history[-2])
        r_steer = -steer_delta * 0.3

        reward = r_progress + r_speed + r_steer

        # 7. TERMINATIONS
        terminated = False
        reason = ""
        in_grace = self.current_step < int(self.GRACE_PERIOD * fps)

        # OCR speed + lap check (every 30 steps — OCR only for lap detection + display)
        if self.current_step % 30 == 0:
            self.current_speed = self.vision.get_speed(raw_frame)
            if self.vision.check_lap_change(
                raw_frame, self.current_step, fps, self.current_speed
            ):
                reward += 1000.0
                terminated = True
                reason = "LAP COMPLETED"
                self._update_curriculum(lap_completed=True)

        # Loiter check — displacement-based
        if not in_grace:
            if self.estimated_speed < 0.8:  # Only true standstill (rolling start)
                self.loiter_frames += 1
            else:
                self.loiter_frames = 0
        else:
            self.loiter_frames = 0

        # Progress gate — snapshot progress at end of grace, check later
        gate_step = int((self.GRACE_PERIOD + self.PROGRESS_GATE_TIME) * fps)
        grace_end_step = int(self.GRACE_PERIOD * fps)
        if self.current_step == grace_end_step:
            self.progress_at_gate = current_progress

        if not terminated:
            # Stuck: low displacement
            if displacement < 1.0 and not in_grace:
                self.stuck_frames += 1
                if self.stuck_frames > (self.STUCK_TOLERANCE * fps):
                    reward += -500.0
                    terminated = True
                    reason = "STUCK"
            else:
                self.stuck_frames = 0

            # Loiter: OCR says not moving
            if self.loiter_frames > (self.LOITER_TOLERANCE * fps):
                reward += -500.0
                terminated = True
                reason = "LOITERING"

            # Wrong direction: negative progress for too long
            if progress_delta < -0.001 and not in_grace:
                self.wrong_dir_frames += 1
                if self.wrong_dir_frames > (self.WRONG_DIR_TOLERANCE * fps):
                    reward += -300.0
                    terminated = True
                    reason = "WRONG DIRECTION"
            else:
                self.wrong_dir_frames = 0

            # Progress gate: must have made 3% progress by grace + 30s
            if (self.current_step == gate_step
                    and self.progress_at_gate is not None):
                progress_made = current_progress - self.progress_at_gate
                if progress_made < -0.5:  # Wraparound
                    progress_made += 1.0
                if progress_made < self.PROGRESS_GATE_MIN:
                    reward += -500.0
                    terminated = True
                    reason = f"NO PROGRESS ({progress_made:.3f} < {self.PROGRESS_GATE_MIN})"

            # Max steps
            if self.current_step >= self.max_steps:
                terminated = True
                reason = "MAX_STEPS"

        # 8. BUILD OBS
        frame_obs = np.stack([line_channel, road_channel, brake_channel], axis=-1)
        aux = self._build_aux()
        obs = {"frame": frame_obs, "aux": aux}

        # 9. RENDER (throttled)
        if self.current_step % self.render_interval == 0:
            self._render_dashboard(
                obs_frame=frame_obs,
                l_pos=line_pos,
                rew=reward,
                action=action,
                fps=fps,
                crash=is_collision,
                r_prog=r_progress,
                r_speed=r_speed,
                r_steer=r_steer,
                displacement=displacement,
                s_frames=self.stuck_frames,
                s_max=int(self.STUCK_TOLERANCE * fps),
                l_frames=self.loiter_frames,
                l_max=int(self.LOITER_TOLERANCE * fps),
                w_frames=self.wrong_dir_frames,
                w_max=int(self.WRONG_DIR_TOLERANCE * fps),
                has_line=(line_pos is not None),
            )

        self.episode_reward += reward
        if terminated:
            self._print_episode_stats(reason)
            self._check_curriculum_advance()

        return obs, float(reward), terminated, False, {"reason": reason}

    def _build_aux(self):
        """
        8-dim proprioceptive vector:
          [speed, progress, steer×3, gas×3]
        """
        speed_norm = min(1.0, self.estimated_speed / 50.0)
        return np.array([
            speed_norm,
            self.prev_progress,
            self.steer_history[-1],
            self.steer_history[-2],
            self.steer_history[-3],
            self.gas_history[-1],
            self.gas_history[-2],
            self.gas_history[-3],
        ], dtype=np.float32)

    def _render_dashboard(
        self, obs_frame, l_pos, rew, action, fps, crash,
        r_prog, r_speed, r_steer, displacement,
        s_frames, s_max, l_frames, l_max, w_frames, w_max, has_line,
    ):
        try:
            W, H = 480, 380
            db = np.zeros((H, W, 3), dtype=np.uint8)
            f = cv2.FONT_HERSHEY_SIMPLEX
            WHITE = (220, 220, 220)
            DIM = (100, 100, 100)
            CYAN = (0, 255, 255)

            # ── ROW 1: VISION CHANNELS ──
            cv2.putText(db, "AGENT VISION", (10, 14), f, 0.35, CYAN, 1)
            labels = ["LINE", "ROAD", "BRAKE", "STACK"]
            colors = [(255, 200, 0), (0, 255, 0), (0, 0, 255), WHITE]
            for i in range(3):
                ch = cv2.cvtColor(obs_frame[:, :, i], cv2.COLOR_GRAY2BGR)
                ch = cv2.resize(ch, (64, 64))
                x0 = 10 + i * 74
                db[20:84, x0:x0+64] = ch
                cv2.putText(db, labels[i], (x0 + 15, 96), f, 0.28, colors[i], 1)
            # Composite
            rgb = cv2.resize(obs_frame, (64, 64))
            db[20:84, 232:296] = rgb
            cv2.putText(db, "STACK", (240, 96), f, 0.28, WHITE, 1)

            # ── STATS ──
            sx = 320
            fps_col = (0, 255, 0) if fps > 10 else (0, 165, 255) if fps > 6 else (0, 0, 255)
            cv2.putText(db, f"FPS: {fps:.0f}", (sx, 28), f, 0.45, fps_col, 2)

            ocr_spd = self.current_speed if self.current_speed else 0
            cv2.putText(db, f"SPD: {ocr_spd} km/h", (sx, 48), f, 0.33, WHITE, 1)
            cv2.putText(db, f"STEP: {self.current_step}", (sx, 64), f, 0.3, DIM, 1)
            cv2.putText(db, f"DISP: {displacement:.1f}", (sx, 80), f, 0.3, DIM, 1)

            r_col = (0, 255, 0) if rew >= 0 else (0, 0, 255)
            cv2.putText(db, f"{rew:+.1f}", (sx, 110), f, 0.7, r_col, 2)
            cv2.putText(db, f"EP: {self.episode_reward:+.0f}", (sx, 128), f, 0.3, DIM, 1)

            if crash:
                status, st_col = "CRASH", (0, 0, 255)
            elif not has_line:
                status, st_col = "NO LINE", (0, 100, 255)
            elif rew > 0.1:
                status, st_col = "RACING", (0, 255, 0)
            else:
                status, st_col = "DRIVING", (200, 200, 200)
            cv2.putText(db, status, (sx, 155), f, 0.5, st_col, 2)

            # ── REWARDS ──
            ry = 110
            cv2.putText(db, "REWARDS", (10, ry), f, 0.33, CYAN, 1)
            def _rc(v):
                return (0, 255, 0) if v > 0.01 else (0, 0, 255) if v < -0.01 else DIM
            for i, (name, val) in enumerate([("Progress", r_prog), ("Speed", r_speed), ("Steer Δ", r_steer)]):
                y = ry + 15 + i * 16
                cv2.putText(db, f"{name}:", (10, y), f, 0.28, DIM, 1)
                cv2.putText(db, f"{val:+.3f}", (85, y), f, 0.28, _rc(val), 1)
            tot_y = ry + 15 + 3 * 16
            cv2.line(db, (10, tot_y - 3), (160, tot_y - 3), (50, 50, 50), 1)
            cv2.putText(db, f"TOTAL: {rew:+.3f}", (10, tot_y + 10), f, 0.33, r_col, 1)

            # ── TIMERS ──
            ty = tot_y + 22
            cv2.putText(db, "TIMERS", (10, ty), f, 0.33, CYAN, 1)
            bar_w = 140
            for i, (name, frames, limit, color) in enumerate([
                ("STUCK", s_frames, s_max, (0, 165, 255)),
                ("LOITER", l_frames, l_max, (0, 255, 255)),
                ("WRONG", w_frames, w_max, (0, 0, 255)),
            ]):
                y = ty + 14 + i * 18
                cv2.putText(db, name, (10, y + 3), f, 0.26, DIM, 1)
                cv2.rectangle(db, (60, y - 3), (60 + bar_w, y + 6), (30, 30, 30), -1)
                fill = int((min(frames, limit) / max(1, limit)) * bar_w)
                cv2.rectangle(db, (60, y - 3), (60 + fill, y + 6), color, -1)

            # ── STEERING ──
            sy = ty + 14 + 3 * 18 + 8
            cv2.putText(db, "STEER", (10, sy), f, 0.26, DIM, 1)
            cx = 130
            cv2.line(db, (60, sy - 3), (200, sy - 3), (40, 40, 40), 5)
            cv2.line(db, (cx, sy - 7), (cx, sy + 1), DIM, 1)
            steer_x = cx + int(action[0] * 70)
            cv2.line(db, (cx, sy - 3), (steer_x, sy - 3), WHITE, 5)

            cv2.putText(db, f"STAGE {self.curriculum_stage}", (sx, H - 8), f, 0.28, (200, 200, 100), 1)

            cv2.imshow("GT Telemetry v3", db)
            cv2.waitKey(1)
        except Exception as e:
            print(f"⚠️ Dash: {e}")

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed, options=options)

        allowed_slots = self._get_curriculum_slots()
        target_slot = np.random.choice(allowed_slots)
        self.controller.load_save_state(target_slot)
        time.sleep(1.5)

        self.current_step = 0
        self.stuck_frames = 0
        self.loiter_frames = 0
        self.wrong_dir_frames = 0
        self.episode_reward = 0.0
        self.pos_buffer.clear()
        self.current_speed = 0
        self.prev_progress = 0.0
        self.estimated_speed = 0.0
        self.progress_at_gate = None
        self.last_step_time = time.time()
        self.progress_buffer.clear()
        self.progress_reward_buffer.clear()
        self.steer_history = deque([0.0, 0.0, 0.0], maxlen=3)
        self.gas_history = deque([0.0, 0.0, 0.0], maxlen=3)

        self.vision.reset_lap()

        raw_frame = self.vision.get_frame()
        if raw_frame is None:
            return self._get_empty_obs(), {}

        line_ch, _ = self.vision.get_line_channel(raw_frame)
        road_ch = self.vision.get_road_channel(raw_frame)
        brake_ch = self.vision.get_brake_channel(raw_frame)
        frame_obs = np.stack([line_ch, road_ch, brake_ch], axis=-1)
        aux = self._build_aux()
        obs = {"frame": frame_obs, "aux": aux}

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
