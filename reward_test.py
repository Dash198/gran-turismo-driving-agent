"""
Reward Test — Drive manually, watch the reward dashboard.
Press Q to quit.
"""
import cv2
import numpy as np
import time
from collections import deque
from vision import VisionInterface

vision = VisionInterface()

NUM_WP = 50
max_wp = -1
start_wp = 0
prev_progress = 0.0
pos_buffer = deque(maxlen=30)
prog_buffer = deque(maxlen=5)
total_reward = 0.0
step = 0
wp_count = 0
# Tracking exactly like training
stuck_frames = 0
loiter_frames = 0
wrong_dir_frames = 0
action = [0.0, 0.0]  # Dummy steering
fps = 15.0
spd = None

print("Drive manually — watching rewards. Press Q to quit.")

while True:
    raw = vision.get_frame()
    if raw is None:
        time.sleep(0.1)
        continue
    step += 1

    # Progress
    pd = vision.get_progress_percent(raw)
    rp = pd[0] if pd and pd[0] is not None else prev_progress
    prog_buffer.append(rp)
    cp = float(np.median(list(prog_buffer)))

    # Displacement
    pos = vision.get_map_position(raw)
    if pos:
        pos_buffer.append(pos)
    elif pos_buffer:
        pos_buffer.append(pos_buffer[-1])
    disp = 0.0
    if len(pos_buffer) >= 2:
        disp = np.linalg.norm(np.array(pos_buffer[-1]) - np.array(pos_buffer[0]))

    # Waypoint
    cwp = int(cp * NUM_WP) % NUM_WP
    r_prog = 0.0
    if max_wp == -1:
        max_wp = cwp
        start_wp = cwp
        print(f"  INIT: start_wp={start_wp}, progress={cp*100:.1f}%")
    else:
        fd = (cwp - max_wp) % NUM_WP
        if 0 < fd <= NUM_WP // 2:  # Forward up to half-track
            capped = min(fd, 5)
            r_prog = capped * 30.0
            wp_count += fd
            max_wp = cwp
            print(f"  ✅ WP +{fd} (capped {capped})! wp={cwp}, crossed={wp_count}, reward=+{r_prog:.0f}")
    prev_progress = cp

# Match gt_env tracking
    r_time = -0.1
    r_steer = 0.0
    rew = r_prog + r_time + r_steer
    total_reward += rew

    in_grace = step < int(10 * fps)
    
    if not in_grace:
        if disp < 1.0:
            stuck_frames += 1
            loiter_frames += 1
        else:
            stuck_frames = 0
            loiter_frames = 0
    else:
        stuck_frames = 0
        loiter_frames = 0

    prog_delta = cp - prev_progress
    if prog_delta < -0.5:
        prog_delta = (1.0 - prev_progress) + cp
    
    if prog_delta < -0.001 and not in_grace:
        wrong_dir_frames += 1
    else:
        wrong_dir_frames = 0

    crash = vision.check_collision(raw)

    # Print every 50 steps
    if step % 50 == 0:
        fd_now = (cwp - max_wp) % NUM_WP
        print(f"  step={step} progress={cp*100:.1f}% wp={cwp} max_wp={max_wp} fwd_dist={fd_now} total={total_reward:+.0f}")

    # Channels
    line_ch, _ = vision.get_line_channel(raw)
    road_ch = vision.get_road_channel(raw)
    brake_ch = vision.get_brake_channel(raw)
    if step % 30 == 0:
        spd = vision.get_speed(raw)

    # Dashboard (Identical to gt_env)
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
        
        obs_frame = np.stack([line_ch, road_ch, brake_ch], axis=-1)
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
        fps_col = (0, 255, 0)
        cv2.putText(db, f"FPS: {fps:.0f}", (sx, 28), f, 0.45, fps_col, 2)

        ocr_spd = spd if spd else 0
        cv2.putText(db, f"SPD: {ocr_spd} km/h", (sx, 48), f, 0.33, WHITE, 1)
        cv2.putText(db, f"STEP: {step}", (sx, 64), f, 0.3, DIM, 1)
        wp_crossed = (max_wp - start_wp) % NUM_WP if max_wp >= 0 else 0
        cv2.putText(db, f"WP: {wp_crossed}/{NUM_WP}  {prev_progress*100:.0f}%", (sx, 80), f, 0.28, (0, 255, 255), 1)

        r_col = (0, 255, 0) if rew >= 0 else (0, 0, 255)
        cv2.putText(db, f"{rew:+.1f}", (sx, 110), f, 0.7, r_col, 2)
        cv2.putText(db, f"EP: {total_reward:+.0f}", (sx, 128), f, 0.3, DIM, 1)

        has_line = (pd and pd[0] is not None)  # approximate
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
        for i, (name, val) in enumerate([("Progress", r_prog), ("Time", r_time), ("Steer", r_steer)]):
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
            ("STUCK", stuck_frames, int(10*fps), (0, 165, 255)),
            ("LOITER", loiter_frames, int(15*fps), (0, 255, 255)),
            ("WRONG", wrong_dir_frames, int(5*fps), (0, 0, 255)),
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

        cv2.imshow("GT Telemetry manual test", db)
    except Exception as e:
        print(f"⚠️ Dash: {e}")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f"\nDone! Steps: {step}, WP: {wp_count}, Total: {total_reward:+.0f}")
