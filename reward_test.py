"""
Reward Test — Drive manually, verify reward system + lap detection.
Press Q to quit.
"""
import cv2
import numpy as np
import time
from collections import deque
from vision import VisionInterface

vision = VisionInterface()

# ── State ──
NUM_WP        = 50
FPS_BASELINE  = 20
GRACE_STEPS   = 10 * FPS_BASELINE   # 200
STUCK_STEPS   = 12 * FPS_BASELINE   # 240
LOITER_STEPS  = 15 * FPS_BASELINE   # 300
WRONG_STEPS   = 5  * FPS_BASELINE   # 100
STAGNATION    = 60 * FPS_BASELINE   # 1200

max_wp        = -1
start_wp      = 0
prev_progress = 0.0
last_wp_step  = 0
pos_buffer    = deque(maxlen=30)
prog_buffer   = deque(maxlen=5)
total_reward  = 0.0
step          = 0
wp_count      = 0
stuck_frames  = 0
loiter_frames = 0
wrong_dir_frames = 0
spd           = None
lap_flash     = 0       # Flash counter for lap event highlight
laps_detected = 0
last_ocr_lap  = "?"
ocr_log       = deque(maxlen=6)  # Rolling OCR read log

# Lap history mirrors vision.py's new format
vision.lap_read_history = []

print("Drive manually — watching rewards + lap detection. Press Q to quit.")

while True:
    t0 = time.time()
    raw = vision.get_frame()
    if raw is None:
        time.sleep(0.1)
        continue
    step += 1

    # ── Progress ──
    pd = vision.get_progress_percent(raw)
    rp = pd[0] if pd and pd[0] is not None else prev_progress
    prog_buffer.append(rp)
    cp = float(np.median(list(prog_buffer)))

    # ── Displacement ──
    pos = vision.get_map_position(raw)
    if pos:
        pos_buffer.append(pos)
    elif pos_buffer:
        pos_buffer.append(pos_buffer[-1])
    disp = 0.0
    if len(pos_buffer) >= 2:
        disp = np.linalg.norm(np.array(pos_buffer[-1]) - np.array(pos_buffer[0]))

    # ── Waypoints ──
    cwp = int(cp * NUM_WP) % NUM_WP
    r_prog = 0.0
    if max_wp == -1:
        max_wp   = cwp
        start_wp = cwp
        last_wp_step = step
        print(f"  INIT: start_wp={start_wp}, progress={cp*100:.1f}%")
    else:
        fd = (cwp - max_wp) % NUM_WP
        if 0 < fd <= NUM_WP // 2:
            if fd <= 2:
                r_prog = 30.0
                wp_count += 1
                last_wp_step = step
                print(f"  ✅ WP +1! wp={cwp}, crossed={wp_count}, reward=+30")
            max_wp = (max_wp + 1) % NUM_WP
    prev_progress = cp

    # ── Rewards ──
    r_time  = -0.1
    r_steer = 0.0
    rew     = r_prog + r_time + r_steer
    total_reward += rew

    # ── Termination trackers (mirrors gt_env exactly) ──
    in_grace = step < GRACE_STEPS
    if not in_grace:
        if disp < 1.0:
            stuck_frames  += 1
            loiter_frames += 1
        else:
            stuck_frames  = 0
            loiter_frames = 0
    else:
        stuck_frames = loiter_frames = 0

    prog_delta = cp - prev_progress
    if prog_delta < -0.5:
        prog_delta = (1.0 - prev_progress) + cp
    if prog_delta < -0.001 and not in_grace:
        wrong_dir_frames += 1
    else:
        wrong_dir_frames = 0

    crash = vision.check_collision(raw)

    # ── LAP DETECTION (every 30 steps, same as training) ──
    lap_event = False
    if step % 30 == 0:
        spd = vision.get_speed(raw)
        lap_event = vision.check_lap_change(raw, step, FPS_BASELINE, spd)
        if lap_event:
            laps_detected += 1
            total_reward  += 1000.0
            lap_flash      = 30   # flash 30 frames
            print(f"\n🏁 LAP DETECTED! Lap #{laps_detected} — +1000 reward!\n")

        # Log OCR attempts at each poll (every 30 steps)
        if step % 30 == 0:
            y0, x0, h0, w0 = vision.LAP_ROI
            if h0 > 0 and w0 > 0:
                gray_d = cv2.cvtColor(raw[y0:y0+h0, x0:x0+w0], cv2.COLOR_BGR2GRAY)
                found = "?"
                for tv in [200, 160, 128]:
                    _, thr = cv2.threshold(gray_d, tv, 255, cv2.THRESH_BINARY)
                    big = cv2.resize(thr, (w0*4, h0*4), interpolation=cv2.INTER_NEAREST)
                    big = cv2.copyMakeBorder(big, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=0)
                    import pytesseract
                    for psm in [7, 10]:
                        t = pytesseract.image_to_string(
                            big, config=f"--psm {psm} -c tessedit_char_whitelist=123"
                        ).strip()
                        try:
                            val = int(t)
                            if val in (1, 2, 3):
                                found = f"'{t}' t={tv} psm={psm}"
                                break
                        except ValueError:
                            pass
                    if found != "?":
                        break
                last_ocr_lap = found
                ocr_log.appendleft(f"s{step}: {found}")


    if lap_flash > 0:
        lap_flash -= 1

    # ── Channels ──
    line_ch, _ = vision.get_line_channel(raw)
    road_ch    = vision.get_road_channel(raw)
    brake_ch   = vision.get_brake_channel(raw)

    # Measure actual fps
    actual_fps = 1.0 / max(time.time() - t0, 0.001)

    # ═══════════════════════════════════════
    # DASHBOARD  (wider to fit lap panel)
    # ═══════════════════════════════════════
    try:
        W, H = 700, 420
        db   = np.zeros((H, W, 3), dtype=np.uint8)
        f    = cv2.FONT_HERSHEY_SIMPLEX
        WHITE = (220, 220, 220)
        DIM   = (100, 100, 100)
        CYAN  = (0, 255, 255)
        GOLD  = (0, 200, 255)

        # ── Vision channels (left column) ──
        cv2.putText(db, "AGENT VISION", (10, 14), f, 0.35, CYAN, 1)
        obs_frame = np.stack([line_ch, road_ch, brake_ch], axis=-1)
        labels = ["LINE", "ROAD", "BRAKE", "STACK"]
        colors = [(255, 200, 0), (0, 255, 0), (0, 0, 255), WHITE]
        for i in range(3):
            ch  = cv2.cvtColor(obs_frame[:, :, i], cv2.COLOR_GRAY2BGR)
            ch  = cv2.resize(ch, (64, 64))
            x0  = 10 + i * 74
            db[20:84, x0:x0+64] = ch
            cv2.putText(db, labels[i], (x0 + 15, 96), f, 0.28, colors[i], 1)
        rgb = cv2.resize(obs_frame, (64, 64))
        db[20:84, 232:296] = rgb
        cv2.putText(db, "STACK", (240, 96), f, 0.28, WHITE, 1)

        # ── Stats (top-right) ──
        sx = 320
        fps_col = (0, 255, 0) if actual_fps > 12 else (0, 165, 255)
        cv2.putText(db, f"FPS: {actual_fps:.0f}", (sx, 28), f, 0.45, fps_col, 2)
        cv2.putText(db, f"SPD: {spd or 0} km/h", (sx, 48), f, 0.33, WHITE, 1)
        cv2.putText(db, f"STEP: {step}", (sx, 64), f, 0.3, DIM, 1)
        wp_crossed = (max_wp - start_wp) % NUM_WP if max_wp >= 0 else 0
        cv2.putText(db, f"WP: {wp_crossed}/{NUM_WP}  {prev_progress*100:.0f}%", (sx, 80), f, 0.28, CYAN, 1)

        r_col = (0, 255, 0) if rew >= 0 else (0, 0, 255)
        cv2.putText(db, f"{rew:+.1f}", (sx, 110), f, 0.7, r_col, 2)
        cv2.putText(db, f"EP: {total_reward:+.0f}", (sx, 128), f, 0.3, DIM, 1)

        has_line = (pd and pd[0] is not None)
        if lap_flash > 0:
            status, st_col = "LAP! +1000", (0, 255, 100)
        elif crash:
            status, st_col = "CRASH", (0, 0, 255)
        elif not has_line:
            status, st_col = "NO LINE", (0, 100, 255)
        elif rew > 0.1:
            status, st_col = "RACING", (0, 255, 0)
        else:
            status, st_col = "DRIVING", (200, 200, 200)
        cv2.putText(db, status, (sx, 155), f, 0.5, st_col, 2)

        # ── Rewards breakdown ──
        ry = 110
        cv2.putText(db, "REWARDS", (10, ry), f, 0.33, CYAN, 1)
        def _rc(v):
            return (0, 255, 0) if v > 0.01 else (0, 0, 255) if v < -0.01 else DIM
        for i, (name, val) in enumerate([("Progress", r_prog), ("Time", r_time), ("Steer", r_steer)]):
            yy = ry + 15 + i * 16
            cv2.putText(db, f"{name}:", (10, yy), f, 0.28, DIM, 1)
            cv2.putText(db, f"{val:+.3f}", (85, yy), f, 0.28, _rc(val), 1)
        tot_y = ry + 15 + 3 * 16
        cv2.line(db, (10, tot_y - 3), (160, tot_y - 3), (50, 50, 50), 1)
        cv2.putText(db, f"TOTAL: {rew:+.3f}", (10, tot_y + 10), f, 0.33, r_col, 1)

        # ── Termination timers ──
        ty = tot_y + 22
        cv2.putText(db, "TIMERS", (10, ty), f, 0.33, CYAN, 1)
        bar_w = 140
        for i, (name, frames, limit, color) in enumerate([
            ("STUCK",  stuck_frames,     STUCK_STEPS,  (0, 165, 255)),
            ("LOITER", loiter_frames,    LOITER_STEPS, (0, 255, 255)),
            ("WRONG",  wrong_dir_frames, WRONG_STEPS,  (0, 0, 255)),
        ]):
            yy = ty + 14 + i * 18
            cv2.putText(db, name, (10, yy + 3), f, 0.26, DIM, 1)
            cv2.rectangle(db, (60, yy - 3), (60 + bar_w, yy + 6), (30, 30, 30), -1)
            fill = int((min(frames, limit) / max(1, limit)) * bar_w)
            cv2.rectangle(db, (60, yy - 3), (60 + fill, yy + 6), color, -1)

        # ── Stagnation bar ──
        stag_y = ty + 14 + 3 * 18
        stag_steps = step - last_wp_step
        cv2.putText(db, "STAG", (10, stag_y + 3), f, 0.26, DIM, 1)
        cv2.rectangle(db, (60, stag_y - 3), (60 + bar_w, stag_y + 6), (30, 30, 30), -1)
        stag_fill = int((min(stag_steps, STAGNATION) / STAGNATION) * bar_w)
        cv2.rectangle(db, (60, stag_y - 3), (60 + stag_fill, stag_y + 6), (180, 0, 180), -1)

        # ── LAP DETECTION PANEL (right column) ──
        lx = 520
        cv2.line(db, (lx - 10, 0), (lx - 10, H), (50, 50, 50), 1)
        cv2.putText(db, "LAP DETECTION", (lx, 14), f, 0.35, GOLD, 1)

        # Raw LAP ROI
        y0, x0, h0, w0 = vision.LAP_ROI
        if h0 > 0 and w0 > 0:
            lap_roi_img = cv2.resize(raw[y0:y0+h0, x0:x0+w0], (80, 40))
            db[20:60, lx:lx+80] = lap_roi_img
            cv2.putText(db, "LAP ROI", (lx, 70), f, 0.28, DIM, 1)

            # Thresholded version (what OCR sees)
            gray_roi = cv2.cvtColor(raw[y0:y0+h0, x0:x0+w0], cv2.COLOR_BGR2GRAY)
            _, thr_roi = cv2.threshold(gray_roi, 200, 255, cv2.THRESH_BINARY)
            thr_show = cv2.resize(cv2.cvtColor(thr_roi, cv2.COLOR_GRAY2BGR), (80, 40))
            db[75:115, lx:lx+80] = thr_show
            cv2.putText(db, "THRESH", (lx, 125), f, 0.28, DIM, 1)

        # OCR read + lap history
        cv2.putText(db, f"OCR raw: '{last_ocr_lap}'", (lx, 145), f, 0.32, GOLD, 1)
        cv2.putText(db, f"Lap history: {[l for _,l in vision.lap_read_history[-3:]]}", (lx, 163), f, 0.27, DIM, 1)
        cv2.putText(db, f"Last detected: {vision.last_detected_lap}", (lx, 180), f, 0.32, WHITE, 1)
        cv2.putText(db, f"Laps this run: {laps_detected}", (lx, 198), f, 0.35,
                    (0, 255, 100) if laps_detected > 0 else DIM, 1)

        # OCR log
        cv2.putText(db, "OCR LOG:", (lx, 225), f, 0.28, CYAN, 1)
        for i, entry in enumerate(ocr_log):
            cv2.putText(db, entry[:22], (lx, 242 + i * 14), f, 0.25, DIM, 1)

        # Grace / lap gate status
        if step < GRACE_STEPS:
            gate_status = f"GRACE ({GRACE_STEPS - step} left)"
            gate_col = (0, 200, 200)
        elif step < 300:
            gate_status = f"LAP GATE ({300 - step} left)"
            gate_col = (0, 165, 255)
        else:
            gate_status = "LAP CHECK ACTIVE"
            gate_col = (0, 255, 0)
        cv2.putText(db, gate_status, (lx, 340), f, 0.3, gate_col, 1)

        # Flash overlay on lap detection
        if lap_flash > 0:
            alpha = lap_flash / 30.0
            overlay = db.copy()
            cv2.rectangle(overlay, (0, 0), (W, H), (0, 180, 60), -1)
            cv2.addWeighted(overlay, alpha * 0.3, db, 1 - alpha * 0.3, 0, db)
            cv2.putText(db, f"LAP {laps_detected} COMPLETE! +1000", (60, H//2),
                        f, 1.0, (0, 255, 100), 3)

        cv2.imshow("GT Manual Test", db)
    except Exception as e:
        print(f"⚠️ Dash: {e}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f"\nDone! Steps: {step}, WP: {wp_count}, Laps: {laps_detected}, Total: {total_reward:+.0f}")
