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

    r_time = -0.1
    rew = r_prog + r_time
    total_reward += rew

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

    # Dashboard
    db = np.zeros((200, 500, 3), dtype=np.uint8)
    f = cv2.FONT_HERSHEY_SIMPLEX
    for i, (ch, lb) in enumerate([(line_ch, "LINE"), (road_ch, "ROAD"), (brake_ch, "BRAKE")]):
        img = cv2.resize(cv2.cvtColor(ch, cv2.COLOR_GRAY2BGR), (50, 50))
        db[10:60, 10+i*60:60+i*60] = img
        cv2.putText(db, lb, (12+i*60, 75), f, 0.28, (200,200,200), 1)

    sx = 200
    cv2.putText(db, f"STEP: {step}  SPD: {spd or 0}", (sx, 20), f, 0.35, (200,200,200), 1)
    cv2.putText(db, f"Progress: {cp*100:.1f}%", (sx, 42), f, 0.4, (0,255,255), 1)
    cv2.putText(db, f"WP crossed: {wp_count}/50", (sx, 62), f, 0.4, (0,255,255), 1)
    cv2.putText(db, f"WP: {cwp} max: {max_wp} start: {start_wp}", (sx, 82), f, 0.28, (100,100,100), 1)

    rc = (0,255,0) if rew >= 0 else (0,0,255)
    cv2.putText(db, f"Reward: {rew:+.1f}", (sx, 110), f, 0.45, rc, 1)
    cv2.putText(db, f"Total: {total_reward:+.0f}", (sx, 135), f, 0.45, (220,220,220), 1)

    bw, by = 460, 175
    cv2.rectangle(db, (20, by), (20+bw, by+12), (50,50,50), -1)
    cv2.rectangle(db, (20, by), (20+int(cp*bw), by+12), (0,200,0), -1)

    cv2.imshow("Reward Test", db)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print(f"\nDone! Steps: {step}, WP: {wp_count}, Total: {total_reward:+.0f}")
