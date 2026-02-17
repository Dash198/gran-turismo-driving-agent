import cv2
import numpy as np

# Use the camera index you've set for OBS (usually 2 or 10)
CAM_INDEX = 2


def debug_stuck():
    cap = cv2.VideoCapture(CAM_INDEX)

    # Updated ROI based on your previous turn's numbers [cite: 11-02-2026]
    # Format: y, x, h, w
    MAP_ROI = (44, 20, 70, 91)
    prev_pos = None

    print("🔍 Starting Stuck Detection Debugger...")
    print("Check if the green circle is actually following the red car dot.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        y, x, h, w = MAP_ROI
        roi = frame[y : y + h, x : x + w]

        # 1. Isolate the Red Dot
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # Red is split at the 0/180 degree mark in HSV
        mask1 = cv2.inRange(hsv, np.array([0, 140, 50]), np.array([10, 255, 255]))
        mask2 = cv2.inRange(hsv, np.array([170, 140, 50]), np.array([180, 255, 255]))
        mask = cv2.bitwise_or(mask1, mask2)

        # 2. Track the Dot
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        current_pos = None
        dist = 0.0

        if contours:
            largest = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest)
            if M["m00"] != 0:
                current_pos = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # 3. Calculate Jitter/Movement
        if prev_pos and current_pos:
            dist = np.linalg.norm(np.array(current_pos) - np.array(prev_pos))

        # 4. Visualization
        debug_view = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        if current_pos:
            # Draw a green circle where the AI thinks the car is
            cv2.circle(debug_view, current_pos, 4, (0, 255, 0), -1)
            cv2.circle(roi, current_pos, 4, (0, 255, 0), -1)

        cv2.putText(
            debug_view,
            f"Move Dist: {dist:.2f}",
            (5, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2,
        )

        # Show what's happening
        cv2.imshow("1. Original Map ROI", roi)
        cv2.imshow("2. What the AI Sees (Red Mask)", debug_view)

        prev_pos = current_pos

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    debug_stuck()
