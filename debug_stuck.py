import cv2
import numpy as np

from vision import VisionInterface


def debug_collisions():
    vision = VisionInterface(camera_index=2)
    print("💥 COLLISION DEBUGGER")
    print("Action: Scrape a wall and watch the 'Spark Mask' window.")

    while True:
        frame = vision.get_frame()
        if frame is None:
            continue

        h, w = frame.shape[:2]
        y, x, height, width = vision.COL_ROI

        # 1. Extract the Collision ROI
        # Ensure the ROI isn't empty
        if height > 0 and width > 0:
            col_roi = frame[y : y + height, x : x + width]
            hsv = cv2.cvtColor(col_roi, cv2.COLOR_BGR2HSV)

            # 2. Spark Color Range (Yellow/Orange/White-ish)
            # We'll use a wider range for debugging:
            # Hue: 10-45 (Orange to Yellow)
            # Sat: 50-255 (Allow lower saturation for 'white' sparks)
            # Val: 150-255 (Must be bright)
            lower_spark = np.array([10, 50, 150])
            upper_spark = np.array([45, 255, 255])

            spark_mask = cv2.inRange(hsv, lower_spark, upper_spark)
            spark_count = cv2.countNonZero(spark_mask)

            # Visual Feedback
            display = frame.copy()
            cv2.rectangle(display, (x, y), (x + width, y + height), (255, 0, 255), 2)

            if spark_count > 20:
                status = f"COLLISION! Pixels: {spark_count}"
                color = (0, 0, 255)
            else:
                status = "CLEAR"
                color = (0, 255, 0)

            cv2.putText(
                display, status, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2
            )

            # Show the spark mask specifically
            cv2.imshow("Spark Mask (What AI sees)", spark_mask)
            cv2.imshow("Collision Monitor", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    vision.cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    debug_collisions()
