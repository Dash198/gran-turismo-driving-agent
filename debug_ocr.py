import cv2
import numpy as np

from vision import VisionInterface

# 1. Initialize Vision
vision = VisionInterface(camera_index=2)

# --- UPDATE THESE WITH YOUR CLICKED COORDS ---
# Format: (y, x, h, w)
TEST_SPEED_ROI = (247, 171, 35, 59)


def validate_speed_roi():
    print("Checking Speedometer ROI... Press 'q' to exit.")

    while True:
        frame = vision.get_frame()
        if frame is None:
            break

        # 2. Extract the ROI using your coordinates
        y, x, h, w = TEST_SPEED_ROI
        speed_roi = frame[y : y + h, x : x + w]

        # 3. Apply the same processing as get_speed()
        gray = cv2.cvtColor(speed_roi, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        # 4. Resize for better visibility in debug
        preview = cv2.resize(thresh, (400, 200), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("RAW ROI (Small)", speed_roi)
        cv2.imshow("WHAT TESSERACT SEES (Processed)", preview)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    validate_speed_roi()
