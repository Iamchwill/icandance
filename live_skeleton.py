import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Initialize MediaPipe model
MODEL_PATH = 'pose_landmarker_lite.task'

# Pairs of landmark indices to connect when drawing the skeleton.
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),   # arms + shoulders
    (11, 23), (12, 24), (23, 24),                        # torso
    (23, 25), (25, 27), (24, 26), (26, 28),              # legs
    (27, 31), (28, 32),                                  # feet
]
 
FACE = set(range(0, 11))      # nose, eyes, ears, mouth
HANDS = set(range(17, 23))    # pinky, index, thumb (both hands)
SKIP = FACE | HANDS

def draw_pose(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
 
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for i, (x, y) in enumerate(pts):
        if i in SKIP:
            continue
        cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

def main():
    players = 2;
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,  # VIDEO mode = you pass a timestamp per frame
        num_poses=players,
    )
 
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open camera.")
 
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        start = time.time()
 
        while True:
            ok, frame = cap.read()
            frame = cv2.resize(
                frame,
                (1280, 960),
                interpolation=cv2.INTER_CUBIC
            )

            if not ok:
                break
 
            frame = cv2.flip(frame, 1)  # mirror, feels more natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
 
            timestamp_ms = int((time.time() - start) * 1000)
            result = landmarker.detect_for_video(mp_image, timestamp_ms)
 
            if result.pose_landmarks:
                for pose in range(players):
                    draw_pose(frame, result.pose_landmarks[0])
    
            cv2.imshow("Pose", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
 
    cap.release()
    cv2.destroyAllWindows()
 
 
if __name__ == "__main__":
    main()