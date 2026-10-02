"""
Pose by pose loop: reference video + webcam picture-in-picture + scoring test.

Setup:
    pip install mediapipe opencv-python numpy
    python dance_scoring.py precompute dance.mp4 dance_ref.npz

Run:
    python pose_by_pose.py dance.mp4 dance_ref.npz

Keys:
    q / ESC  quit
    m        toggle left/right mirroring (see MIRROR note below)
    d        toggle the debug readout
"""

import sys
import threading
import time
from ffpyplayer.player import MediaPlayer

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from dance_scoring import (
    grade,
    load_reference,
    pose_similarity,
    score_pose,
    to_array,
)

MODEL_PATH = "pose_landmarker_lite.task"

WINDOW = "Dance"
STAGE_W, STAGE_H = 1280, 720
PIP_SCALE = 0.28            # webcam inset width as a fraction of the stage
PIP_MARGIN = 24

KEYFRAME_INTERVAL_MS = 2000  # score a "gold move" every 2s (see note at bottom)
POPUP_MS = 900               # how long a grade stays on screen
CALIBRATION_HOLD_S = 2.0     # how long the player must be fully visible

# Joints that must be visible before the song will start.
REQUIRED_JOINTS = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]
VIS_THRESHOLD = 0.5

# Landmarks to hide when drawing (face 0-10, hand detail 17-22).
SKIP_DRAW = set(range(0, 11)) | set(range(17, 23))
CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
    (27, 31), (28, 32),
]

# MediaPipe's left/right landmark pairs, for mirroring.
LR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
            (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26),
            (27, 28), (29, 30), (31, 32)]

GRADE_COLORS = {
    "PERFECT": (80, 230, 255),
    "GREAT": (120, 255, 120),
    "GOOD": (255, 200, 90),
    "MISS": (110, 110, 240),
}


# --------------------------------------------------------------------------
# Camera capture on its own thread
# --------------------------------------------------------------------------

class CameraStream:
    """
    Grabs frames continuously and keeps only the newest one.

    This matters: MediaPipe's VIDEO mode blocks the main loop, and while it's
    blocked OpenCV keeps queueing camera frames. cap.read() would then hand
    back a stale frame and the player would drift further behind the music as
    the song went on. Draining in a thread means read() is always current.
    """

    def __init__(self, index=0):
        self.cap = cv2.VideoCapture(index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera {index}.")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self._lock = threading.Lock()
        self._frame = None
        self._running = True
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    def _pump(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self._lock:
                self._frame = frame

    def read(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def stop(self):
        self._running = False
        self._thread.join(timeout=1.0)
        self.cap.release()


# --------------------------------------------------------------------------
# Pose helpers
# --------------------------------------------------------------------------

def mirror_pose(pose):
    """Flip a pose left-to-right: swap L/R landmark pairs and mirror x."""
    out = pose.copy()
    for a, b in LR_PAIRS:
        out[[a, b]] = out[[b, a]]
    out[:, 0] = 1.0 - out[:, 0]
    return out


def fully_visible(pose):
    return all(pose[j, 3] > VIS_THRESHOLD for j in REQUIRED_JOINTS)


def draw_skeleton(frame, pose, color=(0, 255, 0)):
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y in pose[:, :2]]
    for a, b in CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], color, 2)
    for i, (x, y) in enumerate(pts):
        if i not in SKIP_DRAW:
            cv2.circle(frame, (x, y), 3, (0, 0, 255), -1)


# --------------------------------------------------------------------------
# Drawing the stage
# --------------------------------------------------------------------------

def fit_letterbox(frame, w, h):
    """Scale to fit inside w x h, padding with black. Keeps aspect ratio."""
    fh, fw = frame.shape[:2]
    scale = min(w / fw, h / fh)
    nw, nh = int(fw * scale), int(fh * scale)
    resized = cv2.resize(frame, (nw, nh))

    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    y0, x0 = (h - nh) // 2, (w - nw) // 2
    canvas[y0:y0 + nh, x0:x0 + nw] = resized
    return canvas


def paste_pip(stage, cam_frame):
    """Webcam inset, bottom-left, with a border."""
    pip_w = int(STAGE_W * PIP_SCALE)
    ch, cw = cam_frame.shape[:2]
    pip_h = int(pip_w * ch / cw)
    pip = cv2.resize(cam_frame, (pip_w, pip_h))

    y0 = STAGE_H - pip_h - PIP_MARGIN
    x0 = PIP_MARGIN
    stage[y0:y0 + pip_h, x0:x0 + pip_w] = pip
    cv2.rectangle(stage, (x0 - 2, y0 - 2), (x0 + pip_w + 2, y0 + pip_h + 2),
                  (255, 255, 255), 2)


def draw_hud(stage, total, hits, live_score, popup, progress):
    # Running total, top right
    cv2.putText(stage, f"{int(total)}", (STAGE_W - 260, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 4)
    if hits:
        cv2.putText(stage, f"{hits} moves", (STAGE_W - 258, 105),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    # Live "how am I doing" meter down the right edge. Feedback only --
    # it deliberately does not contribute to the score.
    bar_h, bar_w = 300, 18
    bx, by = STAGE_W - 60, 150
    cv2.rectangle(stage, (bx, by), (bx + bar_w, by + bar_h), (70, 70, 70), -1)
    fill = int(bar_h * max(0.0, min(1.0, (live_score - 0.6) / 0.4)))
    if fill > 0:
        cv2.rectangle(stage, (bx, by + bar_h - fill), (bx + bar_w, by + bar_h),
                      (120, 255, 120), -1)

    # Grade popup, centre-top
    if popup:
        text, color = popup
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 5)
        cv2.putText(stage, text, ((STAGE_W - tw) // 2, 150),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, color, 5)

    # Song progress
    cv2.rectangle(stage, (0, STAGE_H - 8), (int(STAGE_W * progress), STAGE_H),
                  (255, 255, 255), -1)


def draw_banner(stage, line1, line2=None):
    overlay = stage.copy()
    cv2.rectangle(overlay, (0, STAGE_H // 2 - 90), (STAGE_W, STAGE_H // 2 + 60),
                  (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, stage, 0.4, 0, stage)

    (tw, _), _ = cv2.getTextSize(line1, cv2.FONT_HERSHEY_SIMPLEX, 1.6, 4)
    cv2.putText(stage, line1, ((STAGE_W - tw) // 2, STAGE_H // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
    if line2:
        (tw2, _), _ = cv2.getTextSize(line2, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        cv2.putText(stage, line2, ((STAGE_W - tw2) // 2, STAGE_H // 2 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def run(video_path, ref_path, camera_index=0):
    reference, ref_fps = load_reference(ref_path)

    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        raise RuntimeError(f"Could not open video {video_path}.")
    audio = MediaPlayer(video_path, ff_opts={"paused": True})
    video_fps = video.get(cv2.CAP_PROP_FPS) or ref_fps
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    duration_ms = (total_frames / video_fps) * 1000 if total_frames else None

    camera = CameraStream(camera_index)
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    # Keyframes: the moments that actually count. Evenly spaced here; see the
    # note at the bottom of this file about putting them on the beat instead.
    keyframes = []
    if duration_ms:
        t = KEYFRAME_INTERVAL_MS
        while t < duration_ms - 500:
            keyframes.append(t)
            t += KEYFRAME_INTERVAL_MS

    mirror = True
    debug = False
    state = "calibrating"
    visible_since = None
    song_start = None
    next_keyframe = 0
    video_frame_idx = 0
    last_video_frame = None

    total = 0.0
    hits = 0
    live_score = 0.0
    popup = None
    popup_until = 0.0

    loop_start = time.time()
    last_ts = -1

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            now = time.time()
            cam_frame = camera.read()
            if cam_frame is None:
                continue
            cam_frame = cv2.flip(cam_frame, 1)

            # --- pose detection on the webcam frame ---
            rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts = int((now - loop_start) * 1000)
            if ts <= last_ts:          # timestamps must strictly increase
                ts = last_ts + 1
            last_ts = ts

            result = landmarker.detect_for_video(mp_image, ts)
            player = to_array(result.pose_landmarks[0]) if result.pose_landmarks else None
            if player is not None:
                draw_skeleton(cam_frame, player)

            scored_pose = mirror_pose(player) if (player is not None and mirror) else player

            # ---------------- calibration ----------------
            if state == "calibrating":
                ok = player is not None and fully_visible(player)
                if ok:
                    visible_since = visible_since or now
                    held = now - visible_since
                    if held >= CALIBRATION_HOLD_S:
                        state = "playing"
                        song_start = time.time()
                        audio.set_pause(False)
                else:
                    visible_since = None
                    held = 0.0

                stage = np.zeros((STAGE_H, STAGE_W, 3), dtype=np.uint8)
                paste_pip(stage, cam_frame)
                if ok:
                    draw_banner(stage, f"Starting in {CALIBRATION_HOLD_S - held:.1f}",
                                "Hold still")
                else:
                    draw_banner(stage, "Step back",
                                "Head to feet need to be in frame")
                cv2.imshow(WINDOW, stage)

            # ---------------- playing ----------------
            elif state == "playing":
                t_ms = (time.time() - song_start) * 1000

                # Advance the video to match wall clock. If we've fallen behind,
                # skip frames rather than playing catch-up in slow motion --
                # the audio/beat is the thing the player is following.
                target_idx = int(t_ms / 1000 * video_fps)

                # Keep VLC synchronized with the game clock
                audio.get_frame()


                while video_frame_idx <= target_idx:
                    ok, vf = video.read()
                    if not ok:
                        vf = None
                        break
                    last_video_frame = vf
                    video_frame_idx += 1

                if last_video_frame is None or (duration_ms and t_ms >= duration_ms):
                    state = "finished"
                    audio.set_pause(True)
                    continue

                # Continuous feedback score (meter only, not points)
                if scored_pose is not None:
                    raw = score_pose(scored_pose, reference, ref_fps, t_ms)
                    live_score = 0.8 * live_score + 0.2 * raw

                # Keyframe scoring -- this is what counts
                while next_keyframe < len(keyframes) and t_ms >= keyframes[next_keyframe]:
                    kf_ms = keyframes[next_keyframe]
                    if scored_pose is not None:
                        s = score_pose(scored_pose, reference, ref_fps, kf_ms)
                        g = grade(s)
                        total += max(0.0, (s - 0.6) / 0.4) * 1000
                        hits += 1
                        popup = (g, GRADE_COLORS[g])
                        popup_until = now + POPUP_MS / 1000
                    next_keyframe += 1

                stage = fit_letterbox(last_video_frame, STAGE_W, STAGE_H)
                paste_pip(stage, cam_frame)
                if popup and now > popup_until:
                    popup = None
                progress = t_ms / duration_ms if duration_ms else 0.0
                draw_hud(stage, total, hits, live_score, popup, progress)

                if debug:
                    drift = t_ms - (video_frame_idx / video_fps * 1000)
                    cv2.putText(stage,
                                f"t={t_ms/1000:5.1f}s  drift={drift:6.0f}ms  "
                                f"live={live_score:.2f}  mirror={mirror}",
                                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                                (255, 255, 0), 2)

                cv2.imshow(WINDOW, stage)

            # ---------------- finished ----------------
            else:
                stage = np.zeros((STAGE_H, STAGE_W, 3), dtype=np.uint8)
                avg = total / hits if hits else 0
                draw_banner(stage, f"{int(total)} points",
                            f"{hits} moves  -  {int(avg)} average")
                cv2.imshow(WINDOW, stage)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("m"):
                mirror = not mirror
            if key == ord("d"):
                debug = not debug
            if key == ord("x"):
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                audio= MediaPlayer(video_path, ff_opts={"paused": True})
                state = "playing"
                song_start = time.time()
                audio.set_pause(False)
            if key != 255:
                print(key)                                          

    camera.stop()
    video.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    run(sys.argv[1], sys.argv[2])
