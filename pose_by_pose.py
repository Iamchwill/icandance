"""
pose_gate.py -- a video clip that stops at "gates" and refuses to continue
until you hold the same pose as the dancer on screen.

    python pose_gate.py clip.mp4 --model pose_landmarker_full.task --every 4

Keys:
    space  pause/resume everything
    s      skip the current gate (escape hatch for impossible poses)
    m      toggle mirror mode (default on: you mirror the dancer)
    d      toggle debug readout (per-limb scores)
    r      restart from the top
    q      quit
"""

import argparse
import threading
import time
from dataclasses import dataclass, field

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

BaseOptions = mp_python.BaseOptions
PoseLandmarker = vision.PoseLandmarker
PoseLandmarkerOptions = vision.PoseLandmarkerOptions
RunningMode = vision.RunningMode

# ---------------------------------------------------------------- tuning ----

ENTER_THRESH = 0.88      # score needed to start the hold timer
MAINTAIN_THRESH = 0.82   # score needed to keep it running (hysteresis)
HOLD_SECONDS = 0.6       # how long you must stay in the pose to open the gate
MIN_VISIBILITY = 0.5     # landmarks below this are ignored, weights renormalized

# name -> (parent landmark, child landmark, weight)
LIMBS = {
    "left upper arm":  (11, 13, 1.0),
    "left forearm":    (13, 15, 1.0),
    "right upper arm": (12, 14, 1.0),
    "right forearm":   (14, 16, 1.0),
    "left thigh":      (23, 25, 0.8),
    "left shin":       (25, 27, 0.8),
    "right thigh":     (24, 26, 0.8),
    "right shin":      (26, 28, 0.8),
    "left torso":      (11, 23, 0.5),
    "right torso":     (12, 24, 0.5),
}

SKELETON = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28),
]
DOTS = sorted({i for pair in SKELETON for i in pair})  # body only, no face/hands

PANEL_H = 540

# ------------------------------------------------------------- comparison ---


def limb_vectors(landmarks):
    """Unit direction vectors in image space, keyed by limb name."""
    out = {}
    for name, (a, b, _) in LIMBS.items():
        pa, pb = landmarks[a], landmarks[b]
        if min(getattr(pa, "visibility", 1.0), getattr(pb, "visibility", 1.0)) < MIN_VISIBILITY:
            continue
        v = np.array([pb.x - pa.x, pb.y - pa.y], dtype=np.float32)
        n = np.linalg.norm(v)
        if n < 1e-6:
            continue
        out[name] = v / n
    return out


def compare(ref_lm, live_lm):
    """Weighted mean of per-limb cosine similarity, mapped to 0..1.

    Direction-based rather than coordinate-based, so it survives you standing
    closer to the camera or off to one side than the dancer does.
    """
    ref, live = limb_vectors(ref_lm), limb_vectors(live_lm)
    shared = ref.keys() & live.keys()
    if not shared:
        return 0.0, {}
    per_limb, total_w, acc = {}, 0.0, 0.0
    for name in shared:
        cos = float(np.clip(np.dot(ref[name], live[name]), -1.0, 1.0))
        s = (cos + 1.0) / 2.0
        w = LIMBS[name][2]
        per_limb[name] = (s, ref[name], live[name])
        acc += s * w
        total_w += w
    return acc / total_w, per_limb


def worst_limb_hint(per_limb):
    """Cheap coaching: name the worst limb and whether to raise or lower it."""
    if not per_limb:
        return "step into frame"
    name, (score, rv, lv) = min(per_limb.items(), key=lambda kv: kv[1][0])
    if score > 0.95:
        return "almost there"
    # image y grows downward, so a smaller y-component points more upward
    verb = "raise" if rv[1] < lv[1] else "lower"
    return f"{verb} your {name}"


# ----------------------------------------------------------------- camera ---


class ThreadedCamera:
    """Always hands back the newest frame. Critical here: the camera keeps
    running while the video is frozen at a gate, and we must not accumulate
    a backlog of stale frames while you fumble into position."""

    def __init__(self, index=0, width=960, height=540):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.frame = None
        self.running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self.running:
            ok, f = self.cap.read()
            if not ok:
                time.sleep(0.005)
                continue
            with self.lock:
                self.frame = f

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def release(self):
        self.running = False
        time.sleep(0.05)
        self.cap.release()


# ------------------------------------------------------------------ gates ---


@dataclass
class Gate:
    t: float                  # seconds into the clip
    landmarks: list           # reference pose
    frame: np.ndarray = field(repr=False)
    cleared: bool = False


def extract_gates(video_path, gate_times, model_path):
    """Run the landmarker over the clip at each gate time. The target pose is
    whatever the dancer is actually doing there -- nothing to hand-author."""
    opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_poses=1,
    )
    gates = []
    cap = cv2.VideoCapture(video_path)
    with PoseLandmarker.create_from_options(opts) as lm:
        for t in gate_times:
            cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
            ok, frame = cap.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
            if not res.pose_landmarks:
                print(f"  no pose found at {t:.1f}s -- dropping that gate")
                continue
            gates.append(Gate(t, res.pose_landmarks[0], frame.copy()))
    cap.release()
    return gates


# ---------------------------------------------------------------- drawing ---


def draw_skeleton(img, landmarks, color=(80, 220, 120)):
    h, w = img.shape[:2]
    pts = [(int(p.x * w), int(p.y * h)) for p in landmarks]
    for a, b in SKELETON:
        cv2.line(img, pts[a], pts[b], color, 3, cv2.LINE_AA)
    for i in DOTS:
        cv2.circle(img, pts[i], 5, color, -1, cv2.LINE_AA)


def fit(img, height=PANEL_H):
    h, w = img.shape[:2]
    return cv2.resize(img, (int(w * height / h), height))


def draw_meter(img, score, hold_frac, state):
    h, w = img.shape[:2]
    x0, y0, bw, bh = 20, h - 60, w - 40, 18
    cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (60, 60, 60), -1)
    col = (80, 220, 120) if score >= ENTER_THRESH else (60, 170, 230)
    cv2.rectangle(img, (x0, y0), (x0 + int(bw * score), y0 + bh), col, -1)
    for th in (MAINTAIN_THRESH, ENTER_THRESH):
        tx = x0 + int(bw * th)
        cv2.line(img, (tx, y0 - 3), (tx, y0 + bh + 3), (230, 230, 230), 1)
    if state == "WAITING" and hold_frac > 0:
        cv2.rectangle(img, (x0, y0 + bh + 6), (x0 + int(bw * hold_frac), y0 + bh + 14),
                      (90, 240, 240), -1)


def banner(img, text, color=(255, 255, 255), y=44, scale=1.0):
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(img, text, (20, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 2, cv2.LINE_AA)


# ------------------------------------------------------------------- main ---


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--model", default="pose_landmarker_full.task")
    ap.add_argument("--every", type=float, default=4.0,
                    help="auto-place a gate every N seconds")
    ap.add_argument("--gates", type=str, default=None,
                    help="explicit gate times, e.g. 2.0,5.5,9.25")
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()

    probe = cv2.VideoCapture(args.video)
    fps = probe.get(cv2.CAP_PROP_FPS) or 30.0
    n_frames = int(probe.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = n_frames / fps if n_frames else 0.0
    probe.release()

    if args.gates:
        times = [float(x) for x in args.gates.split(",")]
    else:
        times = list(np.arange(args.every, max(duration, args.every), args.every))

    print(f"clip: {duration:.1f}s @ {fps:.1f}fps -- extracting {len(times)} gates")
    gates = extract_gates(args.video, times, args.model)
    print(f"{len(gates)} gates ready")

    live_opts = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=args.model),
        running_mode=RunningMode.VIDEO,
        num_poses=1,
    )

    cam = ThreadedCamera(args.camera)
    vid = cv2.VideoCapture(args.video)

    mirror = True
    debug = False
    paused = False
    state = "PLAYING"
    gate_i = 0
    hold = 0.0
    video_time = 0.0
    frame_idx = 0
    cur_frame = None
    last = time.time()
    t0 = time.time()

    with PoseLandmarker.create_from_options(live_opts) as lm:
        while True:
            now = time.time()
            dt = min(now - last, 0.1)   # clamp so a hiccup can't skip a gate
            last = now

            # ---- live pose (always running, even while the clip is frozen)
            cam_frame = cam.read()
            score, per_limb, live_lm = 0.0, {}, None
            if cam_frame is not None:
                if mirror:
                    cam_frame = cv2.flip(cam_frame, 1)
                rgb = cv2.cvtColor(cam_frame, cv2.COLOR_BGR2RGB)
                res = lm.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb),
                    int((now - t0) * 1000),
                )
                if res.pose_landmarks:
                    live_lm = res.pose_landmarks[0]
                    draw_skeleton(cam_frame, live_lm)

            gate = gates[gate_i] if gate_i < len(gates) else None

            # ---- state machine
            if state == "PLAYING" and not paused:
                video_time += dt
                if gate and video_time >= gate.t:
                    video_time = gate.t
                    state = "WAITING"
                    hold = 0.0

            if state == "WAITING" and gate and live_lm is not None:
                score, per_limb = compare(gate.landmarks, live_lm)
                gate_open = hold >= HOLD_SECONDS
                if not gate_open:
                    if hold > 0:
                        hold = hold + dt if score >= MAINTAIN_THRESH else 0.0
                    elif score >= ENTER_THRESH:
                        hold = dt
                    if hold >= HOLD_SECONDS:
                        gate.cleared = True
                        gate_i += 1
                        state = "PLAYING"
                        hold = 0.0

            # ---- advance decoded frames up to video_time
            target = int(video_time * fps)
            while frame_idx <= target:
                ok, f = vid.read()
                if not ok:
                    state = "DONE"
                    break
                cur_frame = f
                frame_idx += 1
            if frame_idx > 0 and video_time * fps < frame_idx - 1 and cur_frame is None:
                pass

            # ---- render
            left = gate.frame.copy() if state == "WAITING" and gate else (
                cur_frame.copy() if cur_frame is not None else np.zeros((PANEL_H, 960, 3), np.uint8))
            if state == "WAITING" and gate:
                draw_skeleton(left, gate.landmarks, color=(90, 200, 255))
                banner(left, "HIT THIS POSE", (90, 200, 255))

            right = cam_frame if cam_frame is not None else np.zeros((PANEL_H, 960, 3), np.uint8)
            if state == "WAITING":
                banner(right, f"{score * 100:4.0f}%", (255, 255, 255))
                banner(right, worst_limb_hint(per_limb), (200, 200, 200), y=84, scale=0.7)
                draw_meter(right, score, min(hold / HOLD_SECONDS, 1.0), state)
                if hold > 0:
                    banner(right, "HOLD IT", (90, 240, 240), y=124, scale=0.8)
            elif state == "DONE":
                banner(right, "CLIP COMPLETE", (90, 220, 120))
            else:
                banner(right, f"gate {gate_i + 1}/{len(gates)} in "
                              f"{(gate.t - video_time):.1f}s" if gate else "free play",
                       (170, 170, 170), scale=0.7)

            if debug and per_limb:
                for i, (name, (s, _, _)) in enumerate(sorted(per_limb.items(),
                                                             key=lambda kv: kv[1][0])):
                    banner(right, f"{name}: {s:.2f}", (150, 150, 150),
                           y=170 + i * 22, scale=0.5)
            if paused:
                banner(right, "PAUSED", (0, 200, 255), y=PANEL_H - 80, scale=0.8)

            cv2.imshow("pose gate", np.hstack([fit(left), fit(right)]))

            # ---- input
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            elif k == ord(" "):
                paused = not paused
            elif k == ord("m"):
                mirror = not mirror
            elif k == ord("d"):
                debug = not debug
            elif k == ord("s") and state == "WAITING":
                gate_i += 1
                state = "PLAYING"
                hold = 0.0
            elif k == ord("r"):
                vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
                frame_idx, video_time, gate_i, hold = 0, 0.0, 0, 0.0
                state = "PLAYING"
                for g in gates:
                    g.cleared = False

    cam.release()
    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()