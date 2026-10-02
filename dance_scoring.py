"""
Dance-game scoring: precompute reference poses, then score a live pose against them.

    pip install mediapipe opencv-python numpy

Usage:
    python dance_scoring.py precompute dance.mp4 dance_ref.npz
    # then import score_pose() into your live loop
"""

import sys

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

MODEL_PATH = "pose_landmarker_lite.task"

# Bones we score on, with weights. Limbs carry the dance; the torso is mostly
# a stability reference, so it counts for less.
BONES = [
    ((11, 13), 1.5),  # left upper arm
    ((13, 15), 1.5),  # left forearm
    ((12, 14), 1.5),  # right upper arm
    ((14, 16), 1.5),  # right forearm
    ((23, 25), 1.2),  # left thigh
    ((25, 27), 1.2),  # left shin
    ((24, 26), 1.2),  # right thigh
    ((26, 28), 1.2),  # right shin
    ((11, 12), 0.5),  # shoulder line
    ((23, 24), 0.5),  # hip line
    ((11, 23), 0.8),  # left torso
    ((12, 24), 0.8),  # right torso
]

WEIGHTS = np.array([w for _, w in BONES])
VIS_THRESHOLD = 0.5  # ignore bones whose joints the model isn't confident about


# --------------------------------------------------------------------------
# Pose -> feature vector
# --------------------------------------------------------------------------

def to_array(landmarks):
    """MediaPipe landmark list -> (33, 4) array of x, y, z, visibility."""
    return np.array([[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks])


def limb_vectors(pose):
    """
    (33, 4) pose -> (n_bones, 2) unit direction vectors, plus a (n_bones,) mask
    of which bones are trustworthy.

    Using unit vectors makes this invariant to the person's position in frame,
    their distance from the camera, and their body size.
    """
    xy = pose[:, :2]
    vis = pose[:, 3]

    vecs = np.zeros((len(BONES), 2))
    mask = np.zeros(len(BONES), dtype=bool)

    for i, ((a, b), _) in enumerate(BONES):
        v = xy[b] - xy[a]
        norm = np.linalg.norm(v)
        if norm < 1e-6:
            continue
        vecs[i] = v / norm
        mask[i] = vis[a] > VIS_THRESHOLD and vis[b] > VIS_THRESHOLD

    return vecs, mask


def pose_similarity(pose_a, pose_b):
    """Score two poses from 0.0 (nothing alike) to 1.0 (identical)."""
    va, ma = limb_vectors(pose_a)
    vb, mb = limb_vectors(pose_b)

    valid = ma & mb
    if not valid.any():
        return 0.0

    # cosine similarity per bone, remapped from [-1, 1] to [0, 1]
    cos = np.sum(va * vb, axis=1)
    per_bone = (cos + 1.0) / 2.0

    w = WEIGHTS[valid]
    return float(np.sum(per_bone[valid] * w) / np.sum(w))


# --------------------------------------------------------------------------
# Scoring against the reference track, with timing slack
# --------------------------------------------------------------------------

def score_pose(player_pose, reference, fps, t_ms, slack_ms=250):
    """
    Compare the player's current pose against the reference around time t_ms.

    Takes the BEST match within +/- slack_ms rather than the exact frame,
    because people lag the video. Without this, honest players score terribly.
    """
    center = int(t_ms / 1000 * fps)
    span = int(slack_ms / 1000 * fps)

    lo = max(0, center - span)
    hi = min(len(reference), center + span + 1)
    if lo >= hi:
        return 0.0

    return max(pose_similarity(player_pose, reference[i]) for i in range(lo, hi))


def grade(score):
    if score >= 0.95:
        return "PERFECT"
    if score >= 0.90:
        return "GREAT"
    if score >= 0.82:
        return "GOOD"
    return "MISS"


# --------------------------------------------------------------------------
# Offline: run the model over the dance video once and cache the landmarks
# --------------------------------------------------------------------------

def precompute(video_path, out_path):
    options = vision.PoseLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            # Derive the timestamp from the frame index, not the wall clock --
            # this is a file, not a live feed, and it must still increase.
            result = landmarker.detect_for_video(mp_image, int(idx / fps * 1000))

            if result.pose_landmarks:
                frames.append(to_array(result.pose_landmarks[0]))
            else:
                # No detection: repeat the last good pose so indices stay aligned
                # with video time. Never drop a frame here.
                frames.append(frames[-1] if frames else np.zeros((33, 4)))

            idx += 1

    cap.release()
    np.savez_compressed(out_path, poses=np.array(frames), fps=fps)
    print(f"Saved {len(frames)} frames at {fps:.2f} fps -> {out_path}")


def load_reference(path):
    data = np.load(path)
    return data["poses"], float(data["fps"])


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "precompute":
        precompute(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)