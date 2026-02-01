import math
import numpy as np
import cv2
from typing import List, Tuple, Dict

# ==== YOLOv8-pose Keypoints (COCO format, 17 points) ==== #
YOLO_KEYPOINTS = {
    0: "nose",
    1: "left_eye",
    2: "right_eye",
    3: "left_ear",
    4: "right_ear",
    5: "left_shoulder",
    6: "right_shoulder",
    7: "left_elbow",
    8: "right_elbow",
    9: "left_wrist",
    10: "right_wrist",
    11: "left_hip",
    12: "right_hip",
    13: "left_knee",
    14: "right_knee",
    15: "left_ankle",
    16: "right_ankle",
}

# keypoints id required to calculate angle (for YOLO format)
# Format: [point_a, point_b (vertex), point_c]
kpts_angle = {
    "left_shoulder": [7, 5, 11],    # elbow-shoulder-hip
    "right_shoulder": [8, 6, 12],   # elbow-shoulder-hip
    "left_arm": [5, 7, 9],          # shoulder-elbow-wrist
    "right_arm": [6, 8, 10],        # shoulder-elbow-wrist
    "left_hip": [5, 11, 13],        # shoulder-hip-knee
    "right_hip": [6, 12, 14],       # shoulder-hip-knee
    "left_leg": [11, 13, 15],       # hip-knee-ankle
    "right_leg": [12, 14, 16],      # hip-knee-ankle
}

# Joint pairs for angle calculation
joint_pairs = [
    ('Left Shoulder', 7, 5, 11),    # elbow-shoulder-hip
    ('Right Shoulder', 8, 6, 12),
    ('Left Elbow', 5, 7, 9),        # shoulder-elbow-wrist
    ('Right Elbow', 6, 8, 10),
    ('Left Hip', 5, 11, 13),        # shoulder-hip-knee
    ('Right Hip', 6, 12, 14),
    ('Left Knee', 11, 13, 15),      # hip-knee-ankle
    ('Right Knee', 12, 14, 16),
]

# COCO skeleton connections for drawing
SKELETON_CONNECTIONS = [
    [0, 1], [0, 2], [1, 3], [2, 4],  # Head
    [5, 6],  # Shoulders
    [5, 7], [7, 9],  # Left arm
    [6, 8], [8, 10],  # Right arm
    [5, 11], [6, 12],  # Torso
    [11, 12],  # Hips
    [11, 13], [13, 15],  # Left leg
    [12, 14], [14, 16],  # Right leg
]


def get_angle(keypoints: np.ndarray, angle_kpts: List[int]) -> float:
    """
    Calculate the joint angle using 3 keypoints.

    Args:
        keypoints: Array of shape (17, 3) with [x, y, confidence] for each point
        angle_kpts: List of 3 keypoint indices [a, b, c] where b is the vertex

    Returns:
        Calculated angle in degrees
    """
    a = keypoints[angle_kpts[0]][:2]
    b = keypoints[angle_kpts[1]][:2]
    c = keypoints[angle_kpts[2]][:2]

    if keypoints[angle_kpts[0]][2] < 0.3 or \
       keypoints[angle_kpts[1]][2] < 0.3 or \
       keypoints[angle_kpts[2]][2] < 0.3:
        return 0.0

    ang = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) - math.atan2(a[1] - b[1], a[0] - b[0])
    )

    ang = ang + 360 if ang < 0 else ang
    ang = ang - 180 if ang > 270 else ang

    return float(ang)


def calculate_angle_3points(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """
    Calculate angle at point b given three points a, b, c.

    Args:
        a, b, c: numpy arrays of [x, y] coordinates

    Returns:
        Angle in degrees (0-180)
    """
    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

    return np.degrees(angle)


def draw_skeleton(frame: np.ndarray, keypoints: np.ndarray,
                  color: Tuple[int, int, int] = (0, 255, 0),
                  line_color: Tuple[int, int, int] = (0, 0, 255)) -> None:
    """
    Draw skeleton on frame using YOLO keypoints.

    Args:
        frame: OpenCV image
        keypoints: Array of shape (17, 3) with [x, y, confidence]
        color: BGR color for keypoints
        line_color: BGR color for skeleton lines
    """
    for connection in SKELETON_CONNECTIONS:
        pt1_idx, pt2_idx = connection
        pt1 = keypoints[pt1_idx]
        pt2 = keypoints[pt2_idx]

        if pt1[2] > 0.3 and pt2[2] > 0.3:
            x1, y1 = int(pt1[0]), int(pt1[1])
            x2, y2 = int(pt2[0]), int(pt2[1])
            cv2.line(frame, (x1, y1), (x2, y2), line_color, 2)

    for kp in keypoints:
        if kp[2] > 0.3:
            x, y = int(kp[0]), int(kp[1])
            cv2.circle(frame, (x, y), 4, color, -1)
