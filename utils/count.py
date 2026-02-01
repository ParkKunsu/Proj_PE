import numpy as np


def count_repetition(previous_pose, current_pose, previous_state, flag, tolerance=35):
    """
    Determine the number of repetitions of the pose by using the x and y coordinates of the landmarks.
    Modified for YOLO keypoints (17 points).

    Args:
        previous_pose : keypoints of the previous frame (numpy array, shape: 17x3)
        current_pose : current keypoints (numpy array, shape: 17x3)
        previous_state : Check the changes in each part [x_state, y_state]
        flag : pose change flag
        tolerance : threshold. Defaults to 35.

    Returns:
        current_pose : be the previous_pose
        current_state : copy of current state
        flag : flag after calculation
    """

    if current_pose is None or len(current_pose) == 0:
        return previous_pose, previous_state, flag

    if previous_pose is None or len(previous_pose) == 0:
        return current_pose, previous_state, flag

    current_state = previous_state.copy()
    sdx, sdy = 0, 0

    # YOLO uses 17 keypoints. Body parts are indices 5-16
    # (shoulders, elbows, wrists, hips, knees, ankles)
    for i in range(5, 17):
        if current_pose[i][2] < 0.3 or previous_pose[i][2] < 0.3:
            continue

        dx = current_pose[i][0] - previous_pose[i][0]
        dy = current_pose[i][1] - previous_pose[i][1]

        dx, dy = dx * 0.1, dy * 0.1

        if abs(dx) < tolerance:
            dx = 0
        if abs(dy) < tolerance:
            dy = 0

        sdx += dx
        sdy += dy

    if sdx > (tolerance * 3):
        current_state[0] = 1
    elif sdx < (tolerance * -3):
        current_state[0] = 0
    if sdy > (tolerance * 3):
        current_state[1] = 1
    elif sdy < (tolerance * -3):
        current_state[1] = 0

    if current_state != previous_state:
        flag = (flag + 1) % 2

    return current_pose, current_state.copy(), flag


def count_repetition2(angles, previous_state, tolerance=60):
    """
    Count repetitions based on joint angles.

    Args:
        angles : dictionary of joint angles
        previous_state : previous state of each joint
        tolerance : angle threshold

    Returns:
        current_state : updated state
        flag : 1 if state changed, 0 otherwise
    """
    current_state = previous_state.copy()
    flag = 0

    for joint, angle in angles.items():
        if angle > (180 - tolerance):
            current_state[joint] = 1
        elif angle < tolerance:
            current_state[joint] = 0

    if current_state != previous_state:
        flag = 1

    return current_state, flag
