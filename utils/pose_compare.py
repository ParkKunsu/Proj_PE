"""
Shared Pose Compare module.
imgsz, half are passed as parameters so each version can use different settings.
"""

from typing import Dict, Optional
import cv2
import numpy as np
from collections import defaultdict

from utils.count import count_repetition, count_repetition2
from utils.helpers import (
    get_angle,
    kpts_angle,
    joint_pairs,
    draw_skeleton,
    calculate_angle_3points
)


class PoseCompare:

    def __init__(self, imgsz: int = 640, half: bool = False) -> None:
        self.imgsz = imgsz
        self.half = half

        self.cam_person_info = {}
        self.video_person_info = {}

        self.frame_trgt = None
        self.frame_ref = None

        self.person_states = defaultdict(lambda: [2, 2])
        self.person_reps = defaultdict(int)
        self.person_previous_poses = {}
        self.person_flags = defaultdict(lambda: -1)

        self.person_states2 = defaultdict(lambda: {joint[0]: 2 for joint in joint_pairs})
        self.person_reps2 = defaultdict(int)

    def inference(self, frame, yolo, dest: str):
        """
        Use YOLO-pose to detect people and extract keypoints.

        Args:
            frame: video/webcam frame
            yolo: YOLOv8-pose model
            dest: "ref" (video) or "trgt" (webcam)
        """
        results = yolo.track(
            frame,
            imgsz=self.imgsz,
            half=self.half,
            persist=True,
            conf=0.3,
            iou=0.5,
            show=False,
            tracker="bytetrack.yaml",
            verbose=False
        )[0]

        if dest == "trgt":
            self.cam_person_info.clear()

        if results.keypoints is None or len(results.keypoints) == 0:
            return

        for i, (box, kpts) in enumerate(zip(results.boxes, results.keypoints)):
            if box.id is not None:
                person_id = f"person_{int(box.id[0])}"
            else:
                person_id = f"person_{i}"

            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            keypoints = kpts.data[0].cpu().numpy()

            draw_skeleton(frame, keypoints)

            angles = {}
            for k, v in kpts_angle.items():
                angles[k] = get_angle(keypoints, v)

            angles2 = {}
            for joint_name, start_idx, mid_idx, end_idx in joint_pairs:
                if (keypoints[start_idx][2] > 0.3 and
                    keypoints[mid_idx][2] > 0.3 and
                    keypoints[end_idx][2] > 0.3):
                    start_pt = keypoints[start_idx][:2]
                    mid_pt = keypoints[mid_idx][:2]
                    end_pt = keypoints[end_idx][:2]
                    angles2[joint_name] = calculate_angle_3points(start_pt, mid_pt, end_pt)
                else:
                    angles2[joint_name] = 0.0

            person_info = {
                'bbox': (x1, y1, x2, y2),
                'angles': angles,
                'keypoints': keypoints,
                'angles2': angles2,
            }

            if dest == "trgt":
                self.cam_person_info[person_id] = person_info
            else:
                self.video_person_info = person_info
                break

    def counting(self):
        if self.frame_trgt is None:
            return

        trgt_frame = self.frame_trgt

        for person_id, result in self.cam_person_info.items():
            cam_x1, cam_y1, _, _ = result['bbox']
            cam_keypoints = result['keypoints']

            if person_id not in self.person_previous_poses:
                self.person_previous_poses[person_id] = cam_keypoints

            previous_pose, current_state, flag = count_repetition(
                self.person_previous_poses[person_id],
                cam_keypoints,
                self.person_states[person_id],
                self.person_flags[person_id]
            )
            self.person_previous_poses[person_id] = previous_pose
            self.person_states[person_id] = current_state
            self.person_flags[person_id] = flag

            if flag == 1:
                self.person_reps[person_id] += 1
                self.person_flags[person_id] = -1

            result_str = f"count: {self.person_reps[person_id]}"
            cv2.putText(trgt_frame, result_str, (cam_x1, cam_y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(trgt_frame, result_str, (cam_x1, cam_y1),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    def compare(self, offset: int = 20):
        """
        Compare joint angles between reference (video) and target (webcam).
        """
        if self.frame_trgt is None:
            return

        trgt_frame = self.frame_trgt

        for _, result in self.cam_person_info.items():
            cam_x1, cam_y1, _, _ = result['bbox']
            cam_angles = result['angles']

            angle_diff = self.calc_angle_diff(cam_angles)
            if angle_diff is None:
                continue

            ok_cnt = sum(1 for v in angle_diff.values() if abs(v) < offset)

            if ok_cnt >= 5:
                all_ok = "O"
                color = (0, 255, 0)
            elif 3 <= ok_cnt <= 4:
                all_ok = "△"
                color = (0, 255, 255)
            else:
                all_ok = "X"
                color = (0, 0, 255)

            result_str = f"Match: {all_ok} ({ok_cnt}/8)"
            cv2.putText(trgt_frame, result_str, (cam_x1, cam_y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(trgt_frame, result_str, (cam_x1, cam_y1 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 1)

    def load_img(self, frame, yolo, dest: str):
        self.inference(frame, yolo, dest)

        if dest == "trgt":
            self.frame_trgt = frame
        else:
            self.frame_ref = frame

    def calc_angle_diff(self, cam_angle: Dict) -> Optional[Dict[str, float]]:
        if not self.video_person_info or 'angles' not in self.video_person_info:
            return None

        video_angle = self.video_person_info['angles']

        if len(video_angle) == len(cam_angle):
            angle_diff = {k: video_angle[k] - cam_angle[k] for k in video_angle}
            return angle_diff

        return None

    def calculate_similarity(self, cam_keypoints: np.ndarray) -> float:
        if 'keypoints' not in self.video_person_info:
            return 1.0

        vid_keypoints = self.video_person_info['keypoints']

        if vid_keypoints is None or cam_keypoints is None:
            return 1.0

        vid_pts = vid_keypoints[:, :2]
        cam_pts = cam_keypoints[:, :2]

        vid_pts = (vid_pts - vid_pts.min(axis=0)) / (vid_pts.max(axis=0) - vid_pts.min(axis=0) + 1e-6)
        cam_pts = (cam_pts - cam_pts.min(axis=0)) / (cam_pts.max(axis=0) - cam_pts.min(axis=0) + 1e-6)

        distance = np.mean(np.linalg.norm(vid_pts - cam_pts, axis=1))

        return float(distance)
