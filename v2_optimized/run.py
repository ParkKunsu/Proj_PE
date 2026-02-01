"""
v2_optimized - Threaded webcam
- Threaded webcam capture (always latest frame, no buffer delay)
- YOLO 2 models (video/webcam separate for tracking)
- GPU, FP16, imgsz are configurable via argparse
"""

import os
import sys
import argparse
import cv2
import numpy as np
import time
import threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from ultralytics import YOLO
from utils.pose_compare import PoseCompare


class WebcamStream:
    """Threaded webcam capture - always keeps the latest frame only."""
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        self.ret = False
        self.frame = None
        self.lock = threading.Lock()
        self.stopped = False

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.cap.read()
            with self.lock:
                self.ret = ret
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is not None:
                return self.ret, self.frame.copy()
            return False, None

    def isOpened(self):
        return self.cap.isOpened()

    def release(self):
        self.stopped = True
        self.thread.join(timeout=1.0)
        self.cap.release()


class VideoStream:
    """Video file stream wrapper (loops on end)."""
    def __init__(self, path):
        self.cap = cv2.VideoCapture(path)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

    def read(self):
        ret, frame = self.cap.read()
        if not ret:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
        return ret, frame

    def release(self):
        self.cap.release()


def parse_args():
    parser = argparse.ArgumentParser(description="v2_optimized: GPU + imgsz=320 + threaded webcam")
    parser.add_argument('--video', type=str, default="../data/요가-가산-A--나마스카라사나-고급-actorY117-20221017_15.56.54_CAM_1.mp4",
                        help="Reference video path")
    parser.add_argument('--cam', type=int, default=0,
                        help="Webcam device ID")
    parser.add_argument('--image', type=str, default=None,
                        help="Use image file instead of webcam")
    parser.add_argument('--gpu', action='store_true', default=False,
                        help="Enable GPU (default: CPU only)")
    parser.add_argument('--half', action='store_true', default=False,
                        help="Enable FP16 half precision (requires --gpu)")
    parser.add_argument('--imgsz', type=int, default=640,
                        help="YOLO input image size (default: 640)")
    parser.add_argument('--test', type=int, nargs='?', const=30, default=0,
                        help="Benchmark mode: run N seconds (default 30) and print avg FPS")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8n-pose.pt')

    if not args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if args.half and not args.gpu:
        print("WARNING: --half requires --gpu, ignoring --half")
        args.half = False

    print("=" * 50)
    print("v2_optimized - Threaded Webcam")
    print("=" * 50)

    pose_compare = PoseCompare(imgsz=args.imgsz, half=args.half)

    # Load reference video
    vid = VideoStream(args.video)
    vid_fps = vid.fps or 30.0
    frame_delay = 1.0 / vid_fps
    print(f"Reference video: {args.video}")

    # Load input source
    use_image = False
    use_webcam = False
    cam_frame_static = None

    if args.image:
        cam_frame_static = cv2.imread(args.image)
        if cam_frame_static is None:
            print(f"Cannot read image: {args.image}")
            return
        use_image = True
        cam = None
        print(f"Input: image ({args.image})")
    else:
        cam = WebcamStream(args.cam)
        use_webcam = cam.isOpened()

        if use_webcam:
            print("Input: webcam (threaded mode)")
        else:
            print("Webcam not available, using video file instead...")
            cam.release()
            cam = VideoStream(args.video)

    # 2 YOLO models (video/webcam separate for tracking state)
    print("Loading YOLO models...")
    video_yolo = YOLO(model_path)
    webcam_yolo = YOLO(model_path)

    print(f"GPU={'on' if args.gpu else 'off'}, half={args.half}, imgsz={args.imgsz}")

    test_mode = args.test > 0
    test_duration = args.test

    prev_time = time.time()
    start_time = prev_time
    frame_count = 0

    if test_mode:
        print(f"\nBENCHMARK — {test_duration}s")
    else:
        print("\nRunning... Press 'q' to quit")
    print("-" * 50)

    while True:
        loop_start = time.time()
        vid_ret, vid_frame = vid.read()
        if not vid_ret or vid_frame is None:
            continue

        if use_image:
            cam_ret, cam_frame = True, cam_frame_static.copy()
        else:
            cam_ret, cam_frame = cam.read()

        if not cam_ret or cam_frame is None:
            continue

        if use_webcam:
            cam_frame = cv2.flip(cam_frame, 1)

        pose_compare.load_img(frame=vid_frame, yolo=video_yolo, dest="ref")
        pose_compare.load_img(frame=cam_frame, yolo=webcam_yolo, dest="trgt")
        pose_compare.counting()
        pose_compare.compare(offset=25)

        frame_count += 1
        curr_time = time.time()
        prev_time = curr_time

        if test_mode:
            if curr_time - start_time >= test_duration:
                break
        else:
            fps = frame_count / (curr_time - start_time) if curr_time > start_time else 0

            vid_display = cv2.resize(vid_frame, (640, 480))
            cam_display = cv2.resize(cam_frame, (640, 480))

            cv2.putText(cam_display, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            combine_frame = np.hstack((vid_display, cam_display))
            cv2.imshow("Compare (v2_optimized)", combine_frame)

            elapsed = time.time() - loop_start
            wait_ms = max(1, int((frame_delay - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break

    total_elapsed = time.time() - start_time
    avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0

    if test_mode:
        print("=" * 40)
        print(f"  v2_optimized Benchmark Result")
        print(f"  Frames : {frame_count}")
        print(f"  Time   : {total_elapsed:.1f}s")
        print(f"  Avg FPS: {avg_fps:.2f}")
        print("=" * 40)

    vid.release()
    if cam:
        cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
