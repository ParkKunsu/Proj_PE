"""
v3_async - Async inference with Threading
- Threaded webcam + async inference (display and inference run independently)
- YOLO 2 models (video/webcam separate for tracking)
- TensorRT-ready (.engine auto-detect)
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


class AsyncPoseInference:
    """
    Async inference - runs YOLO in a separate thread.
    Main thread displays at full speed, inference thread processes frames independently.
    """
    def __init__(self, video_yolo, webcam_yolo, pose_compare):
        self.video_yolo = video_yolo
        self.webcam_yolo = webcam_yolo
        self.pose_compare = pose_compare

        self.vid_frame = None
        self.cam_frame = None
        self.lock = threading.Lock()

        self.vid_result_frame = None
        self.cam_result_frame = None
        self.result_lock = threading.Lock()

        self.stopped = False
        self.inference_fps = 0

        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()

    def _inference_loop(self):
        prev_time = time.time()

        while not self.stopped:
            with self.lock:
                vid_frame = self.vid_frame.copy() if self.vid_frame is not None else None
                cam_frame = self.cam_frame.copy() if self.cam_frame is not None else None

            if vid_frame is None or cam_frame is None:
                time.sleep(0.001)
                continue

            self.pose_compare.load_img(frame=vid_frame, yolo=self.video_yolo, dest="ref")
            self.pose_compare.load_img(frame=cam_frame, yolo=self.webcam_yolo, dest="trgt")
            self.pose_compare.counting()
            self.pose_compare.compare(offset=25)

            with self.result_lock:
                self.vid_result_frame = vid_frame.copy()
                self.cam_result_frame = cam_frame.copy()

            curr_time = time.time()
            self.inference_fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

    def submit_frames(self, vid_frame, cam_frame):
        with self.lock:
            self.vid_frame = vid_frame
            self.cam_frame = cam_frame

    def get_results(self):
        with self.result_lock:
            vid = self.vid_result_frame.copy() if self.vid_result_frame is not None else None
            cam = self.cam_result_frame.copy() if self.cam_result_frame is not None else None
        return vid, cam

    def stop(self):
        self.stopped = True
        self.thread.join(timeout=1.0)


def parse_args():
    parser = argparse.ArgumentParser(description="v3_async: Async inference with threading")
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

    if not args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if args.half and not args.gpu:
        print("WARNING: --half requires --gpu, ignoring --half")
        args.half = False

    model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'yolov8n-pose.pt')

    print("=" * 60)
    print("v3_async - Async Inference")
    print("=" * 60)

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
            print("Input: webcam (threaded)")
        else:
            print("Webcam not available, using video file")
            cam.release()
            cam = VideoStream(args.video)

    # Load 2 YOLO models
    print(f"Loading PyTorch model: {model_path}")
    video_yolo = YOLO(model_path)
    webcam_yolo = YOLO(model_path)

    print(f"GPU={'on' if args.gpu else 'off'}, half={args.half}, imgsz={args.imgsz}")

    # Start async inference
    async_inference = AsyncPoseInference(video_yolo, webcam_yolo, pose_compare)
    print("\nAsync inference thread started!")

    test_mode = args.test > 0
    test_duration = args.test

    start_time = time.time()
    prev_time = start_time
    frame_count = 0
    inference_fps_samples = []

    if test_mode:
        print(f"\nBENCHMARK — {test_duration}s")
    else:
        print("\nRunning... Press 'q' to quit")
    print("-" * 60)

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

        async_inference.submit_frames(vid_frame.copy(), cam_frame.copy())

        vid_result, cam_result = async_inference.get_results()

        frame_count += 1
        curr_time = time.time()

        if async_inference.inference_fps > 0:
            inference_fps_samples.append(async_inference.inference_fps)

        if test_mode:
            if curr_time - start_time >= test_duration:
                break
        else:
            if vid_result is not None:
                vid_display = vid_result
                cam_display = cam_result
            else:
                vid_display = vid_frame
                cam_display = cam_frame

            vid_display = cv2.resize(vid_display, (640, 480))
            cam_display = cv2.resize(cam_display, (640, 480))

            display_fps = 1.0 / (curr_time - prev_time + 1e-6)
            prev_time = curr_time

            cv2.putText(cam_display, f"Display: {display_fps:.0f} FPS", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(cam_display, f"Inference: {async_inference.inference_fps:.0f} FPS", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            combine_frame = np.hstack((vid_display, cam_display))
            cv2.imshow("Compare (v3_async)", combine_frame)

            elapsed = time.time() - loop_start
            wait_ms = max(1, int((frame_delay - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
                break

    total_elapsed = time.time() - start_time
    avg_display_fps = frame_count / total_elapsed if total_elapsed > 0 else 0
    avg_inference_fps = sum(inference_fps_samples) / len(inference_fps_samples) if inference_fps_samples else 0

    if test_mode:
        print("=" * 40)
        print(f"  v3_async Benchmark Result")
        print(f"  Frames        : {frame_count}")
        print(f"  Time          : {total_elapsed:.1f}s")
        print(f"  Avg Display   : {avg_display_fps:.2f} FPS")
        print(f"  Avg Inference : {avg_inference_fps:.2f} FPS")
        print("=" * 40)

    async_inference.stop()
    vid.release()
    if cam:
        cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
