"""
v1_baseline - Baseline version
- Synchronous sequential processing
- YOLO 2 models (video, webcam each)
- imgsz = 640 (default)
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultralytics import YOLO
from utils.pose_compare import PoseCompare


def parse_args():
    parser = argparse.ArgumentParser(description="v1_baseline: Baseline version")
    parser.add_argument("--video", type=str, default="../data/요가-가산-A--나마스카라사나-고급-actorY117-20221017_15.56.54_CAM_1.mp4", help="Reference video path")
    parser.add_argument("--cam", type=int, default=0, help="Webcam device ID")
    parser.add_argument("--image", type=str, default=None, help="Use image file instead of webcam")
    parser.add_argument("--gpu", action="store_true", default=False, help="Enable GPU (default: CPU only)")
    parser.add_argument("--half", action="store_true", default=False, help="Enable FP16 half precision (requires --gpu)")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO input image size (default: 640)")
    parser.add_argument("--test", type=int, nargs="?", const=30, default=0, help="Benchmark mode: run N seconds (default 30) and print avg FPS")
    return parser.parse_args()


def main():
    args = parse_args()

    if not args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if args.half and not args.gpu:
        print("WARNING: --half requires --gpu, ignoring --half")
        args.half = False

    model_path = os.path.join(os.path.dirname(__file__), "..", "models", "yolov8n-pose.pt")

    pose = PoseCompare(imgsz=args.imgsz, half=args.half)

    # Load reference video
    vid = cv2.VideoCapture(args.video)
    vid_fps = vid.get(cv2.CAP_PROP_FPS) or 30.0
    frame_delay = 1.0 / vid_fps

    # Load input source
    if args.image:
        cam_frame = cv2.imread(args.image)
        if cam_frame is None:
            print(f"Cannot read image: {args.image}")
            return
        use_image = True
        cam = None
    else:
        cam = cv2.VideoCapture(args.cam)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc("M", "J", "P", "G"))
        if not cam.isOpened():
            print("Cannot open camera")
            return
        use_image = False

    # 2 YOLO models
    video_yolo = YOLO(model_path)
    webcam_yolo = YOLO(model_path)

    test_mode = args.test > 0
    test_duration = args.test

    prev_time = time.time()
    start_time = prev_time
    frame_count = 0

    if test_mode:
        print(f"v1_baseline BENCHMARK (GPU={'on' if args.gpu else 'off'}, half={args.half}, imgsz={args.imgsz}) — {test_duration}s")
    else:
        print(f"v1_baseline running (GPU={'on' if args.gpu else 'off'}, half={args.half}, imgsz={args.imgsz})... Press 'q' to quit")

    while True:
        loop_start = time.time()
        vid_ret, vid_frame = vid.read()

        if not vid_ret:
            vid.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        pose.load_img(frame=vid_frame, yolo=video_yolo, dest="ref")

        if use_image:
            frame = cam_frame.copy()
        else:
            cam_ret, frame = cam.read()
            if not cam_ret:
                print("Can't receive frame. Exiting ...")
                break
            frame = cv2.flip(frame, 1)

        pose.load_img(frame=frame, yolo=webcam_yolo, dest="trgt")
        pose.counting()
        pose.compare(offset=20)

        frame_count += 1
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time + 1e-6)
        prev_time = curr_time

        if test_mode:
            elapsed = curr_time - start_time
            if elapsed >= test_duration:
                break
        else:
            vid_display = cv2.resize(vid_frame, (960, 720))
            cam_display = cv2.resize(frame, (960, 720))
            cv2.putText(cam_display, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            combine_frame = np.hstack((vid_display, cam_display))
            cv2.imshow("Compare (v1_baseline)", combine_frame)

            elapsed = time.time() - loop_start
            wait_ms = max(1, int((frame_delay - elapsed) * 1000))
            if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                break

    total_elapsed = time.time() - start_time
    avg_fps = frame_count / total_elapsed if total_elapsed > 0 else 0

    if test_mode:
        print("=" * 40)
        print(f"  v1_baseline Benchmark Result")
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
