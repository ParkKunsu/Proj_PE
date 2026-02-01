"""
TensorRT export script.

Usage:
    python export.py

Result:
    ../models/yolov8n-pose.engine
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ultralytics import YOLO
import torch


def check_environment():
    print("=" * 50)
    print("Environment Check")
    print("=" * 50)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("ERROR: CUDA not available. TensorRT requires NVIDIA GPU.")
        return False

    try:
        import tensorrt
        print(f"TensorRT version: {tensorrt.__version__}")
    except ImportError:
        print("WARNING: TensorRT not installed.")
        print("  Install: pip install tensorrt")

    return True


def export_model(model_path, imgsz=320, half=True):
    print("\n" + "=" * 50)
    print("Exporting to TensorRT")
    print("=" * 50)

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found: {model_path}")
        return None

    print(f"Source model: {model_path}")
    print(f"Input size: {imgsz}")
    print(f"FP16 (half): {half}")

    model = YOLO(model_path)

    print("\nExporting... (this may take 2-5 minutes)")
    engine_path = model.export(
        format="engine",
        imgsz=imgsz,
        half=half,
        device=0,
        verbose=True
    )

    print(f"\nExport complete!")
    print(f"Engine file: {engine_path}")

    return engine_path


def test_engine(engine_path):
    print("\n" + "=" * 50)
    print("Testing TensorRT Engine")
    print("=" * 50)

    if not os.path.exists(engine_path):
        print(f"ERROR: Engine not found: {engine_path}")
        return

    model = YOLO(engine_path)

    import numpy as np
    import time
    dummy_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    print("Warming up...")
    for _ in range(10):
        model(dummy_img, imgsz=320, verbose=False)

    iterations = 100
    start = time.time()
    for _ in range(iterations):
        model(dummy_img, imgsz=320, verbose=False)
    elapsed = time.time() - start

    fps = iterations / elapsed
    print(f"\nPerformance: {fps:.1f} FPS ({elapsed/iterations*1000:.1f} ms per frame)")


def main():
    model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    model_path = os.path.join(model_dir, 'yolov8n-pose.pt')

    if not check_environment():
        return

    engine_path = export_model(
        model_path=model_path,
        imgsz=320,
        half=True
    )

    if engine_path:
        test_engine(engine_path)

        print("\n" + "=" * 50)
        print("Usage")
        print("=" * 50)
        print("Run with TensorRT:")
        print("  cd v3_async && python run.py")
        print("  cd v4_tensorrt && python run.py")
        print("\nThe scripts will auto-detect .engine file.")


if __name__ == "__main__":
    main()
