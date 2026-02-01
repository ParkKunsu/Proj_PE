# prj_pose

YOLOv8-pose 기반 실시간 자세 비교 및 반복 횟수 카운팅 시스템.
레퍼런스 영상(동영상)과 타겟(웹캠/이미지)의 관절 각도를 비교하여 자세 일치도를 판단한다.

## 구조

```
prj_pose/
├── data/                    # 영상, 이미지 데이터
├── models/                  # 모델 파일 (.pt, .engine, .onnx)
├── utils/                   # 공유 유틸리티
│   ├── helpers.py           # 키포인트 정의, 각도 계산, 스켈레톤 그리기
│   ├── count.py             # 반복 횟수 카운팅 (좌표/각도 방식)
│   └── pose_compare.py      # PoseCompare 클래스 (imgsz, half 파라미터)
├── v1_baseline/run.py       # 동기식 기본
├── v2_optimized/run.py      # 스레드 웹캠
├── v3_async/run.py          # 비동기 추론
├── v4_tensorrt/             # TensorRT
│   ├── run.py               # 스레드 웹캠 + 비동기 추론 + TRT
│   ├── run_base.py          # 기본 + TRT (예정)
│   ├── run_thread.py        # 스레드 웹캠 + TRT (예정)
│   ├── run_async.py         # 비동기 추론 + TRT (예정)
│   └── export.py            # .pt → .engine 변환
└── 메모.md
```

## 버전별 특징

| 버전 | 핵심 기능 | GPU 기본 | imgsz 기본 | 웹캠 | 추론 | 모델 |
|------|----------|---------|-----------|------|------|------|
| v1_baseline | 동기식 기본 | off | 640 | 동기 | 동기 | 2개 |
| v2_optimized | 스레드 웹캠 | off | 640 | 스레드 | 동기 | 2개 |
| v3_async | 비동기 추론 | off | 640 | 스레드 | 비동기 | 2개 |
| v4_tensorrt | TensorRT | **on** | **320** | 스레드 | 비동기 | 2개 |

- 모든 버전에서 YOLO 모델 2개 사용 (영상/웹캠 트래킹 상태 분리)
- `python run.py`만 실행하면 각 버전의 핵심 기능만 적용됨

### v4_tensorrt 비교 모드

| 파일 | 웹캠 | 추론 | 설명 |
|------|------|------|------|
| run_base.py | 동기 | 동기 | 기본 + TRT |
| run_thread.py | 스레드 | 동기 | 스레딩만 + TRT |
| run_async.py | 동기 | 비동기 | 비동기만 + TRT |
| run.py | 스레드 | 비동기 | 스레딩 + 비동기 + TRT |

### 벤치마크 결과

최저 10.71 FPS(CPU, imgsz 640, 스레딩+비동기)에서 최고 77.55 FPS(TensorRT, GPU, FP16, imgsz 320, 동기식) 달성 — **624% 증가**.
GPU 전환이 가장 큰 성능 향상 요인(2.6~4배)이며, 사용 모델(yolov8n-pose)이 가벼워 스레딩/비동기의 오버헤드가 추론 시간보다 커서 동기식이 더 빠른 결과를 보임.

## 사용법

### 기본 실행

```bash
cd v1_baseline && python run.py      # CPU, 640, 동기
cd v2_optimized && python run.py     # CPU, 640, 스레드 웹캠
cd v3_async && python run.py         # CPU, 640, 비동기 추론
cd v4_tensorrt && python run.py      # GPU, 320, TRT
```

### 공통 옵션 (모든 버전)

```
--video <path>    레퍼런스 영상 경로
--cam <id>        웹캠 디바이스 ID (기본: 0)
--image <path>    웹캠 대신 이미지 사용
--gpu             GPU 사용 (v1~v3 기본: off, v4 기본: on)
--half            FP16 반정밀도 (--gpu 필요)
--imgsz <int>     YOLO 입력 크기 (v1~v3 기본: 640, v4 기본: 320)
--test [초]       벤치마크 모드 (기본 30초, 워밍업 후 측정)
```

### 예시

```bash
# v1 GPU + FP16 + 작은 이미지
python run.py --gpu --half --imgsz 320

# v2 벤치마크 60초
python run.py --test 60

# v4 이미지 입력
python run.py --image ../data/sample.jpg
```

### TensorRT 엔진 생성

```bash
cd v4_tensorrt && python export.py
```

`models/yolov8n-pose.engine` 생성. `.engine`은 export 시 imgsz에 고정됨 (기본 320).
다른 크기 사용 시 해당 크기로 다시 export 필요.

## 영상 재생

디스플레이 모드에서 레퍼런스 영상은 원본 FPS에 맞춰 재생됨.
`--test` 벤치마크 모드에서는 FPS 제한 없이 최대 속도로 측정.

## 자세 비교 로직

- YOLO 17개 키포인트에서 8개 관절 각도 계산
- 레퍼런스와 타겟의 각도 차이를 offset(기본 25°)과 비교
- 5개 이상 일치 → **O** (초록) / 3~4개 → **△** (노랑) / 미만 → **X** (빨강)

## 요구사항

- Python 3.10+
- ultralytics
- opencv-python
- numpy
- torch (GPU 사용 시 CUDA 버전)
- tensorrt (v4_tensorrt 사용 시)
