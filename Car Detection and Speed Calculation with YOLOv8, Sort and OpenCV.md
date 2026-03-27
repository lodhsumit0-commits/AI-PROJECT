# 🚗 Car Detection & Speed Calculation
### YOLOv8 · SORT Tracker · OpenCV

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A real-time **vehicle detection and speed estimation** system that combines the power of **YOLOv8** for object detection, **SORT** (Simple Online and Realtime Tracking) for multi-object tracking, and **OpenCV** for video processing and visualization.

---

## 📸 Demo

```
[Video / Camera Feed]
        │
        ▼
[YOLOv8 — Detect Vehicles per Frame]
        │
        ▼
[SORT — Assign & Track Unique IDs]
        │
        ▼
[Speed Calculation — pixels/frame → km/h]
        │
        ▼
[Annotated Output — Boxes · IDs · Speeds]
```

> Each detected vehicle is assigned a unique ID and its speed is estimated in **km/h** as it crosses two virtual reference lines.

---

## 🚀 Features

- ✅ Real-time vehicle detection using **YOLOv8**
- ✅ Multi-vehicle tracking with persistent **SORT** IDs
- ✅ Speed estimation in **km/h** using virtual reference lines & frame rate
- ✅ Works on **video files** and **live camera / IP feeds**
- ✅ Configurable reference lines and calibration parameters
- ✅ Vehicle count per frame + cumulative total
- ✅ Annotated output video export
- ✅ Supports both **CPU** and **GPU (CUDA)** inference

---

## 🛠️ Tech Stack

| Component | Library / Tool | Role |
|-----------|---------------|------|
| Detection | YOLOv8 (Ultralytics) | Detect vehicles per frame |
| Tracking | SORT | Assign consistent IDs across frames |
| Vision | OpenCV (`cv2`) | Video I/O, drawing, display |
| Numerics | NumPy | Distance & speed calculations |
| Backend | PyTorch | YOLOv8 inference engine |

---

## 📁 Project Structure

```
car-speed-detection/
│
├── models/
│   └── yolov8n.pt                       # YOLOv8 weights (auto-downloaded)
│
├── sort/
│   ├── sort.py                          # SORT tracker implementation
│   └── __init__.py
│
├── utils/
│   ├── speed_estimator.py               # Speed calculation logic
│   ├── line_zone.py                     # Virtual reference line helpers
│   └── drawing.py                       # Annotation & drawing utilities
│
├── videos/
│   └── highway.mp4                      # Sample input video
│
├── output/
│   └── result.mp4                       # Annotated output video
│
├── main.py                              # Entry point
├── config.py                            # All configurable parameters
├── requirements.txt
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/car-speed-detection.git
cd car-speed-detection
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
venv\Scripts\activate           # Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
ultralytics>=8.0.0
opencv-python>=4.5.0
numpy>=1.21.0
scipy>=1.7.0
filterpy>=1.4.5
lap>=0.4.0
torch>=2.0.0
```

### 4. Install SORT Tracker

```bash
git clone https://github.com/abewley/sort.git sort/
```

### 5. YOLOv8 Weights (Auto-downloaded on first run)

```bash
# Auto-downloaded by Ultralytics on first use — OR download manually:
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt -P models/
```

---

## 🖥️ Usage

### Run on a Video File

```bash
python main.py --source videos/highway.mp4
```

### Run on Webcam or IP Camera

```bash
python main.py --source 0                          # Webcam (index 0)
python main.py --source rtsp://your-camera-url     # IP Camera
```

### Save Annotated Output

```bash
python main.py --source videos/highway.mp4 --save --output output/result.mp4
```

### Full Example

```bash
python main.py \
  --source videos/highway.mp4 \
  --model models/yolov8n.pt \
  --conf 0.4 \
  --classes 2 5 7 \
  --real-distance 10.0 \
  --fps 30 \
  --device cuda \
  --save \
  --output output/result.mp4
```

### CLI Reference

| Argument | Default | Description |
|----------|---------|-------------|
| `--source` | `0` | Video file path or camera index |
| `--model` | `yolov8n.pt` | YOLOv8 weights path |
| `--conf` | `0.4` | Detection confidence threshold |
| `--classes` | `2 5 7` | COCO class IDs to detect |
| `--real-distance` | `10.0` | Real-world distance between lines (meters) |
| `--fps` | `30` | Input video frame rate |
| `--device` | `cpu` | Inference device: `cpu` or `cuda` |
| `--save` | `False` | Save annotated output video |
| `--output` | `output/result.mp4` | Output video path |

---

## 🧠 How It Works

### Step 1 — Vehicle Detection (YOLOv8)

Each frame is processed by YOLOv8. Only vehicle class IDs are retained.

```python
from ultralytics import YOLO

model = YOLO("models/yolov8n.pt")
results = model(frame, classes=[2, 5, 7], conf=0.4)

detections = []
for r in results:
    for box in r.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])
        detections.append([x1, y1, x2, y2, conf])
```

---

### Step 2 — Multi-Object Tracking (SORT)

Detections are passed to SORT, which uses a **Kalman Filter** + **Hungarian Algorithm** to assign stable IDs across frames.

```python
from sort import Sort
import numpy as np

tracker = Sort(max_age=20, min_hits=3, iou_threshold=0.3)
tracked = tracker.update(np.array(detections))

# tracked shape: [x1, y1, x2, y2, track_id]
for x1, y1, x2, y2, track_id in tracked:
    track_id = int(track_id)
```

---

### Step 3 — Speed Calculation

Two virtual horizontal lines are placed on the frame. When a vehicle's center crosses **Line 1** then **Line 2**, the speed is computed:

```
Speed (km/h) = (Real Distance in meters / Time in seconds) × 3.6
Time (s)     = Frames between crossings / FPS
```

```python
def calculate_speed(frame_cross_1, frame_cross_2, fps, real_distance_m):
    frames_elapsed = frame_cross_2 - frame_cross_1
    time_seconds = frames_elapsed / fps
    speed_mps = real_distance_m / time_seconds
    return round(speed_mps * 3.6, 1)  # km/h
```

---

### Step 4 — Visualization

```python
# Draw bounding box
cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

# Draw speed label
label = f"ID {track_id} | {speed:.1f} km/h"
cv2.putText(frame, label, (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

# Draw reference lines
cv2.line(frame, (0, LINE_1_Y), (frame_width, LINE_1_Y), (255, 0, 0), 2)
cv2.line(frame, (0, LINE_2_Y), (frame_width, LINE_2_Y), (0, 0, 255), 2)
```

---

## 📐 Speed Calibration

Accurate speed estimation depends on calibrating pixel distances to real-world meters.

```python
# config.py

# Y-pixel positions of the two virtual reference lines
LINE_1_Y = 300
LINE_2_Y = 450

# Real-world distance between those two lines (measure on-site or from map)
REAL_DISTANCE_METERS = 10.0

# Video FPS
FPS = 30
```

> 💡 **Tip:** Identify two visible road features in your video (e.g., lane markings, road signs) separated by a known distance, and set your lines to those pixel positions.

---

## 📊 COCO Vehicle Class IDs

| Class ID | Label |
|----------|-------|
| 2 | 🚗 Car |
| 3 | 🏍️ Motorcycle |
| 5 | 🚌 Bus |
| 7 | 🚛 Truck |

Pass multiple IDs with `--classes 2 3 5 7`.

---

## 🎯 YOLOv8 Model Comparison

| Model | Size | Speed | mAP | Best For |
|-------|------|-------|-----|----------|
| `yolov8n.pt` | 6 MB | ⚡ Fastest | 37.3 | CPU / Edge devices |
| `yolov8s.pt` | 22 MB | ⚡ Fast | 44.9 | Balanced |
| `yolov8m.pt` | 50 MB | 🟡 Medium | 50.2 | Accuracy priority |
| `yolov8l.pt` | 87 MB | 🐢 Slower | 52.9 | GPU recommended |
| `yolov8x.pt` | 137 MB | 🐢 Slowest | 53.9 | Maximum accuracy |

---

## ⚡ GPU Acceleration

```bash
# Install PyTorch with CUDA (example for CUDA 11.8)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Run with GPU
python main.py --source videos/highway.mp4 --device cuda
```

Check GPU compatibility at [pytorch.org/get-started](https://pytorch.org/get-started/locally/).

---

## 📌 Roadmap

- [ ] Lane-specific speed measurement
- [ ] Speeding violation alert & logging
- [ ] License plate recognition integration
- [ ] Export speed logs to CSV / Excel
- [ ] Live web dashboard (Flask / Streamlit)
- [ ] Multi-camera synchronized tracking
- [ ] Custom YOLOv8 fine-tuned model support

---

## 🤝 Contributing

Contributions are welcome! To get started:

1. Fork this repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please follow existing code style and include comments for complex logic.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) — State-of-the-art real-time object detection
- [SORT by Alex Bewley](https://github.com/abewley/sort) — Simple Online and Realtime Tracking
- [OpenCV](https://opencv.org/) — Open Source Computer Vision Library
- [filterpy](https://github.com/rlabbe/filterpy) — Kalman filtering used by SORT

---

## 📬 Contact

**Your Name** — [@your-twitter](https://twitter.com/your-twitter) — your.email@example.com

Project Link: [https://github.com/your-username/car-speed-detection](https://github.com/your-username/car-speed-detection)
