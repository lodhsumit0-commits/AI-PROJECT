# 🖐️ Hand Gesture Volume Control — OpenCV + MediaPipe

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10-FF6F00?style=for-the-badge&logo=google&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-lightgrey?style=for-the-badge)
![License](https://img.shields.io/badge/License-MIT-22C55E?style=for-the-badge)

**Control your laptop's system volume using just your hand — no hardware, no touch, no keyboard.**  
Pinch your thumb and index finger together to adjust volume in real time via your webcam.

[How It Works](#-how-it-works) · [Installation](#-installation) · [Usage](#-usage) · [Gestures](#-gesture-guide) · [Architecture](#-architecture) · [Demo](#-demo)

</div>

---

## 📸 Demo

```
📷 Webcam feed active...
🖐️  Hand detected  |  Landmarks: 21
📏  Finger distance: 87px  →  Volume: 65%  🔊
📏  Finger distance: 32px  →  Volume: 20%  🔉
📏  Finger distance: 12px  →  Volume:  0%  🔇
```

> **Pinch in** → Volume Down 🔉  
> **Spread out** → Volume Up 🔊  
> **Full pinch** → Mute 🔇

---

## ✨ Features

- 🖐️ **Real-time hand tracking** via MediaPipe (21 landmarks)
- 🔊 **Smooth volume control** — pinch distance maps linearly to system volume
- 🔇 **Auto-mute** when fingers fully pinched together
- 📊 **Live HUD overlay** — volume bar, percentage, and FPS on webcam feed
- 💡 **Visual feedback** — color-coded circle on fingertips while gesturing
- ⚡ **Low latency** — ~30 FPS on standard laptop webcam
- 🖥️ **Cross-platform** — Windows (pycaw), macOS (osascript), Linux (amixer)
- 🧩 **Modular design** — separate detector, controller, and display modules

---

## 🗂️ Project Structure

```
hand-gesture-volume-control/
│
├── 📂 src/
│   ├── hand_detector.py       # MediaPipe hand landmark detection
│   ├── volume_controller.py   # OS-level volume control (cross-platform)
│   ├── gesture_recognizer.py  # Pinch distance → volume mapping logic
│   ├── hud_overlay.py         # OpenCV drawing: bar, circle, FPS, text
│   └── utils.py               # Helper functions (smoothing, interpolation)
│
├── 📂 assets/
│   ├── demo.gif               # Demo animation
│   └── hand_landmarks.png     # MediaPipe landmark diagram
│
├── 📂 tests/
│   ├── test_detector.py
│   ├── test_volume.py
│   └── test_gesture.py
│
├── main.py                    # Entry point — runs webcam loop
├── config.py                  # All tunable parameters
├── requirements.txt
└── README.md
```

---

## ⚙️ How It Works

```
┌─────────────────────────────────────────────────────────┐
│                     PIPELINE                            │
│                                                         │
│  Webcam Frame                                           │
│       ↓                                                 │
│  MediaPipe Hands  →  21 Hand Landmarks (x, y, z)       │
│       ↓                                                 │
│  Extract Landmark #4 (Thumb Tip)                        │
│  Extract Landmark #8 (Index Finger Tip)                 │
│       ↓                                                 │
│  Calculate Euclidean Distance between tips              │
│       ↓                                                 │
│  Interpolate distance → Volume %                        │
│  (e.g., 20px → 0%,  200px → 100%)                      │
│       ↓                                                 │
│  Set System Volume via OS API                           │
│       ↓                                                 │
│  Draw HUD Overlay on Frame → Display                    │
└─────────────────────────────────────────────────────────┘
```

### MediaPipe Hand Landmarks Used

```
        8  ← INDEX FINGER TIP
        |
        7
        |
        6
        |
        5
       /
      4  ← THUMB TIP
     /
    3
   /
  2
 /
1
0 (Wrist)
```

The **Euclidean distance** between landmarks `#4` (thumb tip) and `#8` (index tip) drives the volume:

```python
distance = math.hypot(x8 - x4, y8 - y4)
volume = numpy.interp(distance, [MIN_DIST, MAX_DIST], [0, 100])
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.8 or higher
- A working webcam
- OS: Windows 10/11, macOS 11+, or Ubuntu 20.04+

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/hand-gesture-volume-control.git
cd hand-gesture-volume-control
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### Platform-specific setup

<details>
<summary>🪟 Windows</summary>

```bash
pip install pycaw comtypes
```
Uses the **Windows Core Audio API** via `pycaw` — no extra config needed.

</details>

<details>
<summary>🍎 macOS</summary>

```bash
# No extra packages needed — uses osascript (built-in)
# Grant Terminal/Python microphone + camera access in:
# System Settings → Privacy & Security → Camera / Microphone
```

</details>

<details>
<summary>🐧 Linux</summary>

```bash
sudo apt-get install -y alsa-utils
pip install alsaaudio
```

</details>

---

## 📦 Dependencies

```
opencv-python>=4.8.0
mediapipe>=0.10.0
numpy>=1.24.0
comtypes>=1.2.0       # Windows only
pycaw>=20220416       # Windows only
```

Full list in [`requirements.txt`](requirements.txt).

---

## 🚀 Usage

### Run the app

```bash
python main.py
```

### Optional arguments

```bash
python main.py --cam 0           # Camera index (default: 0)
python main.py --min-dist 20     # Min pinch distance for 0% volume (px)
python main.py --max-dist 200    # Max distance for 100% volume (px)
python main.py --smooth 5        # Smoothing window (frames)
python main.py --no-hud          # Disable on-screen overlay
python main.py --debug           # Show landmark indices on hand
```

### Example

```bash
python main.py --cam 0 --smooth 7 --min-dist 25 --max-dist 180
```

### Quit

Press **`Q`** or **`ESC`** in the webcam window to exit.

---

## 🖐️ Gesture Guide

| Gesture | Action | Visual Cue |
|---|---|---|
| Spread thumb & index wide | 🔊 Volume 100% | Green circle, full bar |
| Medium pinch | 🔉 Volume ~50% | Yellow circle, half bar |
| Pinch almost closed | 🔈 Volume ~10% | Orange circle, low bar |
| Full pinch (touching) | 🔇 Mute | Red circle, empty bar |
| No hand detected | ⏸️ Volume unchanged | No overlay |

> **Tip:** Keep your hand 30–60 cm from the camera for best accuracy. Ensure good lighting.

---

## 🎨 HUD Overlay

The live webcam feed shows:

```
┌────────────────────────────────────────┐
│  FPS: 28                               │
│                                        │
│  [hand with circles on thumb & index]  │
│  ● — — — — — — — ●  87px              │
│                                        │
│  VOL ████████░░  65%                  │
└────────────────────────────────────────┘
```

- **Green dot** on thumb tip, **blue dot** on index tip
- **Line** between tips showing current distance
- **Volume bar** (bottom-left) with percentage
- **FPS counter** (top-left)

---

## 🔧 Configuration

Edit `config.py` to tune behaviour:

```python
# Camera
CAMERA_INDEX          = 0
FRAME_WIDTH           = 1280
FRAME_HEIGHT          = 720

# Hand Detection
MAX_HANDS             = 1
DETECTION_CONFIDENCE  = 0.7
TRACKING_CONFIDENCE   = 0.7

# Gesture → Volume Mapping
MIN_DISTANCE          = 20       # px → 0% volume
MAX_DISTANCE          = 200      # px → 100% volume
SMOOTHING_WINDOW      = 5        # Rolling average over N frames

# HUD
SHOW_LANDMARKS        = True
SHOW_FPS              = True
BAR_COLOR             = (0, 255, 0)   # BGR Green
MUTE_COLOR            = (0, 0, 255)   # BGR Red
```

---

## 📁 Core Code Snippets

### Hand Detection (`src/hand_detector.py`)

```python
import cv2
import mediapipe as mp

class HandDetector:
    def __init__(self, max_hands=1, detection_conf=0.7, tracking_conf=0.7):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=max_hands,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf
        )
        self.mp_draw = mp.solutions.drawing_utils

    def find_hands(self, frame, draw=True):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(rgb)
        if self.results.multi_hand_landmarks and draw:
            for hand_lm in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    frame, hand_lm, self.mp_hands.HAND_CONNECTIONS)
        return frame

    def get_landmark_positions(self, frame, hand_index=0):
        positions = []
        if self.results.multi_hand_landmarks:
            hand = self.results.multi_hand_landmarks[hand_index]
            h, w, _ = frame.shape
            for lm in hand.landmark:
                positions.append((int(lm.x * w), int(lm.y * h)))
        return positions
```

### Volume Controller (`src/volume_controller.py`)

```python
import platform
import numpy as np

class VolumeController:
    def __init__(self):
        self.os = platform.system()
        if self.os == "Windows":
            from ctypes import cast, POINTER
            from comtypes import CLSCTX_ALL
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(
                IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            self.volume = cast(interface, POINTER(IAudioEndpointVolume))
            self.vol_range = self.volume.GetVolumeRange()  # e.g. (-65.25, 0.0)

    def set_volume(self, percent: float):
        percent = np.clip(percent, 0, 100)
        if self.os == "Windows":
            min_v, max_v = self.vol_range[0], self.vol_range[1]
            vol = np.interp(percent, [0, 100], [min_v, max_v])
            self.volume.SetMasterVolumeLevel(vol, None)
        elif self.os == "Darwin":
            import subprocess
            subprocess.call(["osascript", "-e",
                             f"set volume output volume {int(percent)}"])
        elif self.os == "Linux":
            import subprocess
            subprocess.call(["amixer", "-D", "pulse", "sset",
                             "Master", f"{int(percent)}%"])
```

### Main Loop (`main.py`)

```python
import cv2, math, numpy as np
from src.hand_detector import HandDetector
from src.volume_controller import VolumeController
from config import MIN_DISTANCE, MAX_DISTANCE, SMOOTHING_WINDOW

detector   = HandDetector()
controller = VolumeController()
cap        = cv2.VideoCapture(0)
vol_history = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame     = detector.find_hands(frame)
    positions = detector.get_landmark_positions(frame)

    if positions:
        x4, y4 = positions[4]   # Thumb tip
        x8, y8 = positions[8]   # Index tip

        distance   = math.hypot(x8 - x4, y8 - y4)
        volume_pct = np.interp(distance,
                               [MIN_DISTANCE, MAX_DISTANCE], [0, 100])

        vol_history.append(volume_pct)
        if len(vol_history) > SMOOTHING_WINDOW:
            vol_history.pop(0)
        smoothed = sum(vol_history) / len(vol_history)

        controller.set_volume(smoothed)

        cv2.circle(frame, (x4, y4), 10, (0, 255, 0), cv2.FILLED)
        cv2.circle(frame, (x8, y8), 10, (255, 0, 0), cv2.FILLED)
        cv2.line(frame, (x4, y4), (x8, y8), (200, 200, 200), 2)
        cv2.putText(frame, f'Volume: {int(smoothed)}%',
                    (40, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Hand Gesture Volume Control", frame)
    if cv2.waitKey(1) & 0xFF in [ord('q'), 27]:
        break

cap.release()
cv2.destroyAllWindows()
```

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Test individual modules
pytest tests/test_detector.py -v
pytest tests/test_volume.py -v
```

---

## 🗺️ Roadmap

- [x] Pinch-to-volume (thumb + index finger)
- [x] Cross-platform volume control
- [x] Smooth interpolation & live HUD overlay
- [ ] Two-hand gesture support
- [ ] Fist gesture = play/pause media
- [ ] Swipe gesture for screen brightness control
- [ ] System tray icon + background mode
- [ ] GUI settings panel (PyQt6)
- [ ] Custom gesture mapping via config

---

## 🐛 Troubleshooting

| Problem | Fix |
|---|---|
| `No module named 'pycaw'` | `pip install pycaw comtypes` (Windows only) |
| Camera not opening | Try `--cam 1` or `--cam 2` |
| Volume not changing on macOS | Grant Terminal camera + accessibility access in System Settings |
| Hand not detected | Improve lighting; keep hand 30–60 cm from webcam |
| Low FPS | Lower `FRAME_WIDTH` / `FRAME_HEIGHT` in `config.py` |
| Jumpy / erratic volume | Increase `SMOOTHING_WINDOW` in `config.py` |

---

## 🤝 Contributing

1. Fork the repo
2. Create your branch: `git checkout -b feature/swipe-brightness`
3. Commit your changes: `git commit -m 'Add swipe gesture for brightness'`
4. Push: `git push origin feature/swipe-brightness`
5. Open a Pull Request

Please lint and test before submitting:

```bash
flake8 src/ --max-line-length=100
pytest tests/ -v
```

---

## 📄 License

Distributed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 📚 References

- [MediaPipe Hands](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) — Google's 21-point hand landmark model
- [OpenCV Documentation](https://docs.opencv.org/)
- [pycaw — Python Core Audio Windows Library](https://github.com/AndreMiras/pycaw)
- [NumPy interp](https://numpy.org/doc/stable/reference/generated/numpy.interp.html) — linear interpolation used for distance → volume mapping

---

<div align="center">

Built with 🖐️ and Python

**⭐ Star this repo if it saved you from reaching for the volume keys!**

</div>
