# 🎯 OpenCV Face Detection

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

A real-time face detection application built with **Python** and **OpenCV**, capable of detecting faces from a webcam feed, images, or video files using Haar Cascade Classifiers and/or deep learning-based DNN models.

---

## 📸 Demo

```
[Webcam] ──► [OpenCV Processing] ──► [Detected Faces with Bounding Boxes]
```

> Detects faces in real-time with bounding boxes and optional confidence scores.

---

## 🚀 Features

- ✅ Real-time face detection via webcam
- ✅ Face detection on static images
- ✅ Face detection on video files
- ✅ Haar Cascade Classifier support
- ✅ DNN-based face detection (more accurate)
- ✅ Adjustable detection parameters (scale factor, min neighbors, confidence threshold)
- ✅ Face count display on screen
- ✅ Save output images/videos with detections

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| Python 3.8+ | Core language |
| OpenCV (`cv2`) | Image processing & face detection |
| NumPy | Array manipulation |
| imutils | Image utility helpers |

---

## 📁 Project Structure

```
opencv-face-detection/
│
├── models/
│   ├── haarcascade_frontalface_default.xml   # Haar Cascade model
│   ├── deploy.prototxt                        # DNN model architecture
│   └── res10_300x300_ssd_iter_140000.caffemodel  # DNN weights
│
├── images/
│   └── sample.jpg                            # Sample test image
│
├── videos/
│   └── sample.mp4                            # Sample test video
│
├── output/
│   └── ...                                   # Saved detection results
│
├── detect_faces_image.py                     # Detect faces in an image
├── detect_faces_video.py                     # Detect faces in a video file
├── detect_faces_webcam.py                    # Real-time webcam detection
├── detect_faces_dnn.py                       # DNN-based face detection
├── requirements.txt                          # Python dependencies
├── LICENSE
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/opencv-face-detection.git
cd opencv-face-detection
```

### 2. Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate       # On macOS/Linux
venv\Scripts\activate          # On Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

**`requirements.txt`:**
```
opencv-python>=4.5.0
numpy>=1.21.0
imutils>=0.5.4
```

### 4. Download Pre-trained Models

Haar Cascade (included with OpenCV):
```bash
# Already bundled in OpenCV installation
# Located at: cv2.data.haarcascades
```

DNN model (optional, for higher accuracy):
```bash
# Download the Caffe model
wget https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt -P models/
wget https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel -P models/
```

---

## 🖥️ Usage

### Detect Faces in an Image

```bash
python detect_faces_image.py --image images/sample.jpg
```

### Detect Faces in a Video File

```bash
python detect_faces_video.py --video videos/sample.mp4
```

### Real-Time Webcam Detection

```bash
python detect_faces_webcam.py
```

### DNN-Based Detection (Higher Accuracy)

```bash
python detect_faces_dnn.py --image images/sample.jpg --confidence 0.5
```

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--image` | — | Path to input image |
| `--video` | — | Path to input video |
| `--confidence` | `0.5` | Minimum confidence threshold (DNN only) |
| `--scale` | `1.1` | Scale factor for Haar Cascade |
| `--min-neighbors` | `5` | Min neighbors for Haar Cascade |
| `--output` | `output/` | Directory to save results |

---

## 🧠 How It Works

### Method 1 — Haar Cascade Classifier

```python
import cv2

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

img = cv2.imread('images/sample.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

for (x, y, w, h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Face Detection', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Method 2 — DNN-Based Detection

```python
import cv2
import numpy as np

net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/res10_300x300_ssd_iter_140000.caffemodel')

image = cv2.imread('images/sample.jpg')
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

net.setInput(blob)
detections = net.forward()

for i in range(detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 0, 255), 2)
```

---

## 📊 Comparison: Haar vs DNN

| Feature | Haar Cascade | DNN (SSD) |
|--------|-------------|-----------|
| Speed | ⚡ Faster | 🐢 Slower |
| Accuracy | 🟡 Moderate | 🟢 Higher |
| Profile detection | ❌ Limited | ✅ Better |
| Occluded faces | ❌ Poor | ✅ Better |
| Dependency size | Small | Larger |
| Real-time use | ✅ Great | ✅ Good (with GPU) |

---

## 🔧 Configuration Tips

- Increase `--confidence` threshold to reduce false positives
- Decrease `scaleFactor` (e.g. `1.05`) for more detections but slower speed
- Increase `minNeighbors` to reduce false detections at the cost of missing some faces
- Use DNN method for profile faces or partially occluded faces

---

## 📌 Roadmap

- [ ] Add face landmark detection
- [ ] Add face recognition (identify known faces)
- [ ] Add emotion detection
- [ ] Add age & gender estimation
- [ ] Deploy as a web app using Flask/FastAPI
- [ ] Add GPU (CUDA) support

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork this repository
2. Create a new branch: `git checkout -b feature/your-feature`
3. Make your changes and commit: `git commit -m "Add your feature"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a Pull Request

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on the code of conduct.

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- [OpenCV](https://opencv.org/) — Open Source Computer Vision Library
- [Viola-Jones Algorithm](https://en.wikipedia.org/wiki/Viola%E2%80%93Jones_object_detection_framework) — Haar Cascade methodology
- [OpenCV DNN Face Detector](https://github.com/opencv/opencv/tree/master/samples/dnn) — Pre-trained DNN models

---

## 📬 Contact

**Your Name** — [@your-twitter](https://twitter.com/your-twitter) — your.email@example.com

Project Link: [https://github.com/your-username/opencv-face-detection](https://github.com/your-username/opencv-face-detection)
