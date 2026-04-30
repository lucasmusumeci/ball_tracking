# 🟡 Ball Tracking with OpenCV

Real-time ball detection and tracking using color segmentation and the Hough Circle Transform, implemented in Python with OpenCV.

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![NumPy](https://img.shields.io/badge/NumPy-latest-orange?logo=numpy)

---

## 📋 Overview

This project was developed as part of a **Perception** lab at Polytech Montpellier. The goal was to explore OpenCV's image processing capabilities by building a system that can detect and track a ball in real time from a video feed, using only color-based segmentation.

The user simply clicks on the ball in the first frame to set a color reference. From there, the algorithm automatically detects and tracks the ball across every subsequent frame.

---

## ✨ Features

- **Interactive color selection** — click on the ball in the first frame to initialize tracking
- **HSV color segmentation** — more robust than RGB for handling lighting variations
- **Morphological filtering** — closing operation to reduce noise before contour extraction
- **Hough Circle Transform** — accurate circle detection on the gradient image
- **Dynamic color update** — the reference color is recalibrated each frame from the detected ball's center, adapting to lighting changes
- **Search area reduction** — processing is restricted to a region of interest (ROI) around the last known position, significantly speeding up detection
- **Frame skipping with catch-up** — the playback loop compensates for processing delays to maintain smooth, real-time video speed

---

## 🛠️ How It Works

### 1. Color Reference Extraction
The user clicks on the ball. A square ROI of side `2R` centered on the click is extracted in HSV space, and the **median** of each H, S, V channel is computed to obtain a robust color reference — resistant to outliers and specular reflections.

### 2. Color Mask (InRange)
Lower and upper HSV bounds are computed using three tolerances `ε₁` (hue), `ε₂` (saturation), `ε₃` (value). `cv2.inRange()` produces a binary mask. The value tolerance `ε₃ = 125` is key: it excludes pixels that are significantly darker or brighter than the reference, eliminating a large amount of background noise.

### 3. Morphological Operations
A **closing** (dilation → erosion) is applied to fill small holes inside the ball's mask. Notably, an opening (erosion first) was tested but found to **reduce accuracy** by eroding thin arc segments on the ball's boundary that are critical for circle fitting — especially when the search area is already small and noise is minimal.

### 4. Gradient & Hough Transform
A morphological gradient extracts the contours of the mask. `cv2.HoughCircles()` is then applied to detect circles. Only the top-voted circle is retained.

### 5. Search Area Reduction
Once the ball is found in frame `n`, the next frame's processing is restricted to a cropped region around the detected center, with a margin proportional to the detected radius. This reduces computation from O(n²) to O((n/k)²), where `k > 1` — up to **4× speedup** when the search window halves the image dimensions. The offset is stored to remap detected coordinates back to the full image space.

### 6. Color Update
After each detection, the reference color is updated using the median HSV values from a small ROI centered on the detected ball. If no circle is found, the original color is restored to prevent tracking drift.

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install opencv-python numpy
```

### Usage

1. Clone the repository:
```bash
git clone git@github.com:lucasmusumeci/ball_tracking.git
cd ball_tracking
```

2. Update the video path in `TP_Video.py`:
```python
cap = cv.VideoCapture("path/to/your/video.mp4")
```

3. Run the script:
```bash
python TP_Video.py
```

4. **Click on the ball** in the displayed frame to start tracking. Press `q` to quit.

---

## ⚙️ Parameters

| Parameter   | Default | Description |
|-------------|---------|-------------|
| `fps`       | `15`    | Target processing frame rate |
| `epsilon`   | `11`    | Hue tolerance (H channel) |
| `epsilon2`  | `110`   | Saturation tolerance (S channel) |
| `epsilon3`  | `125`   | Value tolerance (V channel) — excludes very dark/bright areas |
| `margin`    | `0.5`   | ROI expansion factor around the last detected radius |
| `R`         | `50`    | Half-size of the click ROI for color sampling |

---

## 📊 Results

The tracker reliably follows the ball across the full video, handling:
- Changes in lighting as the ball moves across the surface
- Partial occlusions (shadow from the table edge)
- Ball deformation / perspective changes

Detection runs smoothly at 15 fps on a standard laptop, with the ROI reduction being the main contributor to real-time performance.

---

## 📁 Project Structure

```
ball_tracking/
├── TP_Video.py       # Main tracking script
└── README.md
```

---

## 📄 Report

A detailed technical report (in French) is available in the repository, covering each processing step with visual comparisons and parameter justifications.

---

## 👤 Author

**Lucas Musumeci** — Polytech Montpellier, MEA  
*November 2025*
