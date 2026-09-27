# Smart Campus Classroom Monitoring

A real-time computer vision application for classroom monitoring using **YOLO11, ByteTrack, OpenCV, and Streamlit**.

The application processes video sequentially as it arrives and displays live classroom analytics through a Streamlit dashboard.

## Features

The system provides:

- **Entry / Exit Detection** – detects people crossing a configured classroom entrance boundary.
- **Motion Detection** – reports `Motion detected` or `No motion`.
- **Currently Present** – continuously estimates the number of people currently visible/present.
- **Unique Entries** – maintains a cumulative count of observed entry events.
- **Seated vs Standing/Moving** – classifies tracked people using movement and bounding-box geometry.
- **Frame Quality** – reports `Clear` or `Blurry - flag for review`.
- **Room Status** – reports `Occupied` or `Empty`.

Tracked people are displayed with IDs and bounding boxes:

```text
ID 4 - Seated
ID 7 - Standing/Moving
```

Green boxes represent seated people and orange boxes represent standing/moving people.

---

## Technologies Used

- **Python**
- **YOLO11s (Ultralytics)** – pretrained person detection
- **ByteTrack** – multi-object tracking
- **Ultralytics ObjectCounter** – entrance/exit crossing
- **OpenCV** – video processing, motion and blur detection
- **Streamlit** – live dashboard
- **streamlit-webrtc** – browser webcam input

No model is trained from scratch.

---

## Video Sources

The application supports:

1. **Project Video** – process included classroom footage.
2. **Upload Video** – upload a new video through the UI.
3. **Webcam** – process a live browser webcam stream.

Video is processed **frame by frame**, rather than loading and analysing the complete video in advance.

The application targets approximately **8 FPS**, reducing inference cost while maintaining useful real-time analytics.

---

## Processing Pipeline

```text
Video / Webcam
      |
      v
Frame Sampling
      |
      v
YOLO11 Person Detection
      |
      v
ByteTrack Tracking
      |
      +---------------------+
      |                     |
      v                     v
Entry / Exit          Current Attendance
      |                     |
      +----------+----------+
                 |
                 v
       Seated / Standing-Moving
                 |
        +--------+--------+
        |                 |
        v                 v
 Motion Detection    Frame Quality
        |                 |
        +--------+--------+
                 |
                 v
        Streamlit Dashboard
```

---

## Installation

### 1. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

Streamlit will normally open the application at:

```text
http://localhost:8501
```

Select a video source from the sidebar and start monitoring.

For webcam input, allow camera permission when requested by the browser.

---

## Main Configuration

```python
MODEL = "yolo11s.pt"
CONFIDENCE = 0.15
IMAGE_SIZE = 640
IOU = 0.5
TARGET_FPS = 8
```

`640` is used as the inference image size to provide a practical balance between detection quality and processing speed.

---

## Entry / Exit Configuration

For fixed CCTV cameras, the entrance boundary is configured using normalized coordinates.

Example:

```python
"classroom_entry.mp4": {
    "door_line": [
        (0.18, 0.08),
        (0.18, 0.72)
    ],
    "reverse_direction": True
}
```

This allows each camera's entrance location and crossing direction to be configured independently.

---

## Limitations

- Heavy occlusion can cause some people to be temporarily missed.
- Long occlusions may cause tracking IDs to change.
- Unique entries represent **observed entry events**, not biometric identity recognition.
- Entry/exit boundaries need to be configured for each fixed camera.
- Seated/standing classification is a lightweight heuristic and can be affected by camera angle or occlusion.
- Motion detection assumes a mostly fixed CCTV camera.

---

## Scaling to Multiple Cameras

For a large deployment, video ingestion, inference, tracking, event processing, and storage can be separated into independent services.

For hundreds of cameras, inference can be distributed across multiple GPU workers or edge devices. Processing selected frames rather than every frame reduces GPU and network requirements.

Major bottlenecks include:

- GPU inference
- Video decoding
- Network bandwidth
- Concurrent camera streams

---

## Avoiding Double Counting

ByteTrack maintains person tracks across frames. For production deployment, counting can be made more robust using:

- Entrance zones and complete-crossing rules
- Hysteresis/debounce
- Track persistence
- Appearance-based Re-Identification for longer occlusions

These methods help reduce duplicate counts when people remain near an entrance or temporarily disappear behind others.

---

## Handling Blurry Feeds

Frame quality is continuously monitored using Laplacian variance.

For production use, sustained poor quality can:

- Flag analytics as lower confidence
- Trigger a camera-health warning
- Request camera inspection or maintenance

---

## Project Structure

```text
smart-campus-monitor/
│
├── app.py
├── README.md
├── requirements.txt
├── yolo11s.pt
├── classroom_entry.mp4
└── video7.mp4
```

---

## Summary

This prototype demonstrates a real-time classroom monitoring pipeline using pretrained computer vision models. It combines person detection, tracking, entrance/exit counting, occupancy estimation, activity classification, motion detection, frame-quality monitoring, and a Streamlit dashboard without requiring custom model training.

