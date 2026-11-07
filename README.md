# AI-Based Traffic Management System

A college mini project that uses **YOLO** and **OpenCV** to detect vehicles in four
traffic videos (South, West, North, East) and dynamically adjusts green signal
timings based on traffic density.

## Features
- Detects cars, motorcycles, buses and trucks using YOLOv8
- Processes four direction videos separately
- Computes vehicle count and traffic density per direction
- All directions start with an equal **50-second** green time
- Green time is then recalculated: high density > 50 s, low density < 50 s, medium = 50 s
- Fixed clockwise rotation: South → West → North → East → South
- Streamlit dashboard with detection output, countdown timer and next direction

## Tech Stack
Python 3.10 · Ultralytics YOLOv8 · OpenCV · NumPy · Streamlit

## Project Structure
```
AI_Based_Traffic_Management/
├── main.py                  # entry point (UI or --cli)
├── requirements.txt
├── videos/                  # south.mp4, west.mp4, north.mp4, east.mp4
├── results/                 # annotated images + summary.json
├── screenshots/
└── src/
    ├── config.py            # settings and thresholds
    ├── detector.py          # YOLO vehicle detector
    ├── video_processor.py   # OpenCV video reading, per-direction counts
    ├── traffic_logic.py     # density and green-time calculation
    ├── signal_controller.py # clockwise rotation and countdown
    └── app.py               # Streamlit dashboard
```

## Installation
```bash
python3.10 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage
1. Place `south.mp4`, `west.mp4`, `north.mp4`, `east.mp4` in `videos/`
   (free clips: Pexels / Pixabay, search "traffic intersection").
2. Run `python main.py` (dashboard) or `python main.py --cli` (terminal).
3. In the dashboard click **Start**, then **Analyse videos with YOLO**.

The first run downloads `yolov8n.pt`, so internet is required once.

## How It Works
1. Every direction begins with 50 s green.
2. Evenly spaced frames are sampled from each video and passed to YOLO.
3. Vehicle count = average vehicles visible per sampled frame.
4. Density % = count / capacity (25 vehicles = 100%).
5. Green time: Low (< 5) → 30–50 s, Medium (5–15) → 50 s, High (> 15) → 50–70 s.
6. The controller rotates clockwise; each direction always gets a turn.

All thresholds are editable in `src/config.py`.

## Limitations
- Count is an average per frame, not unique vehicles (no object tracking).
- Thresholds must be tuned to the camera view of your videos.
- The dashboard shows an annotated snapshot per direction, not live video playback.
- No yellow/all-red phase and no real hardware integration.

## Future Work
Object tracking for unique counts, live video streaming, yellow phase,
emergency vehicle priority, Arduino/Raspberry Pi integration.

## Screenshots
Add `screenshots/dashboard.png` after running the app.

## Author
Your Name · Your College · Year
