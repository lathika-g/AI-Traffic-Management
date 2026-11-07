"""Central configuration for the AI-Based Traffic Management System."""
from pathlib import Path

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent.parent
VIDEOS_DIR = BASE_DIR / "videos"
RESULTS_DIR = BASE_DIR / "results"
SUPPORTED_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

# ---------- Directions (clockwise signal order) ----------
DIRECTIONS = ["South", "West", "North", "East"]

# ---------- YOLO ----------
MODEL_NAME = "yolov8n.pt"      # smallest/fastest; auto-downloads on first run
CONFIDENCE = 0.35              # minimum detection confidence
# COCO class IDs -> names
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

# ---------- Video sampling ----------
NUM_SAMPLE_FRAMES = 15         # frames analysed per video
MAX_FRAME_WIDTH = 960          # frames wider than this are resized (speed)

# ---------- Signal timing (seconds) ----------
INITIAL_GREEN = 50             # SAME for all four directions before detection
MIN_GREEN = 30                 # lowest possible green time
MAX_GREEN = 70                 # highest possible green time

# ---------- Density thresholds (average vehicles per frame) ----------
LOW_DENSITY_COUNT = 5          # below this  -> Low
HIGH_DENSITY_COUNT = 15        # above this  -> High
CAPACITY_COUNT = 25            # count that equals 100% density / max green
