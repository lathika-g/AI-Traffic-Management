"""Entry point for the AI-Based Traffic Management System.

Usage:
    python main.py            -> launch the Streamlit dashboard
    python main.py --cli      -> run detection in the terminal and print timings
"""
import argparse
import subprocess
import sys
from pathlib import Path

from src import config

APP_PATH = Path(__file__).resolve().parent / "src" / "app.py"


def run_ui():
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", str(APP_PATH)], check=True)
    except FileNotFoundError:
        print("Python executable not found.")
    except subprocess.CalledProcessError as exc:
        print(f"Streamlit exited with an error ({exc.returncode}). "
              "Did you run: pip install -r requirements.txt ?")
    except KeyboardInterrupt:
        print("\nStopped.")


def run_cli(num_samples: int):
    from src.detector import VehicleDetector
    from src.traffic_logic import build_plan, save_summary
    from src.video_processor import analyze_all

    try:
        detector = VehicleDetector()
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return

    print(f"Initial green for every direction: {config.INITIAL_GREEN} s\n")
    print("Analysing videos (this can take a minute)...")
    results, errors = analyze_all(detector, num_samples)

    for direction, message in errors.items():
        print(f"[WARNING] {direction}: {message}")

    plans = build_plan({d: r.avg_count for d, r in results.items()})
    print(f"\n{'Direction':<10}{'Vehicles':<10}{'Density%':<10}{'Level':<10}{'Initial':<9}{'Dynamic'}")
    for d in config.DIRECTIONS:
        p = plans[d]
        print(f"{d:<10}{p.vehicle_count:<10}{p.density_percent:<10}{p.level:<10}"
              f"{p.initial_green:<9}{p.green_time} s")

    total = sum(p.green_time for p in plans.values())
    print("\nSignal order:", " -> ".join(config.DIRECTIONS), "-> South ...")
    print(f"Total cycle time: {total} s")
    print(f"Summary saved to: {save_summary(plans)}")


def main():
    parser = argparse.ArgumentParser(description="AI-Based Traffic Management System")
    parser.add_argument("--cli", action="store_true", help="run in the terminal without the UI")
    parser.add_argument("--samples", type=int, default=config.NUM_SAMPLE_FRAMES,
                        help="frames sampled per video (CLI mode)")
    args = parser.parse_args()
    run_cli(args.samples) if args.cli else run_ui()


if __name__ == "__main__":
    main()
