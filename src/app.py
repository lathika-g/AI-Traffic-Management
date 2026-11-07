"""Streamlit dashboard for the AI-Based Traffic Management System."""
import sys
import time
from pathlib import Path

# Make the project root importable when launched with `streamlit run src/app.py`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import streamlit as st

from src import config
from src.signal_controller import SignalController
from src.traffic_logic import build_plan, initial_plan, save_summary
from src.video_processor import analyze_all

st.set_page_config(page_title="AI-Based Traffic Management", layout="wide")


@st.cache_resource(show_spinner="Loading YOLO model...")
def load_detector():
    from src.detector import VehicleDetector
    return VehicleDetector()


def init_state():
    if "initialised" in st.session_state:
        return
    plans = initial_plan()  # 50 s for every direction
    st.session_state.plans = plans
    st.session_state.results = {}
    st.session_state.errors = {}
    st.session_state.controller = SignalController(
        {d: p.green_time for d, p in plans.items()}
    )
    st.session_state.initialised = True


def run_analysis(num_samples: int):
    ctrl = st.session_state.controller
    was_running = ctrl.is_running()
    ctrl.pause()  # don't let the clock run while YOLO is working
    try:
        try:
            detector = load_detector()
        except RuntimeError as exc:
            st.error(str(exc))
            return
        with st.spinner("Running YOLO on all four videos..."):
            results, errors = analyze_all(detector, num_samples)
        counts = {d: r.avg_count for d, r in results.items()}
        plans = build_plan(counts)
        st.session_state.results = results
        st.session_state.errors = errors
        st.session_state.plans = plans
        ctrl.update_green_times({d: p.green_time for d, p in plans.items()})
        save_summary(plans)
        if results:
            st.success(f"Analysed {len(results)} of 4 videos. Summary saved to results/summary.json")
    finally:
        if was_running:
            ctrl.resume()


def render_signal_panel(ctrl: SignalController):
    st.subheader("🚦 Signal Status")
    if ctrl.phase_start is None:
        st.info("Signal not started. Press **Start / Resume** in the sidebar. "
                "Every direction begins with 50 s of green.")
        return None
    s = ctrl.status()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active green", s["active"])
    c2.metric("Green duration", f"{s['duration']} s")
    c3.metric("Countdown", f"{s['remaining']} s")
    c4.metric("Next direction", s["next"])
    st.progress(min(1.0, max(0.0, 1 - s["remaining"] / s["duration"])))
    lights = "  ".join(f"{'🟢' if d == s['active'] else '🔴'} {d}" for d in config.DIRECTIONS)
    st.markdown(f"### {lights}")
    st.caption("Order: South → West → North → East → South ...   "
               f"Completed cycles: {s['cycles']}" + ("   |   ⏸ Paused" if ctrl.paused_at else ""))
    return s["active"]


def render_direction_cards(active):
    plans = st.session_state.plans
    results = st.session_state.results
    errors = st.session_state.errors
    st.subheader("📹 Directions")
    for col, d in zip(st.columns(4), config.DIRECTIONS):
        plan = plans[d]
        with col:
            st.markdown(f"#### {'🟢' if d == active else '🔴'} {d}")
            if d in results:
                rgb = cv2.cvtColor(results[d].annotated_frame, cv2.COLOR_BGR2RGB)
                st.image(rgb, use_container_width=True)
            elif d in errors:
                st.error(errors[d])
            else:
                st.info("Not analysed yet")
            st.write(f"**Vehicles (avg/frame):** {plan.vehicle_count}")
            st.write(f"**Density:** {plan.density_percent}% ({plan.level})")
            st.write(f"**Initial green:** {plan.initial_green} s")
            diff = plan.green_time - plan.initial_green
            st.write(f"**Dynamic green:** {plan.green_time} s ({diff:+d} s)")
            if d in results:
                breakdown = ", ".join(f"{k}: {v}" for k, v in results[d].class_counts.items())
                st.caption(breakdown)


def main():
    init_state()
    ctrl: SignalController = st.session_state.controller

    st.title("🚦 AI-Based Traffic Management System")
    st.caption("YOLO vehicle detection + dynamic green timing with clockwise rotation")

    with st.sidebar:
        st.header("Controls")
        num_samples = st.slider("Frames sampled per video", 5, 40, config.NUM_SAMPLE_FRAMES)
        analyse = st.button("🔍 Analyse videos with YOLO", type="primary")
        c1, c2 = st.columns(2)
        start = c1.button("▶ Start / Resume")
        pause = c2.button("⏸ Pause")
        reset = st.button("↺ Reset signal")
        st.markdown("---")
        st.write(f"Initial green: **{config.INITIAL_GREEN} s** (all directions)")
        st.write(f"Allowed range: {config.MIN_GREEN}–{config.MAX_GREEN} s")

    if analyse:
        run_analysis(num_samples)
    if start:
        ctrl.start() if ctrl.phase_start is None else ctrl.resume()
    if pause:
        ctrl.pause()
    if reset:
        ctrl.reset()

    active = render_signal_panel(ctrl)
    render_direction_cards(active)

    st.subheader("📊 Timing Summary")
    st.table([
        {"Direction": p.direction, "Vehicles": p.vehicle_count,
         "Density %": p.density_percent, "Level": p.level,
         "Initial green (s)": p.initial_green, "Dynamic green (s)": p.green_time}
        for p in st.session_state.plans.values()
    ])

    if ctrl.is_running():  # refresh every second for the countdown
        time.sleep(1)
        st.rerun()


main()
