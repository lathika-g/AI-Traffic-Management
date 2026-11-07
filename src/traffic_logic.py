"""Traffic density classification and green-time calculation."""
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from src import config


@dataclass
class TrafficPlan:
    direction: str
    vehicle_count: float
    density_percent: float
    level: str            # Low / Medium / High / No data
    initial_green: int    # always 50
    green_time: int       # dynamically calculated


def density_percent(count: float) -> float:
    """Density as a percentage of the configured capacity (max 100)."""
    return round(min(100.0, count / config.CAPACITY_COUNT * 100), 1)


def density_level(count: float) -> str:
    if count < config.LOW_DENSITY_COUNT:
        return "Low"
    if count > config.HIGH_DENSITY_COUNT:
        return "High"
    return "Medium"


def calculate_green_time(count: float) -> int:
    """Convert a vehicle count into a green time in seconds.

    Low    -> scales from MIN_GREEN (count 0) up towards INITIAL_GREEN
    Medium -> exactly INITIAL_GREEN
    High   -> scales from INITIAL_GREEN up to MAX_GREEN
    """
    base = config.INITIAL_GREEN
    if count < config.LOW_DENSITY_COUNT:
        ratio = count / config.LOW_DENSITY_COUNT
        green = config.MIN_GREEN + (base - config.MIN_GREEN) * ratio
    elif count > config.HIGH_DENSITY_COUNT:
        span = config.CAPACITY_COUNT - config.HIGH_DENSITY_COUNT
        ratio = min(1.0, (count - config.HIGH_DENSITY_COUNT) / span)
        green = base + (config.MAX_GREEN - base) * ratio
    else:
        green = base
    return int(round(max(config.MIN_GREEN, min(config.MAX_GREEN, green))))


def initial_plan() -> dict:
    """Plan BEFORE detection: every direction gets exactly 50 seconds."""
    return {
        d: TrafficPlan(d, 0.0, 0.0, "No data",
                       config.INITIAL_GREEN, config.INITIAL_GREEN)
        for d in config.DIRECTIONS
    }


def build_plan(counts: dict) -> dict:
    """Build a plan for all four directions from detected counts.

    `counts` maps direction -> average vehicle count. A direction missing from
    `counts` (e.g. its video failed) keeps the initial 50 s.
    """
    plans = initial_plan()
    for direction, count in counts.items():
        plans[direction] = TrafficPlan(
            direction=direction,
            vehicle_count=count,
            density_percent=density_percent(count),
            level=density_level(count),
            initial_green=config.INITIAL_GREEN,
            green_time=calculate_green_time(count),
        )
    return plans


def save_summary(plans: dict, path: Path = None) -> Path:
    """Write the plan to results/summary.json."""
    path = path or (config.RESULTS_DIR / "summary.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({d: asdict(p) for d, p in plans.items()}, f, indent=2)
    return path
