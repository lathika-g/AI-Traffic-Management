"""Clockwise traffic signal controller: South -> West -> North -> East."""
import math
import time

from src import config


class SignalController:
    """Tracks which direction is green and how many seconds remain."""

    def __init__(self, green_times: dict, order=None):
        self.order = list(order or config.DIRECTIONS)
        self.green_times = dict(green_times)
        self.index = 0
        self.phase_start = None   # None = not started
        self.paused_at = None
        self.cycles = 0

    # ----- control -----
    def start(self):
        self.phase_start = time.time()
        self.paused_at = None

    def pause(self):
        if self.phase_start is not None and self.paused_at is None:
            self.paused_at = time.time()

    def resume(self):
        if self.paused_at is not None:
            self.phase_start += time.time() - self.paused_at
            self.paused_at = None

    def reset(self):
        self.index = 0
        self.phase_start = None
        self.paused_at = None
        self.cycles = 0

    def is_running(self) -> bool:
        return self.phase_start is not None and self.paused_at is None

    def update_green_times(self, green_times: dict):
        self.green_times = dict(green_times)

    # ----- state -----
    def status(self) -> dict:
        """Advance the signal if needed and return the current state."""
        if self.phase_start is None:
            self.start()
        now = self.paused_at if self.paused_at is not None else time.time()

        # Move to the next direction(s) when time is up
        while now - self.phase_start >= self.green_times[self.order[self.index]]:
            self.phase_start += self.green_times[self.order[self.index]]
            self.index = (self.index + 1) % len(self.order)
            if self.index == 0:
                self.cycles += 1

        active = self.order[self.index]
        duration = self.green_times[active]
        remaining = max(0, math.ceil(duration - (now - self.phase_start)))
        return {
            "active": active,
            "next": self.order[(self.index + 1) % len(self.order)],
            "duration": duration,
            "remaining": remaining,
            "cycles": self.cycles,
        }
