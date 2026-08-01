"""Tunables for Help Me ERINNNNNN!! fist pump counter.

All thresholds are scale-relative (fraction of torso length) so distance
from camera doesn't change behaviour.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Difficulty(Enum):
    """Open Mode timer presets."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    HELL = "hell"


DIFFICULTY_TIMERS: dict[Difficulty, float] = {
    Difficulty.EASY: 60.0,
    Difficulty.MEDIUM: 45.0,
    Difficulty.HARD: 30.0,
    Difficulty.HELL: 15.0,
}


class Mode(Enum):
    """Game modes."""

    OPEN = "open"
    COMPE = "compe"


@dataclass
class PumpConfig:
    """Detection tunables for the fist-pump gesture.

    The gesture is: wrist rises above head height (up), then drops back
    below shoulder (down). One full cycle = one pump. Each arm tracked
    independently.
    """

    # Wrist must be this far above shoulder (as fraction of torso length)
    # to count as "raised". Separate enter/release for hysteresis.
    raise_enter: float = 0.35
    raise_release: float = 0.20

    # Smoothing on wrist height signal (EMA alpha). Lower = steadier.
    height_alpha: float = 0.50

    # Minimum time between pumps on the same arm (debounce).
    min_pump_seconds: float = 0.25

    # Landmarks below this visibility are treated as missing.
    min_visibility: float = 0.5

    # Shoulder-width to torso-length approximation (when hips not visible).
    shoulder_to_torso: float = 1.2

    # Frame margin for landmark visibility.
    frame_margin: float = 0.02

    # How long pose must be visible before game starts.
    prepare_seconds: float = 1.5

    # Grace period for lost tracking before resetting to prepare.
    lost_grace_seconds: float = 0.8


@dataclass
class CompeConfig:
    """Compe Mode (rhythm game) settings."""

    # How far from the beat timestamp a pump can land and still be valid (seconds).
    hit_window: float = 0.3

    # How long after a beat passes with no pump before it counts as a miss.
    miss_window: float = 0.5

    # Path to the beatmap JSON file.
    beatmap_path: str = ""

    # Path to the audio file.
    song_path: str = ""


@dataclass
class GameConfig:
    """Top-level game configuration."""

    mode: Mode = Mode.OPEN
    difficulty: Difficulty = Difficulty.MEDIUM
    pump: PumpConfig = field(default_factory=PumpConfig)
    compe: CompeConfig = field(default_factory=CompeConfig)

    # Display settings.
    fullscreen: bool = True
    camera_index: int = 0
    mirror: bool = True

    # Web settings.
    web_host: str = "127.0.0.1"
    web_port: int = 5050

    @property
    def timer_seconds(self) -> float:
        """Get the timer duration for the current difficulty."""
        return DIFFICULTY_TIMERS[self.difficulty]
