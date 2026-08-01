"""Help Me ERINNNNNN!! — pure pump detection and game logic.

No camera, no window, no audio here. Takes normalized pose landmarks and a
clock, returns game state. Desktop and web are thin adapters over this.

Detection:
    Each arm is tracked independently. A "pump" is one full cycle:
    wrist rises above head → wrist drops back below shoulder.
    Count increments on the DOWN transition (the actual pump motion).

    Height is measured as (shoulder_y - wrist_y) / torso_length.
    Positive = wrist above shoulder. Uses hysteresis to avoid flicker.

Open Mode:
    Timer counts down. Each pump scores 1 point per arm. Both arms
    pumping independently means up to 2 points per beat.

Compe Mode:
    Song plays, beatmap marks where pumps should happen. Pump within
    ±hit_window of a beat = safe. Pump outside any window = DEAD.
    Beat passes with no pump within miss_window = DEAD.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import (
    CompeConfig,
    Difficulty,
    GameConfig,
    Mode,
    PumpConfig,
)


# --- Arm pump tracker (one per arm) ---

PHASE_DOWN = "down"
PHASE_UP = "up"


@dataclass
class ArmState:
    """Tracks one arm's pump state."""

    phase: str = PHASE_DOWN
    height: float = 0.0
    smoothed_height: float = 0.0
    last_pump_at: float | None = None
    pumps: int = 0


class ArmTracker:
    """Detects fist-pump cycles for a single arm.

    State machine:
        DOWN → UP:   wrist rises above raise_enter threshold
        UP → DOWN:   wrist drops below raise_release threshold → COUNT!
    """

    def __init__(self, side: str, config: PumpConfig):
        self.side = side
        self.config = config
        self.state = ArmState()
        self._alpha = config.height_alpha

    def update(self, raw_height: float | None, now: float) -> bool:
        """Feed a new height reading. Returns True if a pump just completed."""
        if raw_height is None:
            return False

        # EMA smooth
        if self.state.smoothed_height == 0.0:
            self.state.smoothed_height = raw_height
        else:
            self.state.smoothed_height += self._alpha * (
                raw_height - self.state.smoothed_height
            )

        self.state.height = self.state.smoothed_height
        pumped = False

        if self.state.phase == PHASE_DOWN:
            # Arm is down — check if it rose above enter threshold
            if self.state.height >= self.config.raise_enter:
                self.state.phase = PHASE_UP
        elif self.state.phase == PHASE_UP:
            # Arm is up — check if it dropped below release threshold
            if self.state.height <= self.config.raise_release:
                # Debounce
                if (
                    self.state.last_pump_at is None
                    or (now - self.state.last_pump_at) >= self.config.min_pump_seconds
                ):
                    self.state.pumps += 1
                    self.state.last_pump_at = now
                    pumped = True
                self.state.phase = PHASE_DOWN

        return pumped

    def reset(self) -> None:
        self.state = ArmState()


# --- Game phases ---

PHASE_PREPARE = "prepare"
PHASE_PLAYING = "playing"
PHASE_FINISHED = "finished"
PHASE_DEAD = "dead"


# --- Compe mode judging ---

@dataclass
class BeatJudgment:
    """Result of judging a pump against the beatmap."""

    timestamp: float
    hit: bool = False
    hit_at: float | None = None
    missed: bool = False


class CompeJudge:
    """Judges pump timing against a beatmap.

    Rules:
        - Pump within ±hit_window of any unresolved beat = HIT (safe)
        - Pump outside all beat windows = DEATH
        - Beat's miss_window passes with no pump = DEATH
    """

    def __init__(self, beats: list[float], config: CompeConfig):
        self.config = config
        self.judgments: list[BeatJudgment] = [
            BeatJudgment(timestamp=t) for t in sorted(beats)
        ]
        self.next_beat_index: int = 0
        self.dead: bool = False
        self.death_reason: str = ""
        self.hits: int = 0
        self.song_time: float = 0.0

    def update_time(self, song_time: float) -> None:
        """Advance the clock. Check for missed beats."""
        self.song_time = song_time
        if self.dead:
            return

        # Check if any unresolved beats have been missed
        while self.next_beat_index < len(self.judgments):
            beat = self.judgments[self.next_beat_index]
            if beat.hit or beat.missed:
                self.next_beat_index += 1
                continue

            # Has the miss window passed?
            if song_time > beat.timestamp + self.config.miss_window:
                beat.missed = True
                self.dead = True
                self.death_reason = (
                    f"Missed beat at {beat.timestamp:.2f}s"
                )
                return
            break  # Not past this beat yet

    def judge_pump(self, now: float) -> str:
        """A pump happened at song_time=now. Returns 'hit', 'death', or 'ignore'.

        'ignore' means the pump landed but no beat was nearby — in compe mode
        this is DEATH.
        """
        if self.dead:
            return "ignore"

        # Find the closest unresolved beat within the hit window
        best_index: int | None = None
        best_dist: float = float("inf")

        for i, beat in enumerate(self.judgments):
            if beat.hit or beat.missed:
                continue
            dist = abs(now - beat.timestamp)
            if dist <= self.config.hit_window and dist < best_dist:
                best_dist = dist
                best_index = i

        if best_index is not None:
            self.judgments[best_index].hit = True
            self.judgments[best_index].hit_at = now
            self.hits += 1
            return "hit"

        # No beat nearby — DEATH
        self.dead = True
        self.death_reason = f"Pumped at {now:.2f}s — no beat in window"
        return "death"

    @property
    def total_beats(self) -> int:
        return len(self.judgments)

    @property
    def progress(self) -> float:
        """How far through the song (by beats resolved)."""
        if not self.judgments:
            return 0.0
        resolved = sum(1 for j in self.judgments if j.hit or j.missed)
        return resolved / len(self.judgments)


# --- Pose helpers ---

REQUIRED_LANDMARKS = (
    "left_shoulder",
    "right_shoulder",
    "left_wrist",
    "right_wrist",
)

OPTIONAL_LANDMARKS = ("left_hip", "right_hip")


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5


def _body_scale(landmarks: dict[str, tuple[float, float, float]], config: PumpConfig) -> float:
    """Distance-invariant body scale reference.

    Prefers shoulder-to-hip, falls back to shoulder width * factor.
    Landmarks are (x, y, visibility) normalized 0..1.
    """
    margin = config.frame_margin

    # Check if hips are usable
    hips_usable = all(
        name in landmarks
        and landmarks[name][2] >= config.min_visibility
        and -margin <= landmarks[name][1] <= 1 + margin
        for name in OPTIONAL_LANDMARKS
    )

    if hips_usable:
        # Average shoulder-to-hip distance
        left_torso = _distance(
            (landmarks["left_shoulder"][0], landmarks["left_shoulder"][1]),
            (landmarks["left_hip"][0], landmarks["left_hip"][1]),
        )
        right_torso = _distance(
            (landmarks["right_shoulder"][0], landmarks["right_shoulder"][1]),
            (landmarks["right_hip"][0], landmarks["right_hip"][1]),
        )
        scale = (left_torso + right_torso) / 2.0
        if scale > 1e-6:
            return scale

    # Fallback: shoulder width * factor
    shoulder_width = _distance(
        (landmarks["left_shoulder"][0], landmarks["left_shoulder"][1]),
        (landmarks["right_shoulder"][0], landmarks["right_shoulder"][1]),
    )
    return max(shoulder_width * config.shoulder_to_torso, 1e-6)


def _wrist_height(
    landmarks: dict[str, tuple[float, float, float]],
    side: str,
    torso_length: float,
) -> float | None:
    """How far above the shoulder the wrist is, in torso-length units.

    Positive = above shoulder. Image Y grows downward, so shoulder_y - wrist_y.
    Returns None if the wrist landmark is missing.
    """
    wrist_key = f"{side}_wrist"
    shoulder_key = f"{side}_shoulder"
    if wrist_key not in landmarks or shoulder_key not in landmarks:
        return None

    wrist_y = landmarks[wrist_key][1]
    shoulder_y = landmarks[shoulder_key][1]

    # shoulder_y - wrist_y because Y grows downward; positive = wrist above shoulder
    return (shoulder_y - wrist_y) / torso_length


def _check_missing(
    landmarks: dict[str, tuple[float, float, float]], config: PumpConfig
) -> list[str]:
    """Return list of required landmarks that are absent or low-visibility."""
    margin = config.frame_margin
    missing: list[str] = []
    for name in REQUIRED_LANDMARKS:
        if name not in landmarks:
            missing.append(name)
            continue
        x, y, vis = landmarks[name]
        if vis < config.min_visibility:
            missing.append(name)
            continue
        if not (-margin <= x <= 1 + margin and -margin <= y <= 1 + margin):
            missing.append(name)
    return missing


# --- Main game engine ---

class ErinCore:
    """Help Me ERINNNNNN!! game engine.

    Feed it pose landmarks each frame. It handles both Open and Compe modes.

    Landmarks format: dict mapping name -> (x, y, visibility), all normalized 0..1.
    """

    def __init__(self, config: GameConfig | None = None):
        self.config = config or GameConfig()
        self.left_arm = ArmTracker("left", self.config.pump)
        self.right_arm = ArmTracker("right", self.config.pump)
        self.score: int = 0
        self.phase: str = PHASE_PREPARE
        self.missing: list[str] = list(REQUIRED_LANDMARKS)
        self.pose_visible: bool = False
        self.torso_length: float = 0.0

        # Timing
        self._ready_since: float | None = None
        self._lost_since: float | None = None
        self._game_start: float | None = None
        self._time_remaining: float = self.config.timer_seconds

        # Compe mode
        self.judge: CompeJudge | None = None

        # Events for renderer
        self.last_pump_side: str | None = None
        self.last_pump_at: float | None = None
        self.combo: int = 0
        self.max_combo: int = 0

        # History for stats
        self.pump_history: list[dict[str, Any]] = []

    def set_beatmap(self, beats: list[float]) -> None:
        """Load beat timestamps for compe mode."""
        self.judge = CompeJudge(beats, self.config.compe)

    def update(
        self,
        landmarks: dict[str, tuple[float, float, float]] | None,
        now: float,
    ) -> dict[str, Any]:
        """Main update. Call every frame.

        Args:
            landmarks: Pose landmarks as {name: (x, y, visibility)}, or None.
            now: Monotonic timestamp in seconds.

        Returns:
            State dict for the renderer.
        """
        # Check pose visibility
        if landmarks is None:
            self.missing = list(REQUIRED_LANDMARKS)
        else:
            self.missing = _check_missing(landmarks, self.config.pump)
        self.pose_visible = not self.missing

        # Handle lost tracking
        if not self.pose_visible:
            if self._lost_since is None:
                self._lost_since = now
            if (now - self._lost_since) >= self.config.pump.lost_grace_seconds:
                if self.phase == PHASE_PLAYING:
                    # Don't kill the game, just pause detection
                    pass
                elif self.phase != PHASE_FINISHED and self.phase != PHASE_DEAD:
                    self.phase = PHASE_PREPARE
                    self._ready_since = None
            return self._state(now)

        self._lost_since = None

        # Compute body scale
        self.torso_length = _body_scale(landmarks, self.config.pump)  # type: ignore[arg-type]

        # Prepare phase: wait for stable pose
        if self.phase == PHASE_PREPARE:
            if self._ready_since is None:
                self._ready_since = now
            if (now - self._ready_since) >= self.config.pump.prepare_seconds:
                self.phase = PHASE_PLAYING
                self._game_start = now
            return self._state(now)

        # Game over states
        if self.phase in (PHASE_FINISHED, PHASE_DEAD):
            return self._state(now)

        # --- PLAYING ---
        assert self.phase == PHASE_PLAYING
        assert landmarks is not None

        # Open mode timer
        if self.config.mode == Mode.OPEN:
            game_start = self._game_start if self._game_start is not None else now
            elapsed = now - game_start
            self._time_remaining = max(0.0, self.config.timer_seconds - elapsed)
            if self._time_remaining <= 0:
                self.phase = PHASE_FINISHED
                return self._state(now)

        # Compe mode time tracking
        if self.config.mode == Mode.COMPE and self.judge is not None:
            game_start = self._game_start if self._game_start is not None else now
            song_time = now - game_start
            self.judge.update_time(song_time)
            if self.judge.dead:
                self.phase = PHASE_DEAD
                return self._state(now)

        # Detect pumps on each arm
        left_height = _wrist_height(landmarks, "left", self.torso_length)
        right_height = _wrist_height(landmarks, "right", self.torso_length)

        left_pumped = self.left_arm.update(left_height, now)
        right_pumped = self.right_arm.update(right_height, now)

        # Score and judge
        if left_pumped:
            self._on_pump("left", now)
        if right_pumped:
            self._on_pump("right", now)

        return self._state(now)

    def _on_pump(self, side: str, now: float) -> None:
        """Handle a completed pump."""
        self.last_pump_side = side
        self.last_pump_at = now

        if self.config.mode == Mode.OPEN:
            self.score += 1
            self.combo += 1
            self.max_combo = max(self.max_combo, self.combo)
            self.pump_history.append({"side": side, "time": now})

        elif self.config.mode == Mode.COMPE and self.judge is not None:
            game_start = self._game_start if self._game_start is not None else now
            song_time = now - game_start
            result = self.judge.judge_pump(song_time)
            if result == "hit":
                self.score += 1
                self.combo += 1
                self.max_combo = max(self.max_combo, self.combo)
                self.pump_history.append({"side": side, "time": now, "result": "hit"})
            elif result == "death":
                self.phase = PHASE_DEAD
                self.combo = 0
                self.pump_history.append({"side": side, "time": now, "result": "death"})

    def prepare_progress(self, now: float) -> float:
        """0..1 fraction of prepare hold completed."""
        if self.phase != PHASE_PREPARE:
            return 1.0
        if self._ready_since is None:
            return 0.0
        elapsed = now - self._ready_since
        return min(1.0, elapsed / self.config.pump.prepare_seconds)

    @property
    def time_remaining(self) -> float:
        """Seconds left in open mode."""
        return self._time_remaining

    @property
    def pumps_per_minute(self) -> float | None:
        """Tempo over recent pumps."""
        if len(self.pump_history) < 2:
            return None
        recent = self.pump_history[-20:]
        span = recent[-1]["time"] - recent[0]["time"]
        if span <= 0:
            return None
        return round((len(recent) - 1) / span * 60.0, 1)

    @property
    def death_reason(self) -> str:
        """Why the player died (compe mode)."""
        if self.judge and self.judge.dead:
            return self.judge.death_reason
        return ""

    def hint(self) -> str:
        """Short instruction for the user."""
        if self.phase == PHASE_DEAD:
            return "YOU DIED"
        if self.phase == PHASE_FINISHED:
            return "TIME'S UP!"
        if not self.missing:
            if self.phase == PHASE_PREPARE:
                return "HOLD STILL..."
            return "PUMP YOUR FISTS! ✊"
        if len(self.missing) >= len(REQUIRED_LANDMARKS):
            return "STEP INTO FRAME"
        wrist_missing = any("wrist" in m for m in self.missing)
        if wrist_missing:
            return "SHOW BOTH HANDS"
        return "STEP BACK - SHOULDERS NOT VISIBLE"

    def reset(self) -> None:
        """Reset everything for a new game."""
        self.left_arm.reset()
        self.right_arm.reset()
        self.score = 0
        self.phase = PHASE_PREPARE
        self.missing = list(REQUIRED_LANDMARKS)
        self.pose_visible = False
        self._ready_since = None
        self._lost_since = None
        self._game_start = None
        self._time_remaining = self.config.timer_seconds
        self.last_pump_side = None
        self.last_pump_at = None
        self.combo = 0
        self.max_combo = 0
        self.pump_history.clear()
        if self.judge:
            self.judge = CompeJudge(
                [j.timestamp for j in self.judge.judgments],
                self.config.compe,
            )

    def _state(self, now: float) -> dict[str, Any]:
        """Build the state dict for the renderer."""
        return {
            "mode": self.config.mode.value,
            "phase": self.phase,
            "score": self.score,
            "combo": self.combo,
            "max_combo": self.max_combo,
            "time_remaining": round(self._time_remaining, 1),
            "prepare_progress": round(self.prepare_progress(now), 3),
            "pose_visible": self.pose_visible,
            "missing": list(self.missing),
            "hint": self.hint(),
            "left_arm": {
                "phase": self.left_arm.state.phase,
                "height": round(self.left_arm.state.height, 4),
                "pumps": self.left_arm.state.pumps,
            },
            "right_arm": {
                "phase": self.right_arm.state.phase,
                "height": round(self.right_arm.state.height, 4),
                "pumps": self.right_arm.state.pumps,
            },
            "last_pump_side": self.last_pump_side,
            "last_pump_at": self.last_pump_at,
            "pumps_per_minute": self.pumps_per_minute,
            "torso_length": round(self.torso_length, 4),
            "death_reason": self.death_reason,
            # Compe-specific
            "compe_hits": self.judge.hits if self.judge else 0,
            "compe_total": self.judge.total_beats if self.judge else 0,
            "compe_progress": round(self.judge.progress, 3) if self.judge else 0.0,
        }
