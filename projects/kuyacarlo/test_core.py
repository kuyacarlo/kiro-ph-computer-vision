"""Tests for Help Me ERINNNNNN!! core logic.

No camera, no audio, no display. All tests use synthetic landmarks and
a fake monotonic clock.

Run from repository root:
    python -m pytest projects/kuyacarlo/test_core.py -v
"""

from __future__ import annotations

import pytest

from .config import (
    CompeConfig,
    Difficulty,
    DIFFICULTY_TIMERS,
    GameConfig,
    Mode,
    PumpConfig,
)
from .core import (
    PHASE_DEAD,
    PHASE_DOWN,
    PHASE_FINISHED,
    PHASE_PLAYING,
    PHASE_PREPARE,
    PHASE_UP,
    ArmTracker,
    CompeJudge,
    ErinCore,
    _body_scale,
    _check_missing,
    _distance,
    _wrist_height,
)


# --- Fixtures: synthetic landmarks ---

def make_landmarks(
    left_wrist_y: float = 0.5,
    right_wrist_y: float = 0.5,
    left_shoulder_y: float = 0.4,
    right_shoulder_y: float = 0.4,
    left_hip_y: float = 0.7,
    right_hip_y: float = 0.7,
    visibility: float = 0.95,
    include_hips: bool = True,
) -> dict[str, tuple[float, float, float]]:
    """Create synthetic pose landmarks.

    Y axis: 0 = top of frame, 1 = bottom. So wrist_y < shoulder_y means arm raised.
    """
    landmarks = {
        "left_shoulder": (0.6, left_shoulder_y, visibility),
        "right_shoulder": (0.4, right_shoulder_y, visibility),
        "left_wrist": (0.65, left_wrist_y, visibility),
        "right_wrist": (0.35, right_wrist_y, visibility),
    }
    if include_hips:
        landmarks["left_hip"] = (0.55, left_hip_y, visibility)
        landmarks["right_hip"] = (0.45, right_hip_y, visibility)
    return landmarks


def arms_down() -> dict[str, tuple[float, float, float]]:
    """Both arms at rest (below shoulders)."""
    return make_landmarks(left_wrist_y=0.6, right_wrist_y=0.6)


def left_arm_up() -> dict[str, tuple[float, float, float]]:
    """Left arm raised above head."""
    return make_landmarks(left_wrist_y=0.1, right_wrist_y=0.6)


def right_arm_up() -> dict[str, tuple[float, float, float]]:
    """Right arm raised above head."""
    return make_landmarks(left_wrist_y=0.6, right_wrist_y=0.1)


def both_arms_up() -> dict[str, tuple[float, float, float]]:
    """Both arms raised above head."""
    return make_landmarks(left_wrist_y=0.1, right_wrist_y=0.1)


# --- Test helpers ---

class TestDistance:
    def test_same_point(self) -> None:
        assert _distance((0.5, 0.5), (0.5, 0.5)) == 0.0

    def test_horizontal(self) -> None:
        assert abs(_distance((0.0, 0.0), (1.0, 0.0)) - 1.0) < 1e-6

    def test_diagonal(self) -> None:
        d = _distance((0.0, 0.0), (3.0, 4.0))
        assert abs(d - 5.0) < 1e-6


class TestBodyScale:
    def test_with_hips(self) -> None:
        lm = make_landmarks(include_hips=True)
        config = PumpConfig()
        scale = _body_scale(lm, config)
        # Should be ~shoulder-to-hip distance
        assert scale > 0.1
        assert scale < 1.0

    def test_without_hips(self) -> None:
        lm = make_landmarks(include_hips=False)
        config = PumpConfig()
        scale = _body_scale(lm, config)
        # Should fall back to shoulder width * factor
        assert scale > 0.05
        assert scale < 1.0

    def test_never_zero(self) -> None:
        lm = make_landmarks()
        # Shoulders at same position
        lm["left_shoulder"] = (0.5, 0.5, 0.95)
        lm["right_shoulder"] = (0.5, 0.5, 0.95)
        lm.pop("left_hip", None)
        lm.pop("right_hip", None)
        config = PumpConfig()
        scale = _body_scale(lm, config)
        assert scale > 0


class TestWristHeight:
    def test_arm_raised(self) -> None:
        lm = make_landmarks(left_wrist_y=0.1, left_shoulder_y=0.4, include_hips=True)
        config = PumpConfig()
        torso = _body_scale(lm, config)
        height = _wrist_height(lm, "left", torso)
        # Wrist above shoulder → positive
        assert height is not None
        assert height > 0

    def test_arm_lowered(self) -> None:
        lm = make_landmarks(left_wrist_y=0.6, left_shoulder_y=0.4, include_hips=True)
        config = PumpConfig()
        torso = _body_scale(lm, config)
        height = _wrist_height(lm, "left", torso)
        # Wrist below shoulder → negative
        assert height is not None
        assert height < 0

    def test_missing_wrist(self) -> None:
        lm = make_landmarks()
        del lm["left_wrist"]
        height = _wrist_height(lm, "left", 0.3)
        assert height is None


class TestCheckMissing:
    def test_all_present(self) -> None:
        lm = make_landmarks()
        config = PumpConfig()
        missing = _check_missing(lm, config)
        assert missing == []

    def test_no_landmarks(self) -> None:
        config = PumpConfig()
        missing = _check_missing({}, config)
        assert len(missing) == 4

    def test_low_visibility(self) -> None:
        lm = make_landmarks(visibility=0.1)
        config = PumpConfig(min_visibility=0.5)
        missing = _check_missing(lm, config)
        assert len(missing) == 4


# --- Test ArmTracker ---

class TestArmTracker:
    def test_initial_state(self) -> None:
        tracker = ArmTracker("left", PumpConfig())
        assert tracker.state.phase == PHASE_DOWN
        assert tracker.state.pumps == 0

    def test_no_pump_on_raise(self) -> None:
        """Raising the arm should NOT count — pump counts on the down."""
        config = PumpConfig(raise_enter=0.35, raise_release=0.20, height_alpha=1.0)
        tracker = ArmTracker("left", config)
        # Arm goes up
        pumped = tracker.update(0.5, 1.0)
        assert not pumped
        assert tracker.state.phase == PHASE_UP

    def test_pump_on_full_cycle(self) -> None:
        """UP then DOWN = one pump."""
        config = PumpConfig(raise_enter=0.35, raise_release=0.20,
                           height_alpha=1.0, min_pump_seconds=0.0)
        tracker = ArmTracker("left", config)

        # Raise arm
        tracker.update(0.5, 1.0)
        assert tracker.state.phase == PHASE_UP

        # Lower arm
        pumped = tracker.update(0.1, 1.5)
        assert pumped
        assert tracker.state.pumps == 1
        assert tracker.state.phase == PHASE_DOWN

    def test_debounce(self) -> None:
        """Pumps too fast should be debounced."""
        config = PumpConfig(raise_enter=0.35, raise_release=0.20,
                           height_alpha=1.0, min_pump_seconds=0.3)
        tracker = ArmTracker("left", config)

        # First pump
        tracker.update(0.5, 1.0)
        tracker.update(0.1, 1.1)
        assert tracker.state.pumps == 1

        # Too fast
        tracker.update(0.5, 1.15)
        pumped = tracker.update(0.1, 1.2)
        assert not pumped  # Debounced
        assert tracker.state.pumps == 1

    def test_none_height_ignored(self) -> None:
        """None readings should not affect state."""
        tracker = ArmTracker("left", PumpConfig())
        pumped = tracker.update(None, 1.0)
        assert not pumped
        assert tracker.state.phase == PHASE_DOWN

    def test_hysteresis_prevents_flicker(self) -> None:
        """Height between enter and release should not trigger transitions."""
        config = PumpConfig(raise_enter=0.35, raise_release=0.20, height_alpha=1.0)
        tracker = ArmTracker("left", config)

        # Height in between — should stay down
        tracker.update(0.25, 1.0)
        assert tracker.state.phase == PHASE_DOWN

        # Go above enter
        tracker.update(0.4, 1.5)
        assert tracker.state.phase == PHASE_UP

        # Drop to between enter and release — should stay up
        tracker.update(0.25, 2.0)
        assert tracker.state.phase == PHASE_UP

    def test_multiple_pumps(self) -> None:
        """Multiple full cycles should count correctly."""
        config = PumpConfig(raise_enter=0.35, raise_release=0.20,
                           height_alpha=1.0, min_pump_seconds=0.0)
        tracker = ArmTracker("left", config)

        for i in range(5):
            t = float(i * 2)
            tracker.update(0.5, t)       # up
            tracker.update(0.1, t + 1)   # down → pump

        assert tracker.state.pumps == 5


# --- Test CompeJudge ---

class TestCompeJudge:
    def test_hit_within_window(self) -> None:
        beats = [1.0, 2.0, 3.0]
        config = CompeConfig(hit_window=0.3, miss_window=0.5)
        judge = CompeJudge(beats, config)

        result = judge.judge_pump(1.1)  # Within ±0.3 of beat at 1.0
        assert result == "hit"
        assert judge.hits == 1
        assert not judge.dead

    def test_death_outside_window(self) -> None:
        beats = [2.0, 4.0]
        config = CompeConfig(hit_window=0.3, miss_window=0.5)
        judge = CompeJudge(beats, config)

        # Pump at 0.5 — way before any beat
        result = judge.judge_pump(0.5)
        assert result == "death"
        assert judge.dead
        assert "no beat in window" in judge.death_reason.lower()

    def test_miss_on_timeout(self) -> None:
        beats = [1.0, 2.0]
        config = CompeConfig(hit_window=0.3, miss_window=0.5)
        judge = CompeJudge(beats, config)

        # Time passes beyond miss window of first beat
        judge.update_time(1.6)  # 1.0 + 0.5 = 1.5, so 1.6 is past
        assert judge.dead
        assert "missed beat" in judge.death_reason.lower()

    def test_multiple_hits(self) -> None:
        beats = [1.0, 2.0, 3.0]
        config = CompeConfig(hit_window=0.3, miss_window=0.5)
        judge = CompeJudge(beats, config)

        judge.judge_pump(1.0)
        judge.update_time(1.5)
        judge.judge_pump(2.1)
        judge.update_time(2.5)
        judge.judge_pump(2.9)

        assert judge.hits == 3
        assert not judge.dead

    def test_progress(self) -> None:
        beats = [1.0, 2.0, 3.0, 4.0]
        config = CompeConfig(hit_window=0.3, miss_window=0.5)
        judge = CompeJudge(beats, config)

        judge.judge_pump(1.0)
        judge.judge_pump(2.0)
        assert judge.progress == pytest.approx(0.5)

    def test_no_double_hit(self) -> None:
        """A beat can only be hit once."""
        beats = [1.0]
        config = CompeConfig(hit_window=0.3, miss_window=0.5)
        judge = CompeJudge(beats, config)

        judge.judge_pump(1.0)
        # Second pump near same beat — no more unresolved beats
        result = judge.judge_pump(1.1)
        assert result == "death"  # No beat to match


# --- Test ErinCore ---

class TestErinCoreOpenMode:
    def _make_core(self, difficulty: Difficulty = Difficulty.MEDIUM) -> ErinCore:
        config = GameConfig(
            mode=Mode.OPEN,
            difficulty=difficulty,
            pump=PumpConfig(
                raise_enter=0.35,
                raise_release=0.20,
                height_alpha=1.0,
                min_pump_seconds=0.0,
                prepare_seconds=0.5,
            ),
        )
        return ErinCore(config)

    def test_starts_in_prepare(self) -> None:
        core = self._make_core()
        state = core.update(arms_down(), 0.0)
        assert state["phase"] == PHASE_PREPARE

    def test_transitions_to_playing(self) -> None:
        core = self._make_core()
        # Feed landmarks for the prepare duration
        core.update(arms_down(), 0.0)
        core.update(arms_down(), 0.3)
        state = core.update(arms_down(), 0.6)  # Past 0.5s prepare time
        assert state["phase"] == PHASE_PLAYING

    def test_counts_left_pump(self) -> None:
        core = self._make_core()
        # Get past prepare
        core.update(arms_down(), 0.0)
        core.update(arms_down(), 0.6)

        # Raise left arm
        core.update(left_arm_up(), 1.0)
        # Lower left arm → pump!
        state = core.update(arms_down(), 1.5)
        assert state["score"] == 1
        assert state["left_arm"]["pumps"] == 1

    def test_counts_right_pump(self) -> None:
        core = self._make_core()
        core.update(arms_down(), 0.0)
        core.update(arms_down(), 0.6)

        core.update(right_arm_up(), 1.0)
        state = core.update(arms_down(), 1.5)
        assert state["score"] == 1
        assert state["right_arm"]["pumps"] == 1

    def test_both_arms_scores_two(self) -> None:
        core = self._make_core()
        core.update(arms_down(), 0.0)
        core.update(arms_down(), 0.6)

        # Both arms up
        core.update(both_arms_up(), 1.0)
        # Both arms down → 2 pumps
        state = core.update(arms_down(), 1.5)
        assert state["score"] == 2
        assert state["left_arm"]["pumps"] == 1
        assert state["right_arm"]["pumps"] == 1

    def test_timer_ends_game(self) -> None:
        config = GameConfig(
            mode=Mode.OPEN,
            difficulty=Difficulty.HELL,  # 15 seconds
            pump=PumpConfig(prepare_seconds=0.0, height_alpha=1.0),
        )
        core = ErinCore(config)
        core.update(arms_down(), 0.0)  # Immediately playing (0s prepare)

        # Jump way past timer
        state = core.update(arms_down(), 20.0)
        assert state["phase"] == PHASE_FINISHED
        assert state["time_remaining"] == 0.0

    def test_difficulty_timers(self) -> None:
        for diff, expected in DIFFICULTY_TIMERS.items():
            config = GameConfig(mode=Mode.OPEN, difficulty=diff)
            assert config.timer_seconds == expected

    def test_no_pose_stays_in_prepare(self) -> None:
        core = self._make_core()
        state = core.update(None, 0.0)
        assert state["phase"] == PHASE_PREPARE
        assert not state["pose_visible"]

    def test_reset(self) -> None:
        core = self._make_core()
        core.update(arms_down(), 0.0)
        core.update(arms_down(), 0.6)
        core.update(left_arm_up(), 1.0)
        core.update(arms_down(), 1.5)
        assert core.score == 1

        core.reset()
        assert core.score == 0
        assert core.phase == PHASE_PREPARE
        assert core.combo == 0

    def test_combo_increments(self) -> None:
        core = self._make_core()
        core.update(arms_down(), 0.0)
        core.update(arms_down(), 0.6)

        # Three pumps in a row
        for i in range(3):
            t = 1.0 + i * 2
            core.update(left_arm_up(), t)
            core.update(arms_down(), t + 1)

        assert core.combo == 3
        assert core.max_combo == 3


class TestErinCoreCompeMode:
    def _make_core(self, beats: list[float] | None = None) -> ErinCore:
        config = GameConfig(
            mode=Mode.COMPE,
            pump=PumpConfig(
                raise_enter=0.35,
                raise_release=0.20,
                height_alpha=1.0,
                min_pump_seconds=0.0,
                prepare_seconds=0.0,
            ),
            compe=CompeConfig(hit_window=0.3, miss_window=0.5),
        )
        core = ErinCore(config)
        if beats:
            core.set_beatmap(beats)
        return core

    def test_hit_on_beat(self) -> None:
        core = self._make_core(beats=[2.0, 4.0])
        # Get to playing
        core.update(arms_down(), 0.0)
        assert core.phase == PHASE_PLAYING

        # Pump on beat (game time ≈ 2.0)
        core.update(left_arm_up(), 1.5)
        state = core.update(arms_down(), 2.0)  # Pump at game_time=2.0
        assert state["phase"] == PHASE_PLAYING
        assert state["score"] == 1

    def test_death_on_early_pump(self) -> None:
        core = self._make_core(beats=[5.0])
        core.update(arms_down(), 0.0)

        # Pump way before the beat
        core.update(left_arm_up(), 0.5)
        state = core.update(arms_down(), 1.0)  # game_time=1.0, beat at 5.0
        assert state["phase"] == PHASE_DEAD

    def test_death_on_missed_beat(self) -> None:
        core = self._make_core(beats=[1.0])
        core.update(arms_down(), 0.0)

        # Don't pump, just let time pass
        state = core.update(arms_down(), 2.0)  # game_time=2.0, miss_window=0.5
        assert state["phase"] == PHASE_DEAD

    def test_compe_state_fields(self) -> None:
        core = self._make_core(beats=[1.0, 2.0, 3.0])
        state = core.update(arms_down(), 0.0)
        assert state["compe_total"] == 3
        assert state["compe_hits"] == 0
        assert state["compe_progress"] == 0.0


class TestPreparePhase:
    def test_prepare_progress(self) -> None:
        config = GameConfig(
            mode=Mode.OPEN,
            pump=PumpConfig(prepare_seconds=2.0),
        )
        core = ErinCore(config)
        core.update(arms_down(), 0.0)

        state = core.update(arms_down(), 1.0)
        assert state["prepare_progress"] == pytest.approx(0.5, abs=0.05)

    def test_lost_pose_resets_prepare(self) -> None:
        config = GameConfig(
            mode=Mode.OPEN,
            pump=PumpConfig(prepare_seconds=1.0, lost_grace_seconds=0.3),
        )
        core = ErinCore(config)
        core.update(arms_down(), 0.0)
        core.update(arms_down(), 0.5)  # Half-way through prepare

        # Lose tracking for longer than grace period
        core.update(None, 0.6)
        core.update(None, 1.0)  # 0.4s without pose > 0.3s grace

        # Back in frame — should restart prepare
        state = core.update(arms_down(), 1.1)
        assert state["phase"] == PHASE_PREPARE
        assert state["prepare_progress"] < 0.5  # Reset


class TestHint:
    def test_no_pose(self) -> None:
        core = ErinCore(GameConfig())
        core.update(None, 0.0)
        assert "STEP INTO FRAME" in core.hint().upper()

    def test_missing_wrists(self) -> None:
        lm = make_landmarks()
        lm["left_wrist"] = (0.5, 0.5, 0.1)  # Low visibility
        core = ErinCore(GameConfig(pump=PumpConfig(min_visibility=0.5)))
        core.update(lm, 0.0)
        assert "HAND" in core.hint().upper()

    def test_playing_hint(self) -> None:
        config = GameConfig(pump=PumpConfig(prepare_seconds=0.0))
        core = ErinCore(config)
        core.update(arms_down(), 0.0)
        assert "PUMP" in core.hint().upper() or "FIST" in core.hint().upper()
