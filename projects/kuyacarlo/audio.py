"""Audio playback and beatmap loading for Help Me ERINNNNNN!!

Uses pygame.mixer for lightweight audio playback (no full pygame display needed).
Beatmap is a JSON file with an array of beat timestamps in seconds.

Beatmap format:
    {
        "song": "help_me_erin.mp3",
        "bpm": 170,
        "beats": [1.2, 2.4, 3.6, ...]
    }
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class Beatmap:
    """Parsed beatmap data."""

    song_path: str = ""
    bpm: float = 0.0
    beats: list[float] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "Beatmap":
        """Load a beatmap from a JSON file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Beatmap not found: {path}")

        with open(path) as f:
            data = json.load(f)

        return cls(
            song_path=data.get("song", ""),
            bpm=data.get("bpm", 0.0),
            beats=sorted(data.get("beats", [])),
        )

    def save(self, path: str | Path) -> None:
        """Save beatmap to JSON."""
        path = Path(path)
        data = {
            "song": self.song_path,
            "bpm": self.bpm,
            "beats": self.beats,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @property
    def duration(self) -> float:
        """Estimated duration (last beat + a buffer)."""
        if not self.beats:
            return 0.0
        return self.beats[-1] + 2.0


class AudioPlayer:
    """Thin wrapper over pygame.mixer for song playback.

    Handles init/cleanup and provides a monotonic song clock.
    """

    def __init__(self) -> None:
        self._initialized = False
        self._playing = False
        self._start_time: float | None = None
        self._pause_offset: float = 0.0

    def init(self) -> None:
        """Initialize pygame mixer. Call once at startup."""
        if self._initialized:
            return
        try:
            import pygame.mixer

            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            self._initialized = True
        except ImportError:
            raise ImportError(
                "pygame is required for audio playback. "
                "Install with: pip install pygame"
            )

    def load(self, path: str | Path) -> None:
        """Load a song file."""
        self.init()
        import pygame.mixer

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Song file not found: {path}")
        pygame.mixer.music.load(str(path))

    def play(self) -> None:
        """Start playback."""
        if not self._initialized:
            self.init()
        import pygame.mixer

        pygame.mixer.music.play()
        self._start_time = time.monotonic()
        self._pause_offset = 0.0
        self._playing = True

    def stop(self) -> None:
        """Stop playback."""
        if not self._initialized:
            return
        import pygame.mixer

        pygame.mixer.music.stop()
        self._playing = False
        self._start_time = None

    def pause(self) -> None:
        """Pause playback."""
        if not self._initialized or not self._playing:
            return
        import pygame.mixer

        self._pause_offset = self.song_time
        pygame.mixer.music.pause()
        self._playing = False

    def unpause(self) -> None:
        """Resume playback."""
        if not self._initialized:
            return
        import pygame.mixer

        pygame.mixer.music.unpause()
        self._start_time = time.monotonic() - self._pause_offset
        self._playing = True

    @property
    def song_time(self) -> float:
        """Current position in the song (seconds). Monotonic-based for accuracy."""
        if self._start_time is None:
            return self._pause_offset
        if not self._playing:
            return self._pause_offset
        return time.monotonic() - self._start_time

    @property
    def is_playing(self) -> bool:
        """Whether audio is currently playing."""
        if not self._initialized:
            return False
        import pygame.mixer

        return pygame.mixer.music.get_busy()  # type: ignore[no-any-return]

    def cleanup(self) -> None:
        """Shut down the mixer."""
        if not self._initialized:
            return
        import pygame.mixer

        pygame.mixer.music.stop()
        pygame.mixer.quit()
        self._initialized = False
        self._playing = False
