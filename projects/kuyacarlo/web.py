"""Help Me ERINNNNNN!! — Flask web version.

Run from the repository root:
    python -m projects.kuyacarlo.web
    python -m projects.kuyacarlo.web --port 5050 --mode compe --song song.mp3 --beatmap beatmap.json

The browser runs MediaPipe Pose (via @mediapipe/tasks-vision) and streams
landmarks over a WebSocket. This server runs the game logic and streams
state back. The browser renders the UI.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from flask import Flask, render_template
from flask_sock import Sock

from .audio import Beatmap
from .config import CompeConfig, Difficulty, GameConfig, Mode, PumpConfig
from .core import ErinCore

app = Flask(
    __name__,
    template_folder=str(Path(__file__).parent / "templates"),
)
sock = Sock(app)


# Global config — set from CLI args before app starts
_config: GameConfig = GameConfig()
_beatmap_beats: list[float] = []


def _make_core() -> ErinCore:
    """Create a fresh game core with current config."""
    core = ErinCore(_config)
    if _config.mode == Mode.COMPE and _beatmap_beats:
        core.set_beatmap(_beatmap_beats)
    return core


@app.route("/")
def index() -> str:
    """Serve the main page."""
    return render_template(
        "erin.html",
        mode=_config.mode.value,
        difficulty=_config.difficulty.value,
        timer_seconds=_config.timer_seconds,
        song_path=_config.compe.song_path,
        has_audio=bool(_config.compe.song_path),
    )


@sock.route("/ws")
def websocket(ws: Any) -> None:
    """WebSocket handler: receives landmarks, sends game state."""
    core = _make_core()
    start_time = time.monotonic()

    while True:
        try:
            raw = ws.receive(timeout=5.0)
        except Exception:
            break

        if raw is None:
            break

        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            continue

        msg_type = msg.get("type", "landmarks")

        if msg_type == "landmarks":
            # Convert browser landmarks to our format
            landmarks = _parse_browser_landmarks(msg.get("landmarks"))
            now = time.monotonic() - start_time
            state = core.update(landmarks, now)
            ws.send(json.dumps({"type": "state", **state}))

        elif msg_type == "command":
            command = msg.get("command", "")
            if command == "reset":
                core.reset()
                start_time = time.monotonic()
            elif command == "set_mode":
                # Can change mode before game starts
                new_mode = msg.get("mode", "open")
                _config.mode = Mode(new_mode)
                core = _make_core()
                start_time = time.monotonic()
            elif command == "set_difficulty":
                new_diff = msg.get("difficulty", "medium")
                _config.difficulty = Difficulty(new_diff)
                core.config.difficulty = Difficulty(new_diff)
                core.reset()
                start_time = time.monotonic()

            ws.send(json.dumps({"type": "ack", "command": command}))


def _parse_browser_landmarks(
    raw_landmarks: list[dict[str, float]] | None,
) -> dict[str, tuple[float, float, float]] | None:
    """Convert MediaPipe Tasks browser landmarks to our server format.

    Browser sends: [{x, y, z, visibility}, ...] indexed by landmark ID.
    We need: {name: (x, y, visibility)}
    """
    if not raw_landmarks:
        return None

    # MediaPipe Pose landmark indices → names
    LANDMARK_MAP = {
        11: "left_shoulder",
        12: "right_shoulder",
        13: "left_elbow",
        14: "right_elbow",
        15: "left_wrist",
        16: "right_wrist",
        23: "left_hip",
        24: "right_hip",
    }

    landmarks: dict[str, tuple[float, float, float]] = {}
    for idx, name in LANDMARK_MAP.items():
        if idx < len(raw_landmarks):
            lm = raw_landmarks[idx]
            landmarks[name] = (
                lm.get("x", 0.0),
                lm.get("y", 0.0),
                lm.get("visibility", 0.0),
            )

    return landmarks if landmarks else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Help Me ERINNNNNN!! web server")
    parser.add_argument("--port", type=int, default=5050, help="Server port")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument(
        "--mode", choices=["open", "compe"], default="open", help="Game mode"
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard", "hell"], default="medium",
        help="Difficulty preset"
    )
    parser.add_argument("--song", type=str, default="", help="Song file path (compe)")
    parser.add_argument("--beatmap", type=str, default="", help="Beatmap JSON path (compe)")
    return parser.parse_args()


def main() -> None:
    global _config, _beatmap_beats

    args = parse_args()

    _config = GameConfig(
        mode=Mode(args.mode),
        difficulty=Difficulty(args.difficulty),
        web_host=args.host,
        web_port=args.port,
        compe=CompeConfig(song_path=args.song, beatmap_path=args.beatmap),
    )

    if _config.mode == Mode.COMPE and _config.compe.beatmap_path:
        beatmap = Beatmap.load(_config.compe.beatmap_path)
        _beatmap_beats = beatmap.beats

    print(f"\n🎮 Help Me ERINNNNNN!! — Web version")
    print(f"   Mode: {_config.mode.value.upper()}")
    print(f"   Difficulty: {_config.difficulty.value}")
    print(f"   http://{args.host}:{args.port}/")
    print(f"   Press Ctrl+C to stop\n")

    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()
