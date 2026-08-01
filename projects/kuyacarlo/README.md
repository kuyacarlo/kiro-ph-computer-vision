# Help Me ERINNNNNN!! — Fist Pump Counter

A Touhou-themed fist pump counter built with computer vision. Raise your fists to
the sky and pump like you're at a "Help me, ERINNNNNN!!" concert.

Two game modes:

- **Open Mode** — Timed. Pump as many times as you can before the clock runs out.
  Each arm scores independently: one arm up = 1 point, both arms = 2 per beat.
- **Compe Mode** — Rhythm game. A song plays with a beatmap. Pump on the beat and
  you're safe. Pump at the wrong time, or miss a beat entirely, and you get a
  lovingly-crafted BSOD from Eirin.

## How It Works

Your webcam tracks both wrists via MediaPipe Pose. A "pump" is one full cycle:
fist rises above head → drops back below shoulder. The signal is scale-relative
(measured as a fraction of torso length), so it works whether you're close or far
from the camera.

Both arms are tracked independently. No calibration needed — just step into frame
and pump.

## Running

### Desktop (fullscreen OpenCV)

```bash
# From repository root, with the venv active:
python -m projects.kuyacarlo.main

# Options:
python -m projects.kuyacarlo.main --mode open --difficulty hell
python -m projects.kuyacarlo.main --mode compe --song path/to/song.mp3 --beatmap beatmap.json
python -m projects.kuyacarlo.main --no-fullscreen --camera 1
```

**Desktop controls:** Q/ESC = quit, R = restart, F = toggle fullscreen, 1-4 = difficulty

### Web (Flask, fullscreen browser)

```bash
python -m projects.kuyacarlo.web

# Options:
python -m projects.kuyacarlo.web --port 5050 --mode open --difficulty hard
```

Open http://127.0.0.1:5050/ in a browser. Click "Fullscreen" or press F. The browser
handles pose detection locally via MediaPipe Tasks (CDN).

## Dependencies

The core logic and web version use what's already in the repo's `requirements.txt`:
- `opencv-python`, `mediapipe`, `numpy` — desktop pose detection + rendering
- `flask`, `flask-sock` — web version

For audio (compe mode + beatmap tool), add `pygame`:
```bash
pip install pygame
```

## Creating a Beatmap

For compe mode you need a beatmap JSON — a list of timestamps marking when the player
should pump. The tool plays your song and records your spacebar taps:

```bash
python -m projects.kuyacarlo.beatmap_tool --song path/to/help_me_erin.mp3 --out beatmap.json
```

**Controls:** SPACE = tap a beat, BACKSPACE = undo last, R = restart, Q = save & quit

Output format:
```json
{
  "song": "help_me_erin.mp3",
  "bpm": 170,
  "beats": [1.2, 2.4, 3.6, ...]
}
```

## Difficulty Presets (Open Mode)

| Difficulty | Timer | Vibe |
|---|---|---|
| Easy | 60s | Warm up |
| Medium | 45s | Standard |
| Hard | 30s | Sprint |
| Hell | 15s | ☠️ |

## Architecture

```
projects/kuyacarlo/
├── config.py         All tunables + difficulty presets
├── core.py           Pure game logic (no I/O, fully testable)
├── audio.py          pygame mixer playback + beatmap loading
├── beatmap_tool.py   Tap-along helper to create beat timestamps
├── renderer.py       Touhou-styled HUD, BSOD, overlays (OpenCV)
├── main.py           Desktop entry (fullscreen OpenCV + MediaPipe)
├── web.py            Flask web server (WebSocket game loop)
├── templates/
│   └── erin.html     Browser UI (fullscreen, CSS game HUD)
├── test_core.py      44 tests for all detection + game logic
└── README.md         You are here
```

The core is pure: landmarks + clock in, state dict out. Desktop and web are thin
adapters over that. This means gesture behavior is identical across both, and the
whole game logic is testable with synthetic data (no camera needed).

## Tests

```bash
python -m pytest projects/kuyacarlo/test_core.py -v
```

44 tests covering: arm tracking (hysteresis, debounce, full cycles), compe judging
(hit timing, miss detection, death conditions), open mode scoring (timer, combo,
both-arms), and the prepare phase.

## What Surprised Me

The `_game_start = 0.0` bug. In Python, `(0.0 or fallback)` evaluates to `fallback`
because `0.0` is falsy. The timer appeared to never count down. Caught by tests,
fixed with `if x is not None`.
