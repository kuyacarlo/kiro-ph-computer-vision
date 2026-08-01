"""Help Me ERINNNNNN!! — Desktop fullscreen entry point.

Run from the repository root:
    python -m projects.kuyacarlo.main
    python -m projects.kuyacarlo.main --mode compe --song song.mp3 --beatmap beatmap.json
    python -m projects.kuyacarlo.main --mode open --difficulty hell
    python -m projects.kuyacarlo.main --no-fullscreen --camera 1

Controls:
    Q / ESC  - Quit
    R        - Restart game
    F        - Toggle fullscreen
    1-4      - Change difficulty (Easy/Medium/Hard/Hell) before game starts
"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import mediapipe as mp
import numpy as np

from .audio import AudioPlayer, Beatmap
from .config import CompeConfig, Difficulty, GameConfig, Mode, PumpConfig
from .core import ErinCore, PHASE_DEAD, PHASE_FINISHED, PHASE_PLAYING, PHASE_PREPARE
from .renderer import draw_overlay


# MediaPipe pose landmark names we need
_POSE_LANDMARK_NAMES = [
    "nose", "left_eye_inner", "left_eye", "left_eye_outer",
    "right_eye_inner", "right_eye", "right_eye_outer",
    "left_ear", "right_ear",
    "mouth_left", "mouth_right",
    "left_shoulder", "right_shoulder",
    "left_elbow", "right_elbow",
    "left_wrist", "right_wrist",
    "left_pinky", "right_pinky",
    "left_index", "right_index",
    "left_thumb", "right_thumb",
    "left_hip", "right_hip",
    "left_knee", "right_knee",
    "left_ankle", "right_ankle",
    "left_heel", "right_heel",
    "left_foot_index", "right_foot_index",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Help Me ERINNNNNN!! — Touhou fist pump counter"
    )
    parser.add_argument(
        "--mode", choices=["open", "compe"], default="open",
        help="Game mode: open (timed) or compe (rhythm)"
    )
    parser.add_argument(
        "--difficulty", choices=["easy", "medium", "hard", "hell"], default="medium",
        help="Difficulty preset for open mode timer"
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--no-fullscreen", action="store_true", help="Windowed mode")
    parser.add_argument("--no-mirror", action="store_true", help="Don't flip camera")
    parser.add_argument("--song", type=str, default="", help="Path to song file (compe mode)")
    parser.add_argument("--beatmap", type=str, default="", help="Path to beatmap JSON (compe mode)")
    return parser.parse_args()


def extract_landmarks(
    results: object,
) -> dict[str, tuple[float, float, float]] | None:
    """Extract pose landmarks from MediaPipe results into our format.

    Returns dict of {name: (x, y, visibility)} or None if no pose detected.
    """
    if not results.pose_landmarks:  # type: ignore[attr-defined]
        return None

    landmarks: dict[str, tuple[float, float, float]] = {}
    for i, lm in enumerate(results.pose_landmarks.landmark):  # type: ignore[attr-defined]
        if i < len(_POSE_LANDMARK_NAMES):
            name = _POSE_LANDMARK_NAMES[i]
            landmarks[name] = (lm.x, lm.y, lm.visibility)

    return landmarks


def main() -> None:
    args = parse_args()

    # Build config
    config = GameConfig(
        mode=Mode(args.mode),
        difficulty=Difficulty(args.difficulty),
        camera_index=args.camera,
        fullscreen=not args.no_fullscreen,
        mirror=not args.no_mirror,
        pump=PumpConfig(),
        compe=CompeConfig(song_path=args.song, beatmap_path=args.beatmap),
    )

    # Validate compe mode requirements
    if config.mode == Mode.COMPE:
        if not config.compe.song_path:
            print("❌ Compe mode requires --song <path>")
            sys.exit(1)
        if not config.compe.beatmap_path:
            print("❌ Compe mode requires --beatmap <path>")
            sys.exit(1)

    # Initialize game core
    core = ErinCore(config)

    # Load beatmap for compe mode
    audio_player: AudioPlayer | None = None
    if config.mode == Mode.COMPE:
        beatmap = Beatmap.load(config.compe.beatmap_path)
        core.set_beatmap(beatmap.beats)
        audio_player = AudioPlayer()
        audio_player.load(config.compe.song_path)

    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose  # type: ignore[attr-defined]
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,  # 0=lite, 1=full, 2=heavy. 1 is good CPU trade-off.
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    # Initialize camera
    cap = cv2.VideoCapture(config.camera_index)
    if not cap.isOpened():
        print(f"❌ Could not open camera {config.camera_index}. Try --camera 1")
        sys.exit(1)

    # Set camera resolution (optional, use default if it fails)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # Window setup
    window_name = "Help Me ERINNNNNN!!"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    if config.fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    is_fullscreen = config.fullscreen
    game_started_audio = False

    print(f"\n🎮 Help Me ERINNNNNN!! — {config.mode.value.upper()} mode")
    print(f"   Difficulty: {config.difficulty.value}")
    if config.mode == Mode.COMPE:
        print(f"   Song: {config.compe.song_path}")
    print(f"   Camera: {config.camera_index}")
    print(f"   Controls: Q=quit, R=restart, F=fullscreen, 1-4=difficulty\n")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            if config.mirror:
                frame = cv2.flip(frame, 1)

            now = time.monotonic()

            # Run pose detection
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(rgb)
            landmarks = extract_landmarks(results)

            # Update game state
            state = core.update(landmarks, now)

            # Start compe audio when game begins playing
            if (
                config.mode == Mode.COMPE
                and audio_player is not None
                and core.phase == PHASE_PLAYING
                and not game_started_audio
            ):
                audio_player.play()
                game_started_audio = True

            # Draw overlay
            frame = draw_overlay(frame, state, now)

            # Show frame
            cv2.imshow(window_name, frame)

            # Handle input
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27):  # Q or ESC
                break
            elif key == ord("r"):  # Restart
                core.reset()
                game_started_audio = False
                if audio_player:
                    audio_player.stop()
            elif key == ord("f"):  # Toggle fullscreen
                is_fullscreen = not is_fullscreen
                if is_fullscreen:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN)
                else:
                    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_NORMAL)
            elif key == ord("1"):
                config.difficulty = Difficulty.EASY
                core.config.difficulty = Difficulty.EASY
            elif key == ord("2"):
                config.difficulty = Difficulty.MEDIUM
                core.config.difficulty = Difficulty.MEDIUM
            elif key == ord("3"):
                config.difficulty = Difficulty.HARD
                core.config.difficulty = Difficulty.HARD
            elif key == ord("4"):
                config.difficulty = Difficulty.HELL
                core.config.difficulty = Difficulty.HELL

    finally:
        cap.release()
        cv2.destroyAllWindows()
        pose.close()
        if audio_player:
            audio_player.cleanup()


if __name__ == "__main__":
    main()
