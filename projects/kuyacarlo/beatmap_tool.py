"""Beatmap creation tool — tap along to a song to mark beat timestamps.

Usage:
    python -m projects.kuyacarlo.beatmap_tool --song path/to/song.mp3 --out beatmap.json

Controls:
    SPACE  - Tap to mark a beat
    BACKSPACE - Remove last beat
    R - Restart from beginning
    Q / ESC - Save and quit

The tool plays the song and records the timestamps where you press SPACE.
After recording, it saves a JSON beatmap file that compe mode can load.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tap along to a song to create a beatmap for Help Me ERINNNNNN!!"
    )
    parser.add_argument(
        "--song", required=True, help="Path to the audio file (mp3, ogg, wav)"
    )
    parser.add_argument(
        "--out", default="beatmap.json", help="Output beatmap JSON path"
    )
    parser.add_argument(
        "--bpm", type=float, default=0.0, help="Optional: song BPM for reference"
    )
    return parser.parse_args()


def run_terminal_mode(song_path: str, output_path: str, bpm: float) -> None:
    """Terminal-based tap tool using pygame for audio + keyboard events."""
    try:
        import pygame
    except ImportError:
        print("pygame is required. Install with: pip install pygame")
        sys.exit(1)

    pygame.init()
    # Need a display surface for keyboard events
    screen = pygame.display.set_mode((640, 240))
    pygame.display.set_caption("Help Me ERINNNNNN!! — Beatmap Tool")

    # Load and play the song
    pygame.mixer.music.load(song_path)

    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    beats: list[float] = []
    playing = False
    start_time: float = 0.0

    def draw() -> None:
        screen.fill((20, 10, 40))
        elapsed = time.monotonic() - start_time if playing else 0.0

        # Title
        title = font.render("BEATMAP TOOL — Help Me ERINNNNNN!!", True, (200, 200, 255))
        screen.blit(title, (20, 20))

        # Status
        status_text = f"Time: {elapsed:.2f}s | Beats: {len(beats)}"
        if beats:
            status_text += f" | Last: {beats[-1]:.2f}s"
        status = font.render(status_text, True, (180, 255, 180))
        screen.blit(status, (20, 70))

        # Controls
        controls = [
            "SPACE = tap beat | BACKSPACE = undo | R = restart | Q = save & quit",
            f"Song: {Path(song_path).name}",
        ]
        if bpm > 0:
            controls.append(f"BPM: {bpm}")
        for i, line in enumerate(controls):
            text = small_font.render(line, True, (150, 150, 150))
            screen.blit(text, (20, 130 + i * 28))

        # Beat visualization (last 10 beats as dots)
        if beats:
            y = 210
            for i, b in enumerate(beats[-20:]):
                x = 20 + i * 30
                pygame.draw.circle(screen, (255, 100, 100), (x, y), 8)

        pygame.display.flip()

    # Start playback
    print(f"\n🎵 Loading: {song_path}")
    print("Press SPACE to start, then tap SPACE on each beat.\n")

    clock = pygame.time.Clock()
    running = True
    waiting_to_start = True

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_q, pygame.K_ESCAPE):
                    running = False
                elif event.key == pygame.K_SPACE:
                    if waiting_to_start:
                        # Start the song
                        pygame.mixer.music.play()
                        start_time = time.monotonic()
                        playing = True
                        waiting_to_start = False
                        print("▶️  Playing! Tap SPACE on each beat...")
                    elif playing:
                        # Record a beat
                        beat_time = time.monotonic() - start_time
                        beats.append(round(beat_time, 3))
                        print(f"  ✊ Beat #{len(beats)} at {beat_time:.3f}s")
                elif event.key == pygame.K_BACKSPACE:
                    if beats:
                        removed = beats.pop()
                        print(f"  ↩️  Removed beat at {removed:.3f}s")
                elif event.key == pygame.K_r:
                    # Restart
                    pygame.mixer.music.stop()
                    beats.clear()
                    playing = False
                    waiting_to_start = True
                    print("🔄 Restart — press SPACE to begin again")

        # Check if song ended
        if playing and not pygame.mixer.music.get_busy():
            playing = False
            print(f"\n🏁 Song finished! Recorded {len(beats)} beats.")
            print("Press Q to save, or R to redo.")

        draw()
        clock.tick(60)

    pygame.quit()

    # Save beatmap
    if beats:
        beatmap = {
            "song": str(Path(song_path).name),
            "bpm": bpm,
            "beats": sorted(beats),
        }
        output = Path(output_path)
        with open(output, "w") as f:
            json.dump(beatmap, f, indent=2)
        print(f"\n✅ Saved {len(beats)} beats to {output}")
        print(f"   Use with: --beatmap {output}")
    else:
        print("\n⚠️  No beats recorded. Nothing saved.")


def main() -> None:
    args = parse_args()

    song = Path(args.song)
    if not song.exists():
        print(f"❌ Song file not found: {song}")
        sys.exit(1)

    run_terminal_mode(args.song, args.out, args.bpm)


if __name__ == "__main__":
    main()
