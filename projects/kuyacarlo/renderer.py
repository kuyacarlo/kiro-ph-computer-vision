"""Touhou-styled renderer for Help Me ERINNNNNN!!

Draws the game HUD overlay on camera frames. Handles:
- Big glowing pump counter
- Arm height indicators
- Timer / combo display
- "ERINNNNNN!!" flash on pump
- BSOD death screen (compe mode)
- Prepare phase progress bar

Color palette: Eirin's colors (blue/silver/red) with Reimu red for death.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .core import PHASE_DEAD, PHASE_FINISHED, PHASE_PLAYING, PHASE_PREPARE


# --- Color Palette (BGR) ---

class Colors:
    """Eirin/Touhou-inspired color palette (BGR format)."""

    # Primary
    EIRIN_BLUE = (200, 140, 60)      # Eirin's dress blue
    EIRIN_SILVER = (210, 200, 190)   # Eirin's hair silver
    REIMU_RED = (60, 60, 220)        # Reimu red for danger

    # UI
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    DIM = (120, 110, 100)
    GRID = (50, 40, 35)

    # Accents
    PUMP_FLASH = (180, 255, 255)     # Yellow-white flash on pump
    COMBO_GOLD = (50, 200, 255)      # Gold for combo
    TIMER_CYAN = (230, 200, 100)     # Cyan for timer
    HIT_GREEN = (100, 255, 100)      # Green for successful hit
    MISS_RED = (80, 80, 255)         # Red for miss/death

    # BSOD
    BSOD_BLUE = (180, 0, 0)         # Classic Windows BSOD blue (dark)
    BSOD_TEXT = (255, 255, 255)      # White text on BSOD


FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_BOLD = cv2.FONT_HERSHEY_DUPLEX


def _glow_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 2,
    blur: int = 21,
    intensity: float = 0.6,
) -> None:
    """Draw text with a glow effect."""
    h, w = frame.shape[:2]
    layer = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(layer, text, origin, FONT_BOLD, scale, color, thickness + 2, cv2.LINE_AA)
    blurred = cv2.GaussianBlur(layer, (blur, blur), 0)
    mask = blurred.astype(np.float32) * intensity / 255.0
    frame[:] = np.clip(frame.astype(np.float32) + blurred * intensity, 0, 255).astype(np.uint8)
    cv2.putText(frame, text, origin, FONT_BOLD, scale, color, thickness, cv2.LINE_AA)


def _panel(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    alpha: float = 0.6,
    color: tuple[int, int, int] = (20, 15, 10),
) -> None:
    """Draw a semi-transparent panel."""
    overlay = frame.copy()
    cv2.rectangle(overlay, top_left, bottom_right, color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def _progress_bar(
    frame: np.ndarray,
    origin: tuple[int, int],
    size: tuple[int, int],
    progress: float,
    color: tuple[int, int, int] = Colors.EIRIN_BLUE,
    bg_color: tuple[int, int, int] = Colors.GRID,
) -> None:
    """Draw a progress bar."""
    x, y = origin
    w, h = size
    # Background
    cv2.rectangle(frame, (x, y), (x + w, y + h), bg_color, -1)
    # Fill
    fill_w = int(w * min(1.0, max(0.0, progress)))
    if fill_w > 0:
        cv2.rectangle(frame, (x, y), (x + fill_w, y + h), color, -1)
    # Border
    cv2.rectangle(frame, (x, y), (x + w, y + h), Colors.DIM, 1)


def draw_overlay(frame: np.ndarray, state: dict[str, Any], now: float) -> np.ndarray:
    """Main render function. Draws appropriate overlay based on game phase."""
    phase = state["phase"]

    if phase == PHASE_DEAD:
        return draw_bsod(frame, state)
    elif phase == PHASE_FINISHED:
        return draw_finished(frame, state)
    elif phase == PHASE_PREPARE:
        return draw_prepare(frame, state)
    else:
        return draw_playing(frame, state, now)


def draw_prepare(frame: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    """Draw the prepare phase: waiting for pose lock."""
    h, w = frame.shape[:2]

    # Dim the frame slightly
    frame = (frame * 0.7).astype(np.uint8)

    # Title
    _glow_text(frame, "HELP ME ERINNNNNN!!", (w // 2 - 280, h // 3), 1.2,
               Colors.EIRIN_BLUE, thickness=3, blur=31)

    # Mode indicator
    mode_text = f"MODE: {state['mode'].upper()}"
    cv2.putText(frame, mode_text, (w // 2 - 80, h // 3 + 50),
                FONT, 0.7, Colors.EIRIN_SILVER, 2, cv2.LINE_AA)

    # Hint
    hint = state["hint"]
    cv2.putText(frame, hint, (w // 2 - 100, h // 2),
                FONT, 0.8, Colors.PUMP_FLASH, 2, cv2.LINE_AA)

    # Progress bar
    progress = state["prepare_progress"]
    bar_w = 300
    bar_x = w // 2 - bar_w // 2
    _progress_bar(frame, (bar_x, h // 2 + 30), (bar_w, 16), progress, Colors.EIRIN_BLUE)

    # Prepare text
    if progress < 1.0:
        pct = int(progress * 100)
        cv2.putText(frame, f"LOCKING POSE... {pct}%", (bar_x, h // 2 + 70),
                    FONT, 0.5, Colors.DIM, 1, cv2.LINE_AA)

    return frame


def draw_playing(frame: np.ndarray, state: dict[str, Any], now: float) -> np.ndarray:
    """Draw the main gameplay HUD."""
    h, w = frame.shape[:2]

    # --- Score (big, top-right) ---
    score_text = str(state["score"])
    (tw, th), _ = cv2.getTextSize(score_text, FONT_BOLD, 4.0, 8)
    score_x = w - tw - 40
    score_y = 100
    _glow_text(frame, score_text, (score_x, score_y), 4.0,
               Colors.PUMP_FLASH, thickness=6, blur=41)
    cv2.putText(frame, "PUMPS", (score_x, score_y + 35),
                FONT, 0.6, Colors.DIM, 1, cv2.LINE_AA)

    # --- Timer (open mode) or progress (compe mode) ---
    if state["mode"] == "open":
        remaining = state["time_remaining"]
        timer_color = Colors.TIMER_CYAN if remaining > 5 else Colors.REIMU_RED
        timer_text = f"{remaining:.1f}s"
        cv2.putText(frame, timer_text, (30, 60), FONT_BOLD, 1.8,
                    timer_color, 3, cv2.LINE_AA)
        cv2.putText(frame, "TIME", (30, 85), FONT, 0.5, Colors.DIM, 1, cv2.LINE_AA)
    else:
        # Compe mode — show hits/total
        hits_text = f"{state['compe_hits']}/{state['compe_total']}"
        cv2.putText(frame, hits_text, (30, 60), FONT_BOLD, 1.4,
                    Colors.HIT_GREEN, 2, cv2.LINE_AA)
        cv2.putText(frame, "HITS", (30, 85), FONT, 0.5, Colors.DIM, 1, cv2.LINE_AA)

        # Progress bar
        _progress_bar(frame, (30, 95), (200, 8), state["compe_progress"], Colors.EIRIN_BLUE)

    # --- Combo ---
    if state["combo"] > 1:
        combo_text = f"{state['combo']} COMBO"
        combo_x = w // 2 - 80
        combo_y = h - 60
        _glow_text(frame, combo_text, (combo_x, combo_y), 1.0,
                   Colors.COMBO_GOLD, thickness=2, blur=21)

    # --- Arm indicators ---
    _draw_arm_indicator(frame, state["left_arm"], "LEFT", (30, h - 180))
    _draw_arm_indicator(frame, state["right_arm"], "RIGHT", (w - 160, h - 180))

    # --- Pump flash ---
    if state["last_pump_at"] is not None:
        flash_age = now - state["last_pump_at"]
        if flash_age < 0.3:
            alpha = 1.0 - (flash_age / 0.3)
            # Flash "ERIN!!" text
            flash_text = "ERINNNNNN!!"
            (ftw, _), _ = cv2.getTextSize(flash_text, FONT_BOLD, 1.5, 3)
            fx = w // 2 - ftw // 2
            fy = h // 2
            color = tuple(int(c * alpha) for c in Colors.PUMP_FLASH)
            cv2.putText(frame, flash_text, (fx, fy), FONT_BOLD, 1.5,
                        color, 3, cv2.LINE_AA)

    # --- Tempo ---
    ppm = state["pumps_per_minute"]
    if ppm:
        cv2.putText(frame, f"{ppm:.0f}/min", (30, h - 30),
                    FONT, 0.5, Colors.DIM, 1, cv2.LINE_AA)

    # --- Hint (if pose lost) ---
    if not state["pose_visible"]:
        cv2.putText(frame, state["hint"], (w // 2 - 120, h // 2),
                    FONT, 0.8, Colors.REIMU_RED, 2, cv2.LINE_AA)

    return frame


def _draw_arm_indicator(
    frame: np.ndarray,
    arm_state: dict[str, Any],
    label: str,
    origin: tuple[int, int],
) -> None:
    """Draw a single arm's status indicator."""
    x, y = origin
    w_panel, h_panel = 130, 90

    is_up = arm_state["phase"] == "up"
    border_color = Colors.PUMP_FLASH if is_up else Colors.DIM

    _panel(frame, (x, y), (x + w_panel, y + h_panel), alpha=0.6)
    cv2.rectangle(frame, (x, y), (x + w_panel, y + h_panel), border_color, 1)

    # Label
    cv2.putText(frame, label, (x + 10, y + 22), FONT, 0.5, Colors.EIRIN_SILVER, 1, cv2.LINE_AA)

    # Status
    status = "UP ✊" if is_up else "DOWN"
    color = Colors.PUMP_FLASH if is_up else Colors.DIM
    cv2.putText(frame, status, (x + 10, y + 50), FONT, 0.6, color, 2, cv2.LINE_AA)

    # Height bar
    height = arm_state["height"]
    bar_x = x + 10
    bar_y = y + 62
    bar_w = w_panel - 20
    bar_h = 12
    fill = max(0.0, min(1.0, height / 0.5))  # Normalize to 0..1
    _progress_bar(frame, (bar_x, bar_y), (bar_w, bar_h), fill,
                  Colors.EIRIN_BLUE if is_up else Colors.GRID)


def draw_finished(frame: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    """Draw the game-over screen (open mode — time's up)."""
    h, w = frame.shape[:2]

    # Dim background heavily
    frame = (frame * 0.3).astype(np.uint8)

    # Results panel
    panel_x = w // 4
    panel_y = h // 4
    panel_w = w // 2
    panel_h = h // 2
    _panel(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
           alpha=0.85, color=(30, 20, 15))
    cv2.rectangle(frame, (panel_x, panel_y), (panel_x + panel_w, panel_y + panel_h),
                  Colors.EIRIN_BLUE, 2)

    # Title
    _glow_text(frame, "TIME'S UP!", (panel_x + panel_w // 2 - 120, panel_y + 60),
               1.2, Colors.PUMP_FLASH, thickness=3)

    # Score
    score_text = str(state["score"])
    (tw, _), _ = cv2.getTextSize(score_text, FONT_BOLD, 3.0, 6)
    _glow_text(frame, score_text, (panel_x + panel_w // 2 - tw // 2, panel_y + 150),
               3.0, Colors.COMBO_GOLD, thickness=5)
    cv2.putText(frame, "TOTAL PUMPS", (panel_x + panel_w // 2 - 70, panel_y + 180),
                FONT, 0.6, Colors.DIM, 1, cv2.LINE_AA)

    # Stats
    stats_y = panel_y + 220
    stats = [
        f"Max Combo: {state['max_combo']}",
        f"Left Arm:  {state['left_arm']['pumps']}",
        f"Right Arm: {state['right_arm']['pumps']}",
    ]
    if state["pumps_per_minute"]:
        stats.append(f"Tempo:     {state['pumps_per_minute']:.0f}/min")

    for i, line in enumerate(stats):
        cv2.putText(frame, line, (panel_x + 40, stats_y + i * 30),
                    FONT, 0.6, Colors.EIRIN_SILVER, 1, cv2.LINE_AA)

    # Restart hint
    cv2.putText(frame, "Press R to restart | Q to quit",
                (panel_x + 40, panel_y + panel_h - 30),
                FONT, 0.5, Colors.DIM, 1, cv2.LINE_AA)

    return frame


def draw_bsod(frame: np.ndarray, state: dict[str, Any]) -> np.ndarray:
    """Draw the BSOD death screen (compe mode).

    A loving recreation of the Windows XP blue screen of death,
    appropriately themed for an Eirin-related incident.
    """
    h, w = frame.shape[:2]

    # Full blue screen
    frame[:] = Colors.BSOD_BLUE

    # Calculate text positioning
    margin_x = int(w * 0.08)
    line_height = int(h * 0.04)
    y = int(h * 0.1)

    def bsod_line(text: str, y_pos: int, bold: bool = False) -> int:
        font = FONT_BOLD if bold else FONT
        scale = 0.55 if not bold else 0.6
        cv2.putText(frame, text, (margin_x, y_pos), font, scale,
                    Colors.BSOD_TEXT, 1, cv2.LINE_AA)
        return y_pos + line_height

    # Header - white on blue with inverted section
    header = " Eirin "
    (hw, hh), _ = cv2.getTextSize(header, FONT_BOLD, 0.7, 2)
    header_x = margin_x
    header_y = y
    # Draw inverted header background
    cv2.rectangle(frame, (header_x - 4, header_y - hh - 4),
                  (header_x + hw + 4, header_y + 4), Colors.BSOD_TEXT, -1)
    cv2.putText(frame, header, (header_x, header_y), FONT_BOLD, 0.7,
                Colors.BSOD_BLUE, 2, cv2.LINE_AA)
    y += line_height * 2

    # Error text
    y = bsod_line("A problem has been detected and Eirin has shut down your system to", y)
    y = bsod_line("prevent further damage to your scoring record.", y)
    y += line_height

    y = bsod_line("ERIN_PUMP_TIMING_VIOLATION", y, bold=True)
    y += line_height

    y = bsod_line("If this is the first time you've seen this Stop error screen,", y)
    y = bsod_line("restart your performance. If this screen appears again, follow", y)
    y = bsod_line("these steps:", y)
    y += line_height

    y = bsod_line("Check to make sure you are pumping ON THE BEAT.", y)
    y = bsod_line("If this is a new song, ask the beatmap creator for any", y)
    y = bsod_line("timing updates you might need.", y)
    y += line_height

    y = bsod_line("If problems continue, disable or remove any unnecessary", y)
    y = bsod_line("arm movements. Try pumping with proper form to resolve", y)
    y = bsod_line("timing conflicts.", y)
    y += line_height

    y = bsod_line("Technical information:", y)
    y += int(line_height * 0.5)

    # Death reason
    reason = state.get("death_reason", "UNKNOWN_ERROR")
    y = bsod_line(f"*** STOP: 0x000000E4 ({reason})", y)
    y += line_height

    # Score at death
    y = bsod_line(f"*** help_me_erinnnnnn.sys - Score: {state['score']}  "
                  f"Combo: {state['max_combo']}  "
                  f"Hits: {state['compe_hits']}/{state['compe_total']}", y)
    y += line_height * 2

    # Fake progress
    y = bsod_line("Collecting performance data for shrine maiden review...", y)

    # Restart hint (subtle, at the bottom)
    cv2.putText(frame, "Press R to retry | Q to quit",
                (margin_x, h - int(h * 0.06)), FONT, 0.5,
                Colors.BSOD_TEXT, 1, cv2.LINE_AA)

    return frame
