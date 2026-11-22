"""Player settings chosen on the Settings screen, kept on disk across sessions."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import sounds


PLAYER_2 = ("human", "hunter")   # who flies the second ship: a person, or the Campaign Hunter
MULTIPLAYER_MODES = ("compete", "coop", "race")   # taking turns; flying as a team; or racing (alone is not a choice)

COMPETITION_LENGTHS = (1, 3, 5, 10, 20)   # how many levels a campaign Level competition lasts

DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent / "player_settings.json"


class UserSettings:
    """Loads and saves the player's settings, and puts each one into effect."""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path or DEFAULT_SETTINGS_PATH
        self.sound = True
        self.multiplayer_mode = "compete"
        self.player2 = "human"
        self.competition_levels = 5
        self._load()
        sounds.set_muted(not self.sound)

    def _load(self) -> None:
        """Read saved settings; anything missing or unreadable keeps its default."""
        if not self.path.exists():
            return
        try:
            raw = json.loads(self.path.read_text())
        except (ValueError, OSError):
            return
        if isinstance(raw, dict) and isinstance(raw.get("sound"), bool):
            self.sound = raw["sound"]
        if isinstance(raw, dict) and raw.get("multiplayer_mode") in MULTIPLAYER_MODES:
            self.multiplayer_mode = raw["multiplayer_mode"]
        if isinstance(raw, dict) and raw.get("player2") in PLAYER_2:
            self.player2 = raw["player2"]
        if isinstance(raw, dict) and raw.get("competition_levels") in COMPETITION_LENGTHS:
            self.competition_levels = raw["competition_levels"]

    def _save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"sound": self.sound, "multiplayer_mode": self.multiplayer_mode,
                                              "player2": self.player2,
                                              "competition_levels": self.competition_levels}, indent=2))
        except OSError:
            pass

    def set_sound(self, on: bool) -> None:
        """Turn all game sound on or off."""
        self.sound = on
        sounds.set_muted(not on)
        self._save()

    def set_multiplayer_mode(self, mode: str) -> None:
        """Choose which multiplayer mode the MULTIPLAYER screen plays."""
        if mode in MULTIPLAYER_MODES:
            self.multiplayer_mode = mode
            self._save()

    def set_player2(self, who: str) -> None:
        """Choose who flies the second ship."""
        if who in PLAYER_2:
            self.player2 = who
            self._save()

    def set_competition_levels(self, levels: int) -> None:
        """Choose how many levels a campaign Level competition lasts."""
        if levels in COMPETITION_LENGTHS:
            self.competition_levels = levels
            self._save()
