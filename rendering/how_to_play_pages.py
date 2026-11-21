"""The copy for the HOW TO PLAY screen. Every figure is read from the settings, so it can't drift from the rules."""
from typing import List, NamedTuple

import config
from entities.powerup_crystal import PowerupCrystal
from rendering.settings_menu import CONTROL_GROUPS


class Page(NamedTuple):
    title: str
    lines: List[str]


def _next_weapon_label(device: str) -> str:
    rows = dict(CONTROL_GROUPS)[device]
    return next(label for label, action in rows if action == "Next weapon")


def build_pages(controller: bool) -> List[Page]:
    """The four pages, with the controls shown for the keyboard or for a controller."""
    device = "CONTROLLER" if controller else "KEYBOARD"
    powerups = config.SETTINGS.powerups
    crystal_seconds = PowerupCrystal.LIFETIME_FRAMES / config.FPS
    controls = [f"{label}: {action}" for label, action in dict(CONTROL_GROUPS)[device]]
    return [
        Page("GOAL AND CONTROLS", [
            "Fly to the green exit portal before your power runs out.",
            "Most levels keep the portal locked until the mission announced at the start is done.",
            "Your ship keeps its momentum, so plan your braking.",
            *controls,
        ]),
        Page("POWER", [
            "The POWER gauge is the level's score counting down.",
            "Time, thrust, shots, wall bumps and enemy bullet hits all drain it.",
            f"Destroy an enemy: +{config.ENEMY_DESTRUCTION_BONUS}.",
            f"Collect a crystal: +{config.POWERUP_CRYSTAL_BONUS}.",
            "At zero the level is lost.",
            f"In the last {config.SETTINGS.game.criticalWarningThreshold} points the screen edge pulses red "
            "and a tick sounds, faster as power runs out.",
        ]),
        Page("CRYSTALS AND WEAPONS", [
            f"Enemies sometimes drop a crystal. Fly close to pull it in; it vanishes after {crystal_seconds:g} seconds.",
            f"A crystal adds a firepower level for {config.POWERUP_DURATION_SECONDS} seconds: faster fire, then a spread.",
            f"It also gives {config.POWERUP_AMMO_REFILL} ammo, and shots cost "
            f"{config.POWERUP_AMMO_COST_MULTIPLIER:g} of normal while it lasts.",
            f"Weapon crystals add and select a new weapon. {_next_weapon_label(device)} cycles through them.",
            f"A shield crystal makes you invulnerable for {powerups.shieldDurationSeconds:g} seconds.",
        ]),
        Page("STARS AND ENEMIES", [
            "Reaching the exit earns a star for each goal met: no hits, half the enemies, every enemy, "
            "every crystal, and finishing within par time.",
            "Eggs hatch Squidlets, Urchins and Crabs shoot, Jellies and Flockers hunt in groups, "
            "and Wallcrawlers walk the walls.",
            "Every sixth level is a boss arena.",
        ]),
    ]
