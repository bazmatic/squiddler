"""A player: a ship, whoever flies it, and what is counted for it."""
from typing import Optional

from scoring.system import ScoringSystem


class Player:
    """One player in a level. `driver` is anything with decide(game) -> ShipCommand (the Campaign Hunter, in Demo) and
    None for a person at the keyboard or a controller. Power, ammo and fuel live on the ship; the score, wall strikes,
    shots and the time drain on `scoring`."""

    def __init__(self, ship=None, driver=None, scoring: Optional[ScoringSystem] = None):
        self.ship = ship
        self.driver = driver
        self.scoring = scoring if scoring is not None else ScoringSystem()
        self.has_moved = False      # the first move starts this player's time drain and wakes the level
        self.last_shot_time = 0     # when this player's gun last fired, in milliseconds: its fire cooldown
