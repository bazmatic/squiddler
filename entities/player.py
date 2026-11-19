"""A player: a ship, whoever flies it, and what is counted for it."""
from typing import Optional, Tuple

from entities.command_recorder import CommandRecorder
from scoring.system import ScoringSystem


ANY_CONTROLLER = -1     # a player whose ship every connected controller flies (turns passed between players)

PLAYER_2_COLOR = (255, 170, 60)
PLAYER_2_PALETTE = ((255, 220, 160), (230, 110, 30), PLAYER_2_COLOR)   # nose, rear, glow


class Player:
    """One player in a level. `driver` is anything with decide(game) -> ShipCommand (the Campaign Hunter, in Demo) and
    None for a person at the keyboard or a controller. Power, ammo and fuel live on the ship; the score, wall strikes,
    shots and the time drain on `scoring`."""

    def __init__(self, ship=None, driver=None, scoring: Optional[ScoringSystem] = None):
        self.ship = ship
        self.driver = driver
        self.scoring = scoring if scoring is not None else ScoringSystem()
        self.recorder = CommandRecorder()   # this player's commands, which a Replay enemy copies
        self.has_moved = False      # the first move starts this player's time drain and wakes the level
        self.last_shot_time = 0     # when this player's gun last fired, in milliseconds: its fire cooldown
        # What flies the ship: the keyboard (WASD/arrows/space) and/or one controller, by its index (None: no controller)
        self.keyboard = True
        self.controller: Optional[int] = 0
        self.color: Optional[Tuple[int, int, int]] = None   # None: the ship's usual colours
        self.dropped_out = False    # ran out of power: out until the next level
        self.through = False        # went through the portal: out of play, power frozen
        self.final_power: Optional[float] = None   # the power it took through the portal

    @property
    def flying(self) -> bool:
        return not self.dropped_out and not self.through
