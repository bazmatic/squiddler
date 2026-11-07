"""The Campaign Hunter: a neural pilot that flies the player ship."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ShipCommand:
    """What the pilot does this frame: turn -1 left, 0 straight, +1 right."""
    turn: int
    thrust: bool
    fire: bool
