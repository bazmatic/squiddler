"""What the hunter may perceive at the moment it looks.

Surroundings are asked, not stored: every call answers from the world as it is now, so a pilot never
sees an enemy that died earlier in the frame, and nothing is gathered on the frames it does not look.

A surroundings answers:

- `maze`: the maze the hunter flies in
- `enemies()`: every enemy it could perceive, active or not (the observers skip inactive ones)
- `threats()`: the projectiles that may harm it
- `pickups()`: the powerup crystals it may collect
- `friend()`: the player when the hunter may see them, else None
"""
from entities.jev_beacon import JevBeacon


class GameSurroundings:
    """A level's surroundings: every enemy, every projectile, the powerups except Jev beacons (the
    player's to collect), and the player when the hunter may see them."""

    def __init__(self, game):
        self._game = game

    @property
    def maze(self):
        return self._game.maze

    def enemies(self):
        return self._game.roster.all()

    def threats(self):
        return self._game.projectiles

    def pickups(self):
        return [c for c in self._game.powerup_crystals if not isinstance(c, JevBeacon)]

    def friend(self):
        game = self._game
        if not game._hunter_sees_player():
            return None
        nearest = getattr(game, 'nearest_ship', None)   # a training scene's stand-in has just the one ship
        return nearest(game.hunter) if nearest is not None else game.ship


class ProtagonistSurroundings:
    """What an enemy pilot perceives: its protagonist and friendly shots, and nothing else.

    The protagonist is whoever the game names (`Game.protagonist()`): the player's ship in normal play,
    the Hunter in Observer mode or the lab. No other enemy, pickup or friend is shown.
    """

    def __init__(self, game):
        self._game = game

    @property
    def maze(self):
        return self._game.maze

    def enemies(self):
        target = self._game.protagonist()
        return [target] if target is not None and target.active else []

    def threats(self):
        return [p for p in self._game.projectiles if not p.is_enemy]

    def pickups(self):
        return []

    def friend(self):
        return None


class RosterSurroundings:
    """What a friendly pilot perceives inside an Encounter, with no Game to ask (a training scene): every
    enemy-allegiance creature on the Encounter's roster and the enemies' shots. Not itself, no pickups, no friend."""

    def __init__(self, encounter, creature):
        self._encounter, self._creature = encounter, creature

    @property
    def maze(self):
        return self._encounter.maze

    def enemies(self):
        return [c for c in self._encounter.roster.all()
                if c is not self._creature and getattr(c, 'allegiance', 'enemy') == 'enemy']

    def threats(self):
        return [p for p in self._encounter.world.projectiles if p.is_enemy]

    def pickups(self):
        return []

    def friend(self):
        return None
