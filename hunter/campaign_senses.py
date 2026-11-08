"""The Campaign Hunter's senses: what the allied hunter perceives, from the player ship's seat, plus where the
exit is and where an Ally beacon lies (the two things the level asks of the player that a hunter never needed), and
which pickup weapon the ship holds and how long its pickup shield has left."""
import math
import numpy as np
import config
from hunter.senses import Senses, blocked
from entities.jev_beacon import JevBeacon

from hunter.campaign_layout import CAMPAIGN_FEATURE_NAMES, CAMPAIGN_FEATURE_NAMES_V1, HELD_WEAPON_IDS  # noqa: F401

_INDEX = {name: i for i, name in enumerate(CAMPAIGN_FEATURE_NAMES)}


class ShipBody:
    """The player ship, presenting what the hunter's senses read from a hunter. Everything the ship has
    itself (position, motion, heading, guns, weapon) is the ship's; the rest is read from the game."""

    def __init__(self, ship, game, settings):
        self.ship, self.game, self.settings = ship, game, settings
        self._hit = None

    def __getattr__(self, name):
        return getattr(self.ship, name)

    @property
    def health(self):
        """The share of the level's potential score still to be had, on the hunter's health scale."""
        game, scoring = self.game, self.game.scoring
        potential = scoring.calculate_current_potential_score(game.clock(), self.ship.fuel, self.ship.ammo)
        return potential['potential_score'] / scoring.calculate_max_possible_score() * self.settings.health

    @property
    def upgrade_remaining(self):
        return self.ship.gun_upgrade_timer / config.FPS

    def _held(self):
        weapon = self.ship.weapon
        return weapon if weapon is not None and weapon.weapon_id != 'basic' else None

    @property
    def held_weapon_id(self):
        """The pickup weapon in hand, or None for the plain gun (and once a pickup's ammo has run out)."""
        weapon = self._held()
        return weapon.weapon_id if weapon is not None else None

    @property
    def shield_remaining(self):
        """The share of a pickup shield's run still to go: 1 just picked up, 0 with none."""
        full = config.SETTINGS.powerups.shieldDurationSeconds * config.FPS
        return min(1.0, max(0.0, self.ship.powerup_shield_timer / full)) if full > 0 else 0.0

    @property
    def fire_interval(self):
        """Seconds between shots: the held weapon's interval (kept in simulation frames), else the gun's."""
        weapon = self._held()
        if weapon is not None:
            return weapon.view.interval / config.FPS
        from game_handlers.fire_rate_calculator import calculate_fire_cooldown
        return calculate_fire_cooldown(self.ship) / 1000

    @property
    def fire_remaining(self):
        """Seconds until the trigger is next heard: the held weapon's cooldown (kept in frames), else the gun's."""
        weapon = self._held()
        if weapon is not None:
            return weapon.view.cooldown / config.FPS
        ready_at = getattr(self.game, 'last_shot_time', 0) + self.fire_interval * 1000
        return max(0.0, ready_at - self.game.clock_ms()) / 1000

    @property
    def burst_remaining(self):
        return 0       # the ship fires whole volleys, not bursts

    def take_hit(self):
        return None    # the ship keeps no record of knocks


class CampaignSurroundings:
    """A level's surroundings as the Campaign Hunter sees them from the ship: every enemy, every projectile, the
    powerups except Jev beacons, and the allied hunter. The game stays reachable for the exit and the beacon."""

    def __init__(self, game):
        self.game = game

    @property
    def maze(self):
        return self.game.maze

    def enemies(self):
        return self.game.roster.all()

    def threats(self):
        return self.game.projectiles

    def pickups(self):
        return [c for c in self.game.powerup_crystals if c.active and not isinstance(c, JevBeacon)]

    def friend(self):
        return self.game.hunter


class CampaignSenses(Senses):
    def perceive(self, body, surroundings, last_controls, now):
        """The hunter's vector for `body`, then the exit compass, `exit_open`, the beacon compass, the held weapon
        and the pickup shield."""
        base = super().perceive(body, surroundings, last_controls, now)
        out = np.zeros(len(CAMPAIGN_FEATURE_NAMES), dtype=np.float32)
        out[:len(base)] = base
        game, maze = surroundings.game, surroundings.maze
        origin = (body.x, body.y)
        h = math.radians(body.angle)
        width = max(maze.grid_width * maze.cell_size_x, 1.0)
        self._exit_compass(out, origin, h, self.settings.sensor_range(maze), maze, game.maze.exit.get_pos(), width)
        out[_INDEX['exit_open']] = float(game.exit_opened)
        # Straight through walls, like the crystal compass: an Ally beacon is found without being seen.
        beacons = [c for c in game.powerup_crystals if c.active and isinstance(c, JevBeacon)]
        if beacons:
            nearest = min(beacons, key=lambda c: math.dist(origin, (c.x, c.y)))
            rel = math.atan2(nearest.y - origin[1], nearest.x - origin[0]) - h
            out[_INDEX['beacon_compass_present']] = 1.0
            out[_INDEX['beacon_compass_sin']] = math.sin(rel)
            out[_INDEX['beacon_compass_cos']] = math.cos(rel)
            out[_INDEX['beacon_compass_dist']] = min(1.0, math.dist(origin, (nearest.x, nearest.y)) / width)
        held = getattr(body, 'held_weapon_id', None)
        if held in HELD_WEAPON_IDS:
            out[_INDEX[f'weapon_{held}']] = 1.0
        out[_INDEX['shield_remaining']] = getattr(body, 'shield_remaining', 0.0)
        return out

    def _exit_compass(self, out, origin, h, reach, maze, exit_pos, width):
        """Point the way to the exit as the enemy compass does: at the furthest point of the cell route in plain
        sight, with the route's length in maze widths. An exit no route reaches is pointed at directly."""
        walls = [w for w in list(maze.spatial_grid.get_nearby_walls(origin, reach)) + list(getattr(maze, 'gate_segments', ()))
                 if w.active]
        starts = np.array([w.start for w in walls], dtype=float).reshape(-1, 2)
        ends = np.array([w.end for w in walls], dtype=float).reshape(-1, 2)
        found = self._routes.route(maze, origin, [exit_pos])
        if found is None:
            waypoint, length = exit_pos, math.dist(origin, exit_pos)
        else:
            points = [tuple(exit_pos)] + found[1]                      # from the exit back toward the ship
            hidden = blocked(origin, points, starts, ends)
            k = next((i for i, (p, hid) in enumerate(zip(points, hidden)) if not hid and math.dist(origin, p) <= reach),
                     len(points) - 1)
            waypoint = points[k]
            legs = [tuple(origin)] + points[k::-1]
            length = sum(math.dist(a, b) for a, b in zip(legs, legs[1:]))
        rel = math.atan2(waypoint[1] - origin[1], waypoint[0] - origin[0]) - h
        out[_INDEX['exit_compass_present']] = 1.0
        out[_INDEX['exit_compass_sin']] = math.sin(rel)
        out[_INDEX['exit_compass_cos']] = math.cos(rel)
        out[_INDEX['exit_compass_dist']] = min(1.0, length / width)
