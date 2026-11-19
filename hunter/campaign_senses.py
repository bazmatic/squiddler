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

    def __init__(self, ship, game, settings, hit_feedback=True, player=None):
        self.ship, self.game, self.settings = ship, game, settings
        self.player = player if player is not None else game.players[0]   # whose power and gun clock these are
        self.hit_feedback = hit_feedback
        self._hit = None

    def __getattr__(self, name):
        return getattr(self.ship, name)

    @property
    def health(self):
        """How far the level's potential score is from running out (the ship's nearness to death), on the hunter's
        health scale. Kills bank more than the level's maximum, but that is not safety: the share stops at 1, so the
        sense says only how close to scoring out the ship is, not how well it has done."""
        game, scoring = self.game, self.player.scoring
        potential = scoring.calculate_current_potential_score(game.clock(), self.ship.fuel, self.ship.ammo)
        share = potential['potential_score'] / scoring.calculate_max_possible_score()
        return min(share, 1.0) * self.settings.health

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
        ready_at = self.player.last_shot_time + self.fire_interval * 1000
        return max(0.0, ready_at - self.game.clock_ms()) / 1000

    @property
    def burst_remaining(self):
        return 0       # the ship fires whole volleys, not bursts

    def take_hit(self):
        hit = self.ship.take_hit()
        return hit if self.hit_feedback else None


class CampaignSurroundings:
    """A level's surroundings as the Campaign Hunter sees them from the ship: every enemy, every projectile, the
    powerups except Jev beacons, and the allied hunter. The game stays reachable for the exit and the beacon."""

    def __init__(self, game, player=None):
        self.game = game
        self.player = player

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
        """The ally to keep clear of and not shoot: the allied hunter or, in co-op, the other player's ship, whichever
        is nearer and still flying."""
        game, me = self.game, self.player
        if me is None:
            return game.hunter      # no player named: a scene with just the one ship
        others = [p.ship for p in game.players if p is not me and p.flying and p.ship is not None and p.ship.active]
        if not others:
            return game.hunter
        candidates = others + ([game.hunter] if game.hunter is not None and game.hunter.active else [])
        mine = me.ship
        return min(candidates, key=lambda c: (c.x - mine.x) ** 2 + (c.y - mine.y) ** 2)


# Shooting through a wall block takes time the ship could spend flying, so the campaign pilot's routes (and the ground its
# progress is paid for) count a hit point as this many cells: through a block only when the way round is much longer.
CAMPAIGN_HIT_POINT_CELLS = 10.0


class CampaignSenses(Senses):
    hit_point_cells = CAMPAIGN_HIT_POINT_CELLS

    def __init__(self, settings, sticky_targets=False, target_recovery=False):
        super().__init__(settings)
        if target_recovery and not sticky_targets:
            raise ValueError("target recovery requires sticky targeting")
        self.target_recovery = target_recovery
        self.sticky_targets = sticky_targets
        self.preferred_enemy = None
        self._reset_recovery()

    def reset(self):
        super().reset()
        self.preferred_enemy = None
        self._reset_recovery()

    def observe(self, hunter, maze, enemies, projectiles, powerups, player, last_controls, now):
        if self.sticky_targets:
            if self.target_recovery:
                self._recover_target((hunter.x, hunter.y), maze, enemies, now)
            else:
                self._select_target((hunter.x, hunter.y), maze, enemies, now)
        return super().observe(hunter, maze, enemies, projectiles, powerups, player, last_controls, now)

    def _reset_recovery(self):
        self._target_disadvantaged_since = None
        self._target_cooldowns = {}
        self._target_acquired_at = self._target_progress_at = 0.0
        self._target_progress_cost = math.inf
        self._target_hits = None
        self.target_switch_reason = None

    def _route_cost(self, origin, maze, enemy):
        route = self._routes.cheapest_route(maze, origin, [(enemy.x, enemy.y)], self.hit_point_cells)
        if route is None:
            col = int((enemy.x - maze.offset_x) // maze.cell_size_x)
            row = int((enemy.y - maze.offset_y) // maze.cell_size_y)
            on_boundary = (0 <= col < maze.grid_width and 0 <= row < maze.grid_height and
                           (col in (0, maze.grid_width - 1) or row in (0, maze.grid_height - 1)))
            if not on_boundary or math.dist(origin, (enemy.x, enemy.y)) > self.settings.sensor_range(maze):
                return math.inf
            # Curved rock can leave open space inside a blocked boundary cell.
            # An enemy there is still attackable along a clear direct segment;
            # the coarse routing grid must not make us abandon a visible chase.
            walls = [w for w in list(maze.walls) + list(getattr(maze, 'gate_segments', ())) if w.active]
            hidden = blocked(origin, [(enemy.x, enemy.y)],
                             np.array([w.start for w in walls], dtype=float).reshape(-1, 2),
                             np.array([w.end for w in walls], dtype=float).reshape(-1, 2))[0]
            return math.inf if hidden else math.dist(origin, (enemy.x, enemy.y))
        points = [tuple(origin)] + route[1][::-1] + [(enemy.x, enemy.y)]
        return sum(math.dist(a, b) for a, b in zip(points, points[1:])) + route[2]

    @staticmethod
    def _remaining_hits(enemy):
        return float(getattr(enemy, 'effective_hit_points', getattr(enemy, 'hit_points', 1)))

    def _recover_target(self, origin, maze, enemies, now):
        """Keep productive chases; release stalled ones with hysteresis against switch-back."""
        self.target_switch_reason = None
        alive = [e for e in enemies if e.active]
        self._target_cooldowns = {e: until for e, until in self._target_cooldowns.items()
                                  if e in alive and until > now}
        costs = {}

        def cost(enemy):
            if enemy not in costs:
                costs[enemy] = self._route_cost(origin, maze, enemy)
            return costs[enemy]

        def cheapest(candidates):
            found = self._routes.cheapest_route(maze, origin, [(e.x, e.y) for e in candidates],
                                                self.hit_point_cells) if candidates else None
            return candidates[found[0]] if found is not None else None

        current = self.preferred_enemy
        eligible = [e for e in alive if e not in self._target_cooldowns]
        best = cheapest(eligible)
        if best is None:
            best = cheapest(alive)  # A cooldown must never strand the only reachable enemy.
        current_cost = cost(current) if current in alive else math.inf
        forced = current not in alive or not math.isfinite(current_cost)
        reason = None
        chosen = current
        if forced:
            chosen = best if best is not None else min(alive, key=lambda e: math.dist(origin, (e.x, e.y)), default=None)
            reason = 'acquired' if current is None else 'unavailable'
        else:
            hits = self._remaining_hits(current)
            cell = max(maze.cell_size_x, maze.cell_size_y)
            if hits < self._target_hits or self._target_progress_cost - current_cost >= cell * .25:
                self._target_progress_at = now
                self._target_progress_cost = current_cost
            self._target_hits = hits
            committed = now - self._target_acquired_at < 2.0
            disadvantaged = best is not None and best is not current and current_cost > 3 * cost(best) + 4 * cell
            if not committed and disadvantaged:
                if self._target_disadvantaged_since is None:
                    self._target_disadvantaged_since = now
                if now - self._target_disadvantaged_since >= 2.0:
                    chosen, reason = best, 'disadvantaged'
            else:
                self._target_disadvantaged_since = None
            if reason is None and not committed and now - self._target_progress_at >= 4.0:
                reach = self.settings.sensor_range(maze)
                rivals = [e for e in eligible if e is not current and math.dist(origin, (e.x, e.y)) <= reach]
                walls = [w for w in list(maze.spatial_grid.get_nearby_walls(origin, reach)) +
                         list(getattr(maze, 'gate_segments', ())) if w.active]
                hidden = blocked(origin, [(e.x, e.y) for e in rivals],
                                 np.array([w.start for w in walls]).reshape(-1, 2),
                                 np.array([w.end for w in walls]).reshape(-1, 2))
                rivals = [e for e, obscured in zip(rivals, hidden) if not obscured and
                          current_cost - cost(e) >= max(cell, current_cost * .25)]
                if rivals:
                    chosen, reason = min(rivals, key=cost), 'stalled'
        if chosen is not current:
            if not forced and current is not None:
                self._target_cooldowns[current] = now + 6.0
            self.preferred_enemy = chosen
            self._target_acquired_at = self._target_progress_at = now
            self._target_progress_cost = cost(chosen) if chosen is not None else math.inf
            self._target_hits = self._remaining_hits(chosen) if chosen is not None else None
            self._target_disadvantaged_since = None
            self.target_switch_reason = reason

    def _select_target(self, origin, maze, enemies, now):
        """Keep a living, reachable target through range-order changes and temporary loss of sight.

        A rival must be over three times cheaper, with four cells of slack, for two seconds before we give up
        a chase. This allows finishing a knocked-away enemy without committing forever to a distant one.
        """
        alive = [e for e in enemies if e.active]
        found = self._routes.cheapest_route(maze, origin, [(e.x, e.y) for e in alive], self.hit_point_cells) if alive else None
        best = (alive[found[0]] if found is not None else
                min(alive, key=lambda e: math.dist(origin, (e.x, e.y)), default=None))
        current = self.preferred_enemy
        keep = any(e is current for e in alive)
        if keep and current is not best:
            route = self._routes.cheapest_route(maze, origin, [(current.x, current.y)], self.hit_point_cells)
            if route is None and found is not None:
                keep = False
            elif route is not None and found is not None:
                def distance(enemy, result):
                    points = [tuple(origin)] + result[1][::-1] + [(enemy.x, enemy.y)]
                    return sum(math.dist(a, b) for a, b in zip(points, points[1:])) + result[2]
                slack = 4 * max(maze.cell_size_x, maze.cell_size_y)
                disadvantaged = distance(current, route) > 3 * distance(best, found) + slack
                if disadvantaged:
                    if self._target_disadvantaged_since is None:
                        self._target_disadvantaged_since = now
                    keep = now - self._target_disadvantaged_since < 2.0
                else:
                    self._target_disadvantaged_since = None
            else:
                self._target_disadvantaged_since = None
        else:
            self._target_disadvantaged_since = None
        if not keep:
            self.preferred_enemy = best
            self._target_disadvantaged_since = None

    def _enemy_compass(self, out, origin, h, reach, starts, ends, maze, enemies):
        if self.sticky_targets and self.preferred_enemy is not None:
            enemies = [self.preferred_enemy]
        super()._enemy_compass(out, origin, h, reach, starts, ends, maze, enemies)

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
