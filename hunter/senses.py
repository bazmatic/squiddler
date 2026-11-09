"""The neural pilot's senses: a cheap fixed-length vector built straight from the game objects.

Only raw geometry (bearings, distances, relative velocities, wall distances along rays) and the hunter's
own motion. None of the rules pilot's solved quantities (lead angles, burn headings, safe speeds).
"""
import math
import weakref
import numpy as np
from hunter.maze_route import MazeRoutes

RAYS = 16
MAX_ENEMIES, MAX_BULLETS, MAX_POWERUPS, MAX_MEMORY = 6, 4, 2, 2
MAX_POWERUP_MEMORY = 2
POWERUP_MEMORY_TTL = 90.0   # maximum seconds a crystal lost from view is remembered
MEMORY_TTL = 10.0          # seconds an out-of-sight enemy is remembered
OBSERVATION_VERSION = 2
UPGRADE_CAP = 6.0
HITS_CAP = 20             # hits left read as log(1 + hits) / log(1 + HITS_CAP), capped at 1: a Mother reads 1
HIT_POINT_CELLS = 2.0      # the enemy compass routes through a wall block at this many cells of route per hit point

_RAY_ANGLES = np.arange(RAYS) * (2 * math.pi / RAYS)
_RAY_COS, _RAY_SIN = np.cos(_RAY_ANGLES), np.sin(_RAY_ANGLES)


def _names():
    names = ['fwd_speed', 'side_speed', 'speed', 'heading_sin', 'heading_cos', 'health', 'cooldown',
             'burst', 'upgrade', 'enemies_present']
    names += [f'ray_{i}' for i in range(RAYS)]
    for group, count in (('enemy', MAX_ENEMIES), ('bullet', MAX_BULLETS)):
        for i in range(count):
            names += [f'{group}{i}_{part}' for part in ('present', 'sin', 'cos', 'dist', 'relfwd', 'relside')]
    for i in range(MAX_POWERUPS):
        names += [f'powerup{i}_{part}' for part in ('present', 'sin', 'cos', 'dist')]
    names += [f'player_{part}' for part in ('present', 'sin', 'cos', 'dist', 'relfwd', 'relside')]
    for i in range(MAX_MEMORY):
        names += [f'memory{i}_{part}' for part in ('present', 'sin', 'cos', 'dist', 'age')]
    names += ['last_turn_left', 'last_turn_none', 'last_turn_right', 'last_thrust', 'last_fire']
    # Added after the first trained pilots: kept at the end so their layout stays a prefix of this one.
    for i in range(MAX_POWERUP_MEMORY):
        names += [f'powerup_memory{i}_{part}' for part in ('present', 'sin', 'cos', 'dist', 'age')]
    names += [f'crystal_compass_{part}' for part in ('present', 'sin', 'cos', 'dist')]
    names += ['hit_present', 'hit_fwd', 'hit_side']
    names += [f'enemy_compass_{part}' for part in ('present', 'sin', 'cos', 'dist')]
    names += [f'enemy{i}_hits' for i in range(MAX_ENEMIES)]
    names += ['power_boost', 'power_boost_remaining']
    for group, count in (('powerup', MAX_POWERUPS), ('powerup_memory', MAX_POWERUP_MEMORY)):
        names += [f'{group}{i}_lifetime' for i in range(count)]
    return tuple(names)


FEATURE_NAMES = _names()
SENSE_SIZE = len(FEATURE_NAMES)
_INDEX = {name: i for i, name in enumerate(FEATURE_NAMES)}


def ray_distances(origin, heading_degrees, wall_starts, wall_ends, reach):
    """Distance to the nearest wall along each of 16 rays, ray 0 along the nose, capped at `reach`."""
    if len(wall_starts) == 0:
        return np.full(RAYS, float(reach))
    h = math.radians(heading_degrees)
    ch, sh = math.cos(h), math.sin(h)
    dx = _RAY_COS * ch - _RAY_SIN * sh
    dy = _RAY_SIN * ch + _RAY_COS * sh
    ex = wall_ends[:, 0] - wall_starts[:, 0]
    ey = wall_ends[:, 1] - wall_starts[:, 1]
    ax = wall_starts[:, 0] - origin[0]
    ay = wall_starts[:, 1] - origin[1]
    denom = dx[:, None] * ey[None, :] - dy[:, None] * ex[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (ax[None, :] * ey[None, :] - ay[None, :] * ex[None, :]) / denom
        u = (ax[None, :] * dy[:, None] - ay[None, :] * dx[:, None]) / denom
    hit = (np.abs(denom) > 1e-9) & (t >= 0) & (u >= 0) & (u <= 1)
    t = np.where(hit, t, np.inf)
    return np.minimum(t.min(axis=1), float(reach))


def ray_hits(origin, heading_degrees, walls, reach):
    """The wall each of the 16 sight rays meets first within `reach` (None where it meets none), ray 0 along the nose.
    `walls` are segments with .start and .end."""
    walls = list(walls)
    if not walls:
        return [None] * RAYS
    starts = np.array([w.start for w in walls], dtype=float).reshape(-1, 2)
    ends = np.array([w.end for w in walls], dtype=float).reshape(-1, 2)
    h = math.radians(heading_degrees)
    ch, sh = math.cos(h), math.sin(h)
    dx = _RAY_COS * ch - _RAY_SIN * sh
    dy = _RAY_SIN * ch + _RAY_COS * sh
    ex, ey = ends[:, 0] - starts[:, 0], ends[:, 1] - starts[:, 1]
    ax, ay = starts[:, 0] - origin[0], starts[:, 1] - origin[1]
    denom = dx[:, None] * ey[None, :] - dy[:, None] * ex[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (ax[None, :] * ey[None, :] - ay[None, :] * ex[None, :]) / denom
        u = (ax[None, :] * dy[:, None] - ay[None, :] * dx[:, None]) / denom
    hit = (np.abs(denom) > 1e-9) & (t >= 0) & (t <= reach) & (u >= 0) & (u <= 1)
    t = np.where(hit, t, np.inf)
    nearest = t.argmin(axis=1)
    return [walls[j] if np.isfinite(t[i, j]) else None for i, j in enumerate(nearest)]


def blocked(origin, targets, wall_starts, wall_ends):
    """Whether a wall lies on the straight line from `origin` to each row of `targets`."""
    targets = np.asarray(targets, dtype=float).reshape(-1, 2)
    if len(targets) == 0 or len(wall_starts) == 0:
        return np.zeros(len(targets), dtype=bool)
    dx = targets[:, 0] - origin[0]
    dy = targets[:, 1] - origin[1]
    ex = wall_ends[:, 0] - wall_starts[:, 0]
    ey = wall_ends[:, 1] - wall_starts[:, 1]
    ax = wall_starts[:, 0] - origin[0]
    ay = wall_starts[:, 1] - origin[1]
    denom = dx[:, None] * ey[None, :] - dy[:, None] * ex[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        t = (ax[None, :] * ey[None, :] - ay[None, :] * ex[None, :]) / denom
        u = (ax[None, :] * dy[:, None] - ay[None, :] * dx[:, None]) / denom
    hit = (np.abs(denom) > 1e-9) & (t > 0) & (t < 1) & (u >= 0) & (u <= 1)
    return hit.any(axis=1)


class Senses:
    def __init__(self, settings):
        self.settings = settings
        self._memory = weakref.WeakKeyDictionary()    # enemy -> (x, y, seen_at)
        self._crystal_memory = {}    # id -> (crystal, x, y, seen_at); held strongly so an id is not reused
        self._routes = MazeRoutes()

    hit_point_cells = HIT_POINT_CELLS   # route cells one hit point of wall block counts as; a subclass may price blocks higher

    def reset(self):
        self._memory = weakref.WeakKeyDictionary()
        self._crystal_memory = {}
        self._routes = MazeRoutes()

    def perceive(self, hunter, surroundings, last_controls, now):
        """The senses of `surroundings` (see hunter.surroundings) as they are now."""
        return self.observe(hunter, surroundings.maze, surroundings.enemies(), surroundings.threats(),
                            surroundings.pickups(), surroundings.friend(), last_controls, now)

    def observe(self, hunter, maze, enemies, projectiles, powerups, player, last_controls, now):
        out = np.zeros(SENSE_SIZE, dtype=np.float32)
        origin = (hunter.x, hunter.y)
        reach = self.settings.sensor_range(maze)
        vmax = max(hunter.max_speed, 1e-6)
        h = math.radians(hunter.angle)
        ch, sh = math.cos(h), math.sin(h)

        walls = [w for w in list(maze.spatial_grid.get_nearby_walls(origin, reach)) + list(getattr(maze, 'gate_segments', ()))
                 if w.active]
        starts = np.array([w.start for w in walls], dtype=float).reshape(-1, 2)
        ends = np.array([w.end for w in walls], dtype=float).reshape(-1, 2)

        fwd = hunter.vx * ch + hunter.vy * sh
        side = -hunter.vx * sh + hunter.vy * ch
        out[_INDEX['fwd_speed']] = fwd / vmax
        out[_INDEX['side_speed']] = side / vmax
        out[_INDEX['speed']] = math.hypot(hunter.vx, hunter.vy) / vmax
        out[_INDEX['heading_sin']] = sh
        out[_INDEX['heading_cos']] = ch
        out[_INDEX['health']] = hunter.health / max(self.settings.health, 1)
        out[_INDEX['cooldown']] = min(1.0, hunter.fire_remaining / max(hunter.fire_interval, 1e-6))
        out[_INDEX['burst']] = hunter.burst_remaining / max(self.settings.burst_size, 1)
        out[_INDEX['upgrade']] = min(1.0, hunter.gun_upgrade_level / UPGRADE_CAP)
        # A pickup weapon is a boost too, lasting as long as its ammo
        from weapons.state import held_weapon  # here, not at the top: the weapons package loads pygame
        weapon = held_weapon(hunter)
        out[_INDEX['power_boost']] = float(hunter.gun_upgrade_level > 0 or weapon is not None)
        import config
        out[_INDEX['power_boost_remaining']] = (weapon.view.ammo_fraction if weapon is not None else min(1.0, max(0.0,
            getattr(hunter, 'upgrade_remaining', 0) / config.POWERUP_DURATION_SECONDS)))
        out[_INDEX['enemies_present']] = 1.0 if any(e.active for e in enemies) else 0.0

        rays = ray_distances(origin, hunter.angle, starts, ends, reach) / reach
        out[_INDEX['ray_0']:_INDEX['ray_0'] + RAYS] = rays

        def candidates(entities, limit):
            """The nearest `limit` active entities within reach and in line of sight."""
            near = []
            for e in entities:
                if not e.active:
                    continue
                d = math.hypot(e.x - origin[0], e.y - origin[1])
                if d <= reach:
                    near.append((d, e))
            near.sort(key=lambda item: item[0])
            near = near[:limit + 3]                                  # a few spare for the ones that are hidden
            if not near:
                return []
            hidden = blocked(origin, [(e.x, e.y) for _, e in near], starts, ends)
            return [item for item, hid in zip(near, hidden) if not hid][:limit]

        def contact(prefix, d, e, with_velocity=True):
            dx, dy = e.x - origin[0], e.y - origin[1]
            rel = math.atan2(dy, dx) - h
            out[_INDEX[f'{prefix}_present']] = 1.0
            out[_INDEX[f'{prefix}_sin']] = math.sin(rel)
            out[_INDEX[f'{prefix}_cos']] = math.cos(rel)
            out[_INDEX[f'{prefix}_dist']] = d / reach
            if with_velocity:
                rvx, rvy = e.vx - hunter.vx, e.vy - hunter.vy
                out[_INDEX[f'{prefix}_relfwd']] = (rvx * ch + rvy * sh) / vmax
                out[_INDEX[f'{prefix}_relside']] = (-rvx * sh + rvy * ch) / vmax

        seen = candidates(enemies, MAX_ENEMIES)
        for i, (d, e) in enumerate(seen):
            contact(f'enemy{i}', d, e)
            hits = max(0.0, float(getattr(e, 'effective_hit_points', getattr(e, 'hit_points', 1))))
            out[_INDEX[f'enemy{i}_hits']] = min(1.0, math.log1p(hits) / math.log1p(HITS_CAP))
            self._memory[e] = (e.x, e.y, now)
        for i, (d, b) in enumerate(candidates([p for p in projectiles if p.is_enemy], MAX_BULLETS)):
            contact(f'bullet{i}', d, b)
        crystals_seen = candidates(powerups, MAX_POWERUPS)
        for i, (d, c) in enumerate(crystals_seen):
            contact(f'powerup{i}', d, c, with_velocity=False)
            self._pickup_lifetime(out, f'powerup{i}', c)
            self._crystal_memory[id(c)] = (c, c.x, c.y, now)
        if player is not None and player.active:
            d = math.hypot(player.x - origin[0], player.y - origin[1])
            if d <= reach and not blocked(origin, [(player.x, player.y)], starts, ends)[0]:
                contact('player', d, player)

        self._remember(out, origin, h, reach, starts, ends, now, {id(e) for _, e in seen})
        self._remember_crystals(out, origin, h, reach, now, {id(c) for _, c in crystals_seen})
        # A collision with a bullet or a ship since the last decision: the strongest push, in the hunter's frame as a
        # share of top speed. Walls are not counted: the rays already tell it how close they are.
        from creatures.traits.energy import take_hit
        read_hit = getattr(hunter, 'take_hit', None)
        hit = read_hit() if read_hit is not None else take_hit(hunter)
        if hit is not None:
            out[_INDEX['hit_present']] = 1.0
            out[_INDEX['hit_fwd']] = max(-1.0, min(1.0, (hit[0] * ch + hit[1] * sh) / vmax))
            out[_INDEX['hit_side']] = max(-1.0, min(1.0, (-hit[0] * sh + hit[1] * ch) / vmax))
        # The compass points at the nearest uncollected crystal wherever it is, through walls: straight-line
        # bearing and distance (in maze widths), the way the game's own pickups can be found without being seen.
        loose = [c for c in powerups if c.active]
        if loose:
            nearest = min(loose, key=lambda c: (c.x - origin[0]) ** 2 + (c.y - origin[1]) ** 2)
            rel = math.atan2(nearest.y - origin[1], nearest.x - origin[0]) - h
            out[_INDEX['crystal_compass_present']] = 1.0
            out[_INDEX['crystal_compass_sin']] = math.sin(rel)
            out[_INDEX['crystal_compass_cos']] = math.cos(rel)
            out[_INDEX['crystal_compass_dist']] = min(
                1.0, math.hypot(nearest.x - origin[0], nearest.y - origin[1]) / max(maze.grid_width * maze.cell_size_x, 1.0))
        self._enemy_compass(out, origin, h, reach, starts, ends, maze, enemies)
        turn, thrust, fire = last_controls
        out[_INDEX[('last_turn_left', 'last_turn_none', 'last_turn_right')[turn]]] = 1.0
        out[_INDEX['last_thrust']] = 1.0 if thrust else 0.0
        out[_INDEX['last_fire']] = 1.0 if fire else 0.0
        return out

    @staticmethod
    def _pickup_lifetime(out, prefix, crystal):
        lifetime = getattr(crystal, 'LIFETIME_FRAMES', None)
        if lifetime is not None:
            out[_INDEX[f'{prefix}_lifetime']] = max(0.0, 1-crystal.age/lifetime)

    def _remember_crystals(self, out, origin, h, reach, now, seen_ids):
        """Fill the powerup memory slots with the nearest crystals seen lately but not in the two sight slots now.

        Memories end when a crystal is collected, expires, or grows too old.
        """
        kept = []
        for key, (crystal, x, y, seen_at) in list(self._crystal_memory.items()):
            if not crystal.active or now - seen_at > POWERUP_MEMORY_TTL:
                del self._crystal_memory[key]
            elif key not in seen_ids:
                kept.append((math.hypot(x - origin[0], y - origin[1]), x, y, seen_at, crystal))
        kept.sort(key=lambda item: item[0])
        for i, (d, x, y, seen_at, crystal) in enumerate(kept[:MAX_POWERUP_MEMORY]):
            self._pickup_lifetime(out, f'powerup_memory{i}', crystal)
            rel = math.atan2(y - origin[1], x - origin[0]) - h
            out[_INDEX[f'powerup_memory{i}_present']] = 1.0
            out[_INDEX[f'powerup_memory{i}_sin']] = math.sin(rel)
            out[_INDEX[f'powerup_memory{i}_cos']] = math.cos(rel)
            out[_INDEX[f'powerup_memory{i}_dist']] = min(1.0, d / reach)
            out[_INDEX[f'powerup_memory{i}_age']] = (now - seen_at) / POWERUP_MEMORY_TTL

    def _remember(self, out, origin, h, reach, starts, ends, now, seen_ids):
        """Fill the memory slots with the nearest enemies seen lately but not in sight now."""
        entries = []
        for enemy, (x, y, seen_at) in list(self._memory.items()):
            if not enemy.active or now - seen_at > MEMORY_TTL:
                del self._memory[enemy]
                continue
            if id(enemy) in seen_ids:
                continue
            entries.append((math.hypot(x - origin[0], y - origin[1]), enemy, x, y, seen_at))
        if not entries:
            return
        in_view = ~blocked(origin, [(x, y) for _, _, x, y, _ in entries], starts, ends)
        kept = []
        for (d, enemy, x, y, seen_at), visible in zip(entries, in_view):
            # Its last position is in plain view and the enemy is not there: the memory is stale.
            if visible and d <= reach:
                del self._memory[enemy]
            else:
                kept.append((d, x, y, seen_at))
        kept.sort(key=lambda item: item[0])
        for i, (d, x, y, seen_at) in enumerate(kept[:MAX_MEMORY]):
            rel = math.atan2(y - origin[1], x - origin[0]) - h
            out[_INDEX[f'memory{i}_present']] = 1.0
            out[_INDEX[f'memory{i}_sin']] = math.sin(rel)
            out[_INDEX[f'memory{i}_cos']] = math.cos(rel)
            out[_INDEX[f'memory{i}_dist']] = min(1.0, d / reach)
            out[_INDEX[f'memory{i}_age']] = (now - seen_at) / MEMORY_TTL

    def _enemy_compass(self, out, origin, h, reach, starts, ends, maze, enemies):
        """Point the way to the nearest live enemy by route, wherever it is: at the furthest point of the cell route
        in plain sight, with the route's length in maze widths. The route may go through breakable wall blocks,
        each costing HIT_POINT_CELLS cells per hit point left. An enemy no route reaches is pointed at directly."""
        alive = [e for e in enemies if e.active]
        if not alive:
            return
        found = self._routes.cheapest_route(maze, origin, [(e.x, e.y) for e in alive], self.hit_point_cells)
        if found is None:
            target = min(((e.x, e.y) for e in alive), key=lambda p: math.dist(origin, p))
            waypoint, length = target, math.dist(origin, target)
        else:
            index, route, extra = found
            target = (alive[index].x, alive[index].y)
            points = [target] + route                                  # from the target back toward the hunter
            hidden = blocked(origin, points, starts, ends)
            k = next((i for i, (p, hid) in enumerate(zip(points, hidden)) if not hid and math.dist(origin, p) <= reach),
                     len(points) - 1)
            waypoint = points[k]
            # Straight to the waypoint, then along the rest of the route.
            legs = [tuple(origin)] + points[k::-1]
            # Each hit point left in a block on the route counts as hit_point_cells cells more to go.
            length = sum(math.dist(a, b) for a, b in zip(legs, legs[1:])) + extra
        rel = math.atan2(waypoint[1] - origin[1], waypoint[0] - origin[0]) - h
        out[_INDEX['enemy_compass_present']] = 1.0
        out[_INDEX['enemy_compass_sin']] = math.sin(rel)
        out[_INDEX['enemy_compass_cos']] = math.cos(rel)
        out[_INDEX['enemy_compass_dist']] = min(1.0, length / max(maze.grid_width * maze.cell_size_x, 1.0))
