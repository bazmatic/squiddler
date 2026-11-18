"""Non-mutating, decision-time checks for declining a usable direct shot. No steering or firing assist."""
import math
from dataclasses import dataclass

import config
from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES
from hunter.settings_for import hunter_settings_for
from hunter.senses import MAX_ENEMIES
from weapons.geometry import segment_contact

_INDEX = {name: i for i, name in enumerate(CAMPAIGN_FEATURE_NAMES)}
ROUGH_AIM_DEGREES = 30.0


def _lanes(ship, turn):
    # The game applies the chosen turn before firing on this frame.
    shot_heading = ship.angle + turn * ship.current_rotation_speed
    heading = math.radians(shot_heading)
    muzzle = (ship.x + math.cos(heading) * (ship.radius + 5),
              ship.y + math.sin(heading) * (ship.radius + 5))
    if ship.weapon is not None:
        d = ship.weapon.definition
        if d.behaviour not in ('bolt', 'rail', 'inflator', 'seeker'):
            return  # bouncing/explosive opportunities need a different prediction model
        radius = d.radius or d.base_radius
        from weapons.projectile import RailProjectile
        hit_radius = RailProjectile.BEAM_RADIUS if d.behaviour == 'rail' else radius
        speed, lifetime, offsets = d.base_speed * d.speed, d.lifetime, d.offsets
    else:
        upgraded = ship.gun_upgrade_level >= 1
        extra = max(0, ship.gun_upgrade_level - 3)
        radius = config.PROJECTILE_SIZE * (config.UPGRADED_PROJECTILE_SIZE_MULTIPLIER if upgraded else 1)
        radius *= 1 + extra * config.POWERUP_BEYOND_LEVEL_3_SIZE_INCREMENT
        hit_radius = radius
        speed = config.PROJECTILE_SPEED * (config.UPGRADED_PROJECTILE_SPEED_MULTIPLIER if upgraded else 1)
        speed *= 1 + extra * config.POWERUP_BEYOND_LEVEL_3_SPEED_INCREMENT
        lifetime = config.PROJECTILE_LIFETIME
        from game_handlers.fire_rate_calculator import shots_per_volley
        spread = config.UPGRADED_PROJECTILE_SPREAD_ANGLE
        offsets = ((0, -spread), (0, 0), (0, spread)) if shots_per_volley(ship.gun_upgrade_level) > 1 else ((0, 0),)
    for side, offset in offsets:
        start = (muzzle[0] - math.sin(heading) * side, muzzle[1] + math.cos(heading) * side)
        angle = math.radians(shot_heading + offset)
        yield start, (math.cos(angle) * speed, math.sin(angle) * speed), lifetime, hit_radius, radius


def opportunity(game, senses, settings=None, turn=0):
    """(eligible targets, firing interval in seconds), using only enemies in the decision's sight slots.

    A straight shot must intersect a target moving at its current velocity before expiry, with no wall or gate
    contact first. Shots do not inherit ship velocity. Seeker shots qualify only if their direct ballistic path
    works; grenade/bounce/blast predictions are deliberately excluded. This is a local prediction, not a promise
    that an enemy will keep its course or that a moving gate will stay put.
    """
    from hunter.campaign_senses import ShipBody
    settings = settings or hunter_settings_for('neural')
    ship, maze = game.ship, game.maze
    if ship.weapon is not None:
        interval = ship.weapon.definition.interval / config.FPS
        ready = ship.weapon.cooldown <= 1e-8
    else:
        body = ShipBody(ship, game, settings)
        interval = body.fire_interval
        from game_handlers.fire_rate_calculator import calculate_fire_cooldown
        ready = game.clock_ms() - getattr(game, 'last_shot_time', 0) > calculate_fire_cooldown(ship)
    interval = max(interval, 1e-6)
    if not ready:
        return (), interval
    if ship.weapon is None and ship.ammo_policy.charges_for_shots() and ship.ammo <= 0:
        return (), interval
    if ship.weapon is not None and ship.weapon.ammo is not None and ship.weapon.ammo <= 0:
        return (), interval
    reach = settings.sensor_range(maze)
    observed = []
    for i in range(MAX_ENEMIES):
        if senses[_INDEX[f'enemy{i}_present']] != 1:
            continue
        bearing = math.radians(ship.angle) + math.atan2(senses[_INDEX[f'enemy{i}_sin']],
                                                       senses[_INDEX[f'enemy{i}_cos']])
        distance = senses[_INDEX[f'enemy{i}_dist']] * reach
        observed.append((ship.x + math.cos(bearing) * distance, ship.y + math.sin(bearing) * distance))
    if not observed:
        return (), interval
    lanes = tuple(_lanes(ship, turn))
    eligible = []
    for target in game.roster.alive():
        if (getattr(target, 'indestructible', False) or getattr(target, 'immunity_remaining', 0) > 0
                or not any(math.dist(target.get_pos(), point) <= max(.1, reach * 1e-5) for point in observed)):
            continue
        for start, velocity, lifetime, hit_radius, wall_radius in lanes:
            relative = ((velocity[0] - target.vx) * lifetime, (velocity[1] - target.vy) * lifetime)
            time = target.projectile_contact(start, relative, hit_radius)
            if time is None:
                continue
            delta = (velocity[0] * lifetime, velocity[1] * lifetime)
            impact = (start[0] + delta[0] * time, start[1] + delta[1] * time)
            grid = getattr(maze, 'spatial_grid', None)
            walls = list(grid.get_walls_along_path(start, impact, wall_radius + 1)) if grid else list(maze.walls)
            walls += list(getattr(maze, 'gate_segments', ()))
            obstructed = False
            for wall in walls:
                if not wall.active:
                    continue
                contact = segment_contact(start, delta, wall.start, wall.end, wall_radius)
                if contact is not None and contact[0] <= time + 1e-9:
                    obstructed = True
                    break
            if not obstructed:
                eligible.append(target)
                break
    return tuple(eligible), interval


def observed_targets(game, senses, settings=None):
    """Live references to the enemies actually delivered in this decision's observation slots."""
    settings = settings or hunter_settings_for('neural')
    ship = game.ship
    reach = settings.sensor_range(game.maze)
    points = []
    for i in range(MAX_ENEMIES):
        if senses[_INDEX[f'enemy{i}_present']] != 1:
            continue
        angle = math.radians(ship.angle) + math.atan2(senses[_INDEX[f'enemy{i}_sin']],
                                                     senses[_INDEX[f'enemy{i}_cos']])
        distance = senses[_INDEX[f'enemy{i}_dist']] * reach
        points.append((ship.x + math.cos(angle)*distance, ship.y + math.sin(angle)*distance))
    return tuple(target for target in game.roster.alive()
                 if any(math.dist(target.get_pos(), point) <= max(.1, reach*1e-5) for point in points))


@dataclass(frozen=True)
class AimFeedback:
    rough: bool
    accuracy: float
    usable_targets: tuple


def aim_feedback(game, targets, lanes, heading, settings=None):
    """Neutral inside the broad aim window; continuous positive credit toward a clear intercept direction.

    Rough alignment may use the current bearing or predicted intercept. Positive credit and the missed-shot
    opportunity additionally require an intercept within projectile lifetime and a clear path. Score a volley
    once using its best lane, so spread weapons cannot multiply the reward.
    """
    from hunter.pilot_sensors import firing_solution
    from hunter.senses import blocked
    settings = settings or hunter_settings_for('neural')
    ship, maze = game.ship, game.maze
    walls = [w for w in list(maze.walls) + list(getattr(maze, 'gate_segments', ())) if w.active]
    starts = [w.start for w in walls]
    ends = [w.end for w in walls]
    import numpy as np
    starts = np.asarray(starts, dtype=float).reshape(-1, 2)
    ends = np.asarray(ends, dtype=float).reshape(-1, 2)
    rough, accuracy, usable = False, 0.0, []
    for target in targets:
        if (not target.active or getattr(target, 'allegiance', 'enemy') != 'enemy'
                or math.dist(ship.get_pos(), target.get_pos()) > settings.sensor_range(maze)
                or blocked(ship.get_pos(), [target.get_pos()], starts, ends)[0]):
            continue
        bearing = math.degrees(math.atan2(target.y-ship.y, target.x-ship.x))
        direct = abs((bearing-heading+180) % 360-180) <= ROUGH_AIM_DEGREES
        rough |= direct
        for start, velocity, lifetime, hit_radius, wall_radius in lanes:
            speed = math.hypot(*velocity)
            solution = firing_solution((target.x-start[0], target.y-start[1]), (target.vx, target.vy), speed)
            if solution is None:
                continue
            _, ideal = solution
            actual = math.degrees(math.atan2(velocity[1], velocity[0]))
            error = abs((actual-ideal+180) % 360-180)
            near = error <= ROUGH_AIM_DEGREES
            rough |= near
            if (getattr(target, 'indestructible', False) or getattr(target, 'immunity_remaining', 0) > 0):
                continue
            angle = math.radians(ideal)
            ideal_velocity = (math.cos(angle)*speed, math.sin(angle)*speed)
            relative = ((ideal_velocity[0]-target.vx)*lifetime, (ideal_velocity[1]-target.vy)*lifetime)
            contact = target.projectile_contact(start, relative, hit_radius)
            if contact is None:
                continue
            delta = (ideal_velocity[0]*lifetime, ideal_velocity[1]*lifetime)
            if any((hit := segment_contact(start, delta, w.start, w.end, wall_radius)) is not None
                   and hit[0] <= contact + 1e-9 for w in walls):
                continue
            if direct or near:
                if target not in usable:
                    usable.append(target)
                accuracy = max(accuracy, max(0.0, 1-error/ROUGH_AIM_DEGREES))
    return AimFeedback(bool(rough), accuracy, tuple(usable))


def rough_opportunity(game, senses, settings=None, turn=0):
    """A ready shot roughly aimed toward a seen target, rather than the historical exact-hit requirement."""
    from hunter.campaign_senses import ShipBody
    settings = settings or hunter_settings_for('neural')
    ship = game.ship
    if ship.weapon is not None:
        interval, ready = ship.weapon.definition.interval/config.FPS, ship.weapon.cooldown <= 1e-8
        empty = ship.weapon.ammo is not None and ship.weapon.ammo <= 0
    else:
        interval = ShipBody(ship, game, settings).fire_interval
        from game_handlers.fire_rate_calculator import calculate_fire_cooldown
        ready = game.clock_ms()-getattr(game, 'last_shot_time', 0) > calculate_fire_cooldown(ship)
        empty = ship.ammo_policy.charges_for_shots() and ship.ammo <= 0
    if not ready or empty:
        return (), interval
    feedback = aim_feedback(game, observed_targets(game, senses, settings), tuple(_lanes(ship, turn)),
                            ship.angle+turn*ship.current_rotation_speed, settings)
    return feedback.usable_targets, interval


def fired_feedback(game, targets, projectiles, settings=None):
    """Grade the actual emitted volley before it advances, using its actual muzzle and velocity."""
    lanes = []
    for shot in projectiles:
        behaviour = getattr(getattr(shot, 'definition', None), 'behaviour', 'bolt')
        if behaviour not in ('bolt', 'rail', 'inflator', 'seeker'):
            continue
        from weapons.projectile import RailProjectile
        radius = RailProjectile.BEAM_RADIUS if behaviour == 'rail' else shot.radius
        lanes.append((shot.get_pos(), (shot.vx, shot.vy), shot.lifetime, radius, shot.radius))
    return aim_feedback(game, targets, lanes, game.ship.angle, settings)
