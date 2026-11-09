"""Fire control: the firing solution the hunter tracks and holds fire for.

The pilot decides whether to engage. Holding a nose within a few degrees of a
target is beyond a pilot that chooses a few times a second, so that last part
belongs to the ship: it trims the nose onto the solution found here and
releases shots only when the solution says they will hit.
"""
import math
import config
from hunter.model import FiringSolution
from hunter.pilot_sensors import firing_solution, nose_miss_distance, off_nose
from hunter.visibility import visible_point
from weapons.state import held_weapon


def find_firing_solution(hunter, enemies, maze, settings):
    """The firing solution for the nearest enemy the hunter can see and reach, or None.

    Uses the same visibility and lead calculation as the readings shown to the
    pilot, so the ship engages the enemy the pilot was looking at.
    """
    origin = hunter.get_pos()
    weapon = held_weapon(hunter)
    if weapon is not None:
        shot_speed = weapon.definition.base_speed * weapon.definition.speed
        shot_lifetime = weapon.definition.lifetime
    else:
        boosted = getattr(hunter, 'gun_upgrade_level', 0) > 0
        shot_speed = config.PROJECTILE_SPEED * (config.UPGRADED_PROJECTILE_SPEED_MULTIPLIER if boosted else 1)
        shot_lifetime = config.PROJECTILE_LIFETIME
    sensor_range = settings.sensor_range(maze)
    in_range = sorted(
        (math.hypot(enemy.x - origin[0], enemy.y - origin[1]), index, enemy)
        for index, enemy in enumerate(enemies) if enemy.active)
    in_range = [(distance, enemy) for distance, _, enemy in in_range if distance <= sensor_range]
    if not in_range:
        return None
    for _, enemy in in_range:
        # This runs every frame, so only the walls that could lie across the line
        # of sight are tested: those whose extent overlaps the box around it.
        left, right = min(origin[0], enemy.x) - 1, max(origin[0], enemy.x) + 1
        top, bottom = min(origin[1], enemy.y) - 1, max(origin[1], enemy.y) + 1
        # The spatial grid's walls along the line are a superset of those (it indexes each wall in
        # every bucket its box touches), and visible_point does not care about their order
        grid = getattr(maze, 'spatial_grid', None)
        nearby = (grid.get_walls_along_path(origin, (enemy.x, enemy.y), 1) if grid is not None
                  else list(maze.walls))
        across = [
            (w.start, w.end) for w in nearby + list(getattr(maze, 'gate_segments', ()))
            if w.active
            and not (w.start[0] < left and w.end[0] < left) and not (w.start[0] > right and w.end[0] > right)
            and not (w.start[1] < top and w.end[1] < top) and not (w.start[1] > bottom and w.end[1] > bottom)]
        if not visible_point(origin, enemy.get_pos(), across, sensor_range):
            continue
        relative = (enemy.x - origin[0], enemy.y - origin[1])
        velocity = (enemy.vx, enemy.vy)
        lead = firing_solution(relative, velocity, shot_speed)
        if lead is None or lead[0] > shot_lifetime:
            continue
        miss = nose_miss_distance(relative, velocity, hunter.angle,
                                  shot_speed, shot_lifetime)
        reach = enemy.radius + config.PROJECTILE_SIZE
        return FiringSolution(off_nose(lead[1], hunter.angle, digits=None),
                              miss is not None and miss <= reach)
    return None
