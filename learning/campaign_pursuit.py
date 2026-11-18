"""Short training-only searches: a stationary hidden enemy reachable through the real maze."""
import math

import numpy as np

from hunter.maze_route import MazeRoutes
from hunter.senses import blocked


def setup(game, seed):
    """Choose a hidden target two to eight route cells away; waiting cannot bring it into sight.

    Keep ship controls, collision physics, senses and wall geometry unchanged. The drill removes ordinary combat
    and pickups, so route following is the task. A failed placement is an error, never an easy visible target.
    """
    maze, ship = game.maze, game.ship
    routes = MazeRoutes()
    positions = [maze.position_calculator.grid_center_to_screen(x, y)
                 for y in range(1, maze.grid_height - 1) for x in range(1, maze.grid_width - 1)
                 if not maze.grid[y][x]]
    # A moving gate can reveal a stationary target to a stationary ship. Require fixed-wall occlusion;
    # route() still checks the live gates when confirming that the target is reachable.
    walls = [w for w in maze.walls if w.active]
    starts = np.array([w.start for w in walls], dtype=float).reshape(-1, 2)
    ends = np.array([w.end for w in walls], dtype=float).reshape(-1, 2)
    hidden = blocked(ship.get_pos(), positions, starts, ends)
    candidates = []
    for position, occluded in zip(positions, hidden):
        if not occluded or math.dist(ship.get_pos(), position) < 2 * ship.radius:
            continue
        found = routes.route(maze, ship.get_pos(), [position])
        if found is not None and 2 <= len(found[1]) <= 8:
            candidates.append(position)
    if not candidates:
        raise ValueError(f'level {game.level} has no hidden pursuit target with an open route of 2–8 cells')
    rng = np.random.default_rng(seed)
    target = candidates[int(rng.integers(len(candidates)))]
    game._close_enemy_hunters()
    game.encounter.close()
    game.roster.clear()
    enemy = game.encounter.spawn('static', target, game.level)
    game.encounter.ready()
    enemy.vx = enemy.vy = 0.0
    game.powerup_crystals.clear()
    game.projectiles.clear()
    ship.angle = float(rng.uniform(0, 360))
    ship.vx = ship.vy = 0.0
    game.autoplay.start_level(1)
    return target
