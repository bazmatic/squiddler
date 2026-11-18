"""A narrow demonstration teacher for leaving an open exit in Campaign practice.

This is an experiment-only teacher, not a gameplay controller. It steers toward
visible waypoints and damps momentum; its labels are not valid combat lessons.
"""
import math
import numpy as np
from hunter.campaign_pilot import ShipCommand
from hunter.maze_route import MazeRoutes
from hunter.senses import blocked


class ExitTeacher:
    def __init__(self):
        self.routes = MazeRoutes()

    def reset(self):
        self.routes = MazeRoutes()

    def decide(self, game):
        ship, maze = game.ship, game.maze
        if not game.exit_opened:
            return None
        origin, target = (ship.x, ship.y), tuple(maze.exit.get_pos())
        route = self.routes.route(maze, origin, [target])
        if route is None:
            return None
        reach = max(maze.cell_size_x, maze.cell_size_y) * 8
        walls = [w for w in maze.walls if w.active] + [g for g in maze.gate_segments if g.active]
        starts = np.array([w.start for w in walls], dtype=float).reshape(-1, 2)
        ends = np.array([w.end for w in walls], dtype=float).reshape(-1, 2)
        points = [target] + route[1]
        hidden = blocked(origin, points, starts, ends)
        visible = [p for p, obscured in zip(points, hidden)
                   if not obscured and math.dist(origin, p) <= reach]
        if not visible:
            return None
        waypoint = visible[0]
        dx, dy = waypoint[0] - ship.x, waypoint[1] - ship.y
        distance = math.hypot(dx, dy)
        if distance < 1e-6:
            return ShipCommand(0, False, False)
        # Aim acceleration at a velocity that closes on the waypoint without
        # carrying the ship too fast into the exit-side rock.
        desired_speed = min(4.0, max(1.0, distance / 40.0))
        ax, ay = dx / distance * desired_speed - ship.vx, dy / distance * desired_speed - ship.vy
        if math.hypot(ax, ay) < .25:
            return ShipCommand(0, False, False)
        desired = math.degrees(math.atan2(ay, ax))
        diff = (desired - ship.angle + 180) % 360 - 180
        # Actions are held for six frames; a half-step deadband prevents jitter.
        deadband = ship.current_rotation_speed * 3.0
        turn = 0 if abs(diff) <= deadband else (1 if diff > 0 else -1)
        return ShipCommand(turn, abs(diff) <= 35.0, False)
