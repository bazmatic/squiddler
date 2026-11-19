"""Where two ships start side by side: touching but not overlapping, and clear of rock."""
import math
from typing import List, Tuple

Point = Tuple[float, float]


def _distance_to_segment(p: Point, a: Point, b: Point) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    length_sq = dx * dx + dy * dy
    t = 0.0 if length_sq == 0 else max(0.0, min(1.0, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / length_sq))
    return math.hypot(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def circle_clear(maze, pos: Point, radius: float) -> bool:
    """Whether a circle at `pos` touches no wall of the maze."""
    for wall in maze.spatial_grid.get_nearby_walls(pos, radius * 2.0):
        if hasattr(wall, 'get_segment'):
            if not wall.active:
                continue
            a, b = wall.get_segment()
        else:
            a, b = wall
        if _distance_to_segment(pos, a, b) < radius:
            return False
    return True


def side_by_side_starts(maze, start: Point, toward: Point, radius: float) -> List[Point]:
    """Two start positions for ships of `radius`: either side of `start`, across the way to `toward` (the exit), when
    both fit there; otherwise the nearest clear spot to `start` for the second ship. A start cell too narrow for
    both leaves the pair overlapping at `start`, and the bounce parts them."""
    gap = radius + 1.0
    base = math.atan2(toward[1] - start[1], toward[0] - start[0]) + math.pi / 2
    # Directions across the way first, then the others, nearest to across first
    turns = sorted(range(16), key=lambda i: min(i, 16 - i) if i not in (0, 8) else -1)
    for i in turns:
        angle = base + i * math.pi / 8
        d = (math.cos(angle), math.sin(angle))
        a = (start[0] - d[0] * gap, start[1] - d[1] * gap)
        b = (start[0] + d[0] * gap, start[1] + d[1] * gap)
        if circle_clear(maze, a, radius) and circle_clear(maze, b, radius):
            return [a, b]
    if circle_clear(maze, start, radius):
        for ring in (2.0, 3.0, 4.0):
            for i in turns:
                angle = base + i * math.pi / 8
                spot = (start[0] + math.cos(angle) * gap * ring, start[1] + math.sin(angle) * gap * ring)
                if circle_clear(maze, spot, radius):
                    return [start, spot]
    return [start, start]
