"""A swept-path query must return every wall the path can touch, however long or slanted the path."""
import math
import random

import pytest

from utils.spatial_grid import SpatialGrid


class Wall:
    active = True

    def __init__(self, start, end):
        self.start, self.end = start, end

    def get_segment(self):
        return self.start, self.end


def _touches(wall, a, b, radius):
    from creatures.wallcrawler_team import capsule_hits
    return capsule_hits(a, b, radius, wall.start, wall.end)


@pytest.mark.parametrize('seed', range(3))
def test_long_slanted_paths_still_find_every_wall_they_touch(seed):
    rng = random.Random(seed)
    grid = SpatialGrid(2000, 1200, 50.0)
    walls = []
    for _ in range(400):
        x, y, angle = rng.uniform(0, 2000), rng.uniform(0, 1200), rng.uniform(0, math.tau)
        walls.append(Wall((x, y), (x + 60 * math.cos(angle), y + 60 * math.sin(angle))))
    grid.add_walls(walls)
    for _ in range(300):
        a = (rng.uniform(0, 2000), rng.uniform(0, 1200))
        angle, length = rng.uniform(0, math.tau), rng.choice([20, 300, 1200, 2500])
        b = (a[0] + length * math.cos(angle), a[1] + length * math.sin(angle))
        radius = rng.choice([0.0, 1.0, 18.0, 30.0])
        found = set(grid.get_walls_along_path(a, b, radius))
        assert all(w in found for w in walls if _touches(w, a, b, radius))


def test_long_diagonal_path_skips_walls_far_from_it():
    grid = SpatialGrid(2000, 1200, 50.0)
    near, far = Wall((1000, 600), (1010, 610)), Wall((1900, 100), (1910, 110))
    grid.add_walls([near, far])
    found = grid.get_walls_along_path((0, 0), (2000, 1200), 1.0)
    assert near in found and far not in found


def test_zero_length_path_with_large_radius():
    grid = SpatialGrid(500, 500, 50.0)
    wall = Wall((240, 240), (260, 260))
    grid.add_walls([wall])
    assert wall in grid.get_walls_along_path((250, 250), (250, 250), 120.0)
