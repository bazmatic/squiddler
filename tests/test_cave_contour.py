import math

import numpy as np
import pytest

from maze.cave_contour import _square_segments, tile_polylines
from maze.cave_field import THRESHOLD, SampleGrid

R = 40.0
CENTRE = (90.0, 90.0)


def _circle(per=8):
    samples = SampleGrid(3, 3, 60.0, 60.0, (0.0, 0.0), per, per)
    rows, cols = samples.shape
    iy, ix = np.mgrid[0:rows, 0:cols]
    x, y = ix*samples.h_x, iy*samples.h_y
    dist = np.hypot(x - CENTRE[0], y - CENTRE[1])
    return samples, THRESHOLD + 0.5*(R - dist)/R  # rock inside, 0.5 exactly at radius R


def _all(samples, values, tolerance=0.0):
    return {(c, r): tile_polylines(values, samples, (c, r), tolerance)
            for r in range(samples.rows) for c in range(samples.cols)}


def _lookup(samples, values, x, y):
    fy, fx = samples.fractional(x, y)
    iy, ix = min(int(fy), values.shape[0]-2), min(int(fx), values.shape[1]-2)
    ty, tx = fy-iy, fx-ix
    top = values[iy, ix]*(1-tx) + values[iy, ix+1]*tx
    bottom = values[iy+1, ix]*(1-tx) + values[iy+1, ix+1]*tx
    return top*(1-ty) + bottom*ty


def test_circle_traced_closely():
    samples, values = _circle()
    lines = [l for ls in _all(samples, values).values() for l in ls]
    assert lines
    for line in lines:
        for x, y in line:
            assert abs(math.hypot(x-CENTRE[0], y-CENTRE[1]) - R) < 0.5
    # The union is closed: every end point is matched by another polyline's end point
    ends = [p for l in lines for p in (l[0], l[-1])]
    for p in ends:
        assert sum(1 for q in ends if math.dist(p, q) < 1e-6) == 2


def test_rock_on_the_right():
    samples, values = _circle()
    for ls in _all(samples, values).values():
        for line in ls:
            for (x0, y0), (x1, y1) in zip(line, line[1:]):
                dx, dy = x1-x0, y1-y0
                n = math.hypot(dx, dy)
                right = (-dy/n, dx/n)  # screen coordinates, y down: clockwise on screen
                mx, my = (x0+x1)/2 + 2*right[0], (y0+y1)/2 + 2*right[1]
                assert _lookup(samples, values, mx, my) > THRESHOLD


def test_tiles_join_exactly():
    samples, values = _circle()
    tiles = _all(samples, values)
    closed_seen = False
    for (c, r), ls in tiles.items():
        for line in ls:
            if math.dist(line[0], line[-1]) < 1e-9:
                closed_seen = True
                continue
            for end in (line[0], line[-1]):
                found = any(math.dist(end, q) < 1e-6
                            for nc, nr in ((c+1, r), (c-1, r), (c, r+1), (c, r-1))
                            for other in tiles.get((nc, nr), [])
                            for q in (other[0], other[-1]))
                assert found
    assert not closed_seen  # the circle crosses tile borders, so nothing is closed within a tile


def test_simplify_reduces_vertices():
    samples, values = _circle(16)
    raw = _all(samples, values, 0.0)
    simple = _all(samples, values, 0.75)
    n_raw = sum(len(l) for ls in raw.values() for l in ls)
    n_simple = sum(len(l) for ls in simple.values() for l in ls)
    assert n_simple < n_raw / 2
    for tile, ls in simple.items():
        raw_points = {p for l in raw[tile] for p in l}
        assert len(ls) == len(raw[tile])
        for line in ls:
            assert all(p in raw_points for p in line)


def test_closed_loop_inside_a_tile():
    samples = SampleGrid(1, 1, 40.0, 40.0, (0.0, 0.0), 8, 8)
    values = np.zeros(samples.shape)
    values[3:6, 3:6] = 1.0
    lines = tile_polylines(values, samples, (0, 0), 0.0)
    assert len(lines) == 1 and lines[0][0] == lines[0][-1]
    assert _lookup(samples, values, 20.0, 20.0) > THRESHOLD


def _mids(segments):
    return sorted(((a[1][0]+b[1][0])/2, (a[1][1]+b[1][1])/2) for a, b in segments)


def test_saddle_decided_by_centre():
    samples = SampleGrid(1, 1, 1.0, 1.0, (0.0, 0.0), 1, 1)
    # Rock on TL and BR with centre average 0.5: the rock joins, each segment cuts off an open corner
    joined = _mids(_square_segments(np.array([[1.0, 0.0], [0.0, 1.0]]), samples, 0, 0))
    assert np.array(joined) == pytest.approx(np.array([(0.25, 0.75), (0.75, 0.25)]))
    # Centre average 0.3: the rock splits, each segment cuts off a rock corner
    split = _mids(_square_segments(np.array([[0.6, 0.0], [0.0, 0.6]]), samples, 0, 0))
    assert split[0][0] < 0.5 and split[0][1] < 0.5 and split[1][0] > 0.5 and split[1][1] > 0.5
