"""Routes through the maze's cells: a 4-neighbour search, with which cells are linked cached per wall layout."""
import heapq
from collections import deque
from utils.math_utils import line_line_collision


class MazeRoutes:
    def __init__(self):
        self._links = {}
        self._links_walls = None
        self._gates, self._gate_blocks, self._gate_cells = (), {}, {}

    def _geometry(self, maze):
        sx, sy, ox, oy = maze.cell_size_x, maze.cell_size_y, maze.offset_x, maze.offset_y
        active = [w for w in maze.walls if w.active]
        # A destroyed block can expose as many faces as it loses, so count alone is not enough
        walls_key = (id(maze), len(active), getattr(maze, 'wall_generation', 0))
        if self._links_walls != walls_key:
            self._links, self._links_walls, self._gates = {}, walls_key, None     # a new layout: the gates are indexed afresh

        def cell(pos):
            return (min(maze.grid_width - 1, max(0, int((pos[0] - ox) // sx))),
                    min(maze.grid_height - 1, max(0, int((pos[1] - oy) // sy))))

        # Gates move independently of the wall layout, but not between the many route queries of one frame: which
        # cells each reaches is worked out when a gate moves, and a link is tested only against the gates in its cells.
        gates = tuple((tuple(g.start), tuple(g.end)) for g in getattr(maze, 'gate_segments', ()) if g.active)
        if gates != self._gates:
            self._gates, self._gate_blocks, self._gate_cells = gates, {}, {}
            for gate in gates:
                low = cell((min(gate[0][0], gate[1][0]), min(gate[0][1], gate[1][1])))
                high = cell((max(gate[0][0], gate[1][0]), max(gate[0][1], gate[1][1])))
                for cx in range(low[0], high[0] + 1):
                    for cy in range(low[1], high[1] + 1):
                        self._gate_cells.setdefault((cx, cy), []).append(gate)
        gate_cells = self._gate_cells

        def centre(c):
            return (ox + (c[0] + .5) * sx, oy + (c[1] + .5) * sy)

        def linked(a, b):
            # Judged by live wall segments, which follow the blocks as they are destroyed.
            key = (a, b) if a < b else (b, a)
            if key not in self._links:
                near = active
                if getattr(maze, 'spatial_grid', None) is not None:
                    near = maze.spatial_grid.get_walls_along_path(centre(a), centre(b), 1)
                self._links[key] = not any(
                    w.active and line_line_collision(centre(a), centre(b), w.start, w.end)
                    for w in near)
            if not self._links[key]:
                return False
            if key not in self._gate_blocks:
                self._gate_blocks[key] = any(line_line_collision(centre(a), centre(b), start, end)
                                             for start, end in gate_cells.get(a, []) + gate_cells.get(b, []))
            return not self._gate_blocks[key]
        return cell, centre, linked

    def cheapest_route(self, maze, origin, targets, hit_point_cells):
        """Like `route`, but breakable wall blocks may be shot through: entering one costs a cell plus
        `hit_point_cells` cells for each hit point it has left. Strong walls and the outer ring stay closed.
        Returns (index, route, extra): `extra` is what the blocks on the route add to its length, in pixels.
        None when no target can be reached even through the breakable blocks."""
        cell, centre, linked = self._geometry(maze)
        blocks = getattr(maze, 'blocks', None) or {}
        strong = getattr(maze, 'strong_cells', ()) or ()
        width, height = maze.grid_width, maze.grid_height
        hit_point = hit_point_cells * (maze.cell_size_x + maze.cell_size_y) / 2    # the same whichever way it is shot

        def breakable(c):
            return c not in strong and 0 < c[0] < width - 1 and 0 < c[1] < height - 1

        goals = {}
        for i, target in enumerate(targets):
            goals.setdefault(cell(target), i)
        start = cell(origin)
        best, came, queue = {start: 0.0}, {start: None}, [(0.0, start)]
        reached = None
        while queue:
            cost, here = heapq.heappop(queue)
            if cost > best[here]:
                continue
            if here in goals:
                reached = here
                break
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                there = (here[0] + dx, here[1] + dy)
                if not (0 <= there[0] < width and 0 <= there[1] < height):
                    continue
                step = maze.cell_size_x if dx else maze.cell_size_y
                if there in blocks:
                    if not breakable(there):
                        continue
                    step += blocks[there] * hit_point
                elif here not in blocks and not linked(here, there):
                    continue
                if cost + step < best.get(there, float('inf')):
                    best[there], came[there] = cost + step, here
                    heapq.heappush(queue, (cost + step, there))
        if reached is None:
            return None
        route, extra, step = [], 0.0, reached
        while step != start:
            route.append(centre(step))
            if step in blocks:
                extra += blocks[step] * hit_point
            step = came[step]
        return goals[reached], route, extra

    def reachable_targets(self, maze, origin, targets):
        """Indices of targets connected to origin by open cells, including the current live gate layout."""
        cell, _, linked = self._geometry(maze)
        start = cell(origin)
        seen, queue = {start}, deque([start])
        while queue:
            here = queue.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                there = (here[0] + dx, here[1] + dy)
                if (0 <= there[0] < maze.grid_width and 0 <= there[1] < maze.grid_height
                        and there not in seen and linked(here, there)):
                    seen.add(there)
                    queue.append(there)
        return [i for i, target in enumerate(targets) if cell(target) in seen]

    def route(self, maze, origin, targets):
        """The shortest cell route from `origin` to the nearest of `targets` (by route), as (index, route): the
        index of the target reached and the cell centres from the target's cell back to the one after the start
        (empty when they share a cell). None when no target can be reached."""
        cell, centre, linked = self._geometry(maze)

        goals = {}
        for i, target in enumerate(targets):
            goals.setdefault(cell(target), i)
        start = cell(origin)
        came, frontier = {start: None}, deque([start])
        reached = start if start in goals else None
        while frontier and reached is None:
            here = frontier.popleft()
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                there = (here[0] + dx, here[1] + dy)
                if (0 <= there[0] < maze.grid_width and 0 <= there[1] < maze.grid_height
                        and there not in came and linked(here, there)):
                    came[there] = here
                    frontier.append(there)
                    if there in goals:
                        reached = there
                        break
        if reached is None:
            return None
        route = []
        step = reached
        while step != start:
            route.append(centre(step))
            step = came[step]
        return goals[reached], route
