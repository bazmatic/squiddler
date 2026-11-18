"""Which completed block removals advance a necessary route to a campaign enemy."""
from hunter.maze_route import MazeRoutes


class BreachPlanner:
    def __init__(self):
        self._routes = MazeRoutes()
        self._key, self._cells = None, frozenset()

    def blocking_cells(self, maze, origin, targets):
        """Breakable blocks on the cheapest enemy route, only when no enemy has an open route.

        Cache the pre-break plan while geometry is unchanged. This also keeps one blast's planned blocks eligible
        during a batched cave rebuild, when several blocks disappear before the wall generation advances.
        """
        targets = tuple(tuple(p) for p in targets)
        gates = tuple((tuple(g.start), tuple(g.end)) for g in maze.gate_segments if g.active)
        key = (id(maze), maze.wall_generation, tuple(origin), targets, gates)
        if key != self._key:
            self._key, self._cells = key, frozenset()
            if targets and self._routes.route(maze, origin, targets) is None:
                from hunter.campaign_senses import CAMPAIGN_HIT_POINT_CELLS
                found = self._routes.cheapest_route(maze, origin, targets, CAMPAIGN_HIT_POINT_CELLS)
                if found is not None:
                    cells = (maze.position_calculator.screen_to_grid(*p) for p in found[1])
                    self._cells = frozenset(cell for cell in cells if cell in maze.blocks
                                            and cell not in maze.strong_cells)
        return self._cells
