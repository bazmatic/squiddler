"""Completed, necessary breakthroughs pay; damage, side projects and other actors do not."""
import pytest

from experiments.campaign_rollout import CampaignShaping, fly_campaign
from learning.campaign_breach import BreachPlanner
from learning.campaign_rewards import TABLE
from maze.config import MazeComplexity
from maze.generator import Maze
from tests.test_campaign_training import (ScriptedPilot, spec, undo_harness_side_effects)  # noqa: F401


@pytest.mark.parametrize('curved', [False, True])
@pytest.mark.parametrize('case,points', [('partial', 0), ('destroyed', 3), ('open_route', 0),
                                       ('irrelevant', 0), ('ally', 0), ('enemy', 0), ('no_enemies', 0)])
def test_campaign_pays_only_a_player_completed_necessary_block(spec, curved, case, points):
    target = (4, 5) if case != 'irrelevant' else (2, 2)

    def script(game, decision):
        maze = game.maze
        if decision == 0:
            maze.curved = curved
            for y, row in enumerate(maze.grid):
                for x in range(len(row)):
                    row[x] = int(x in (0, maze.grid_width-1) or y in (0, maze.grid_height-1)
                                 or x == 4 or (x,y) == (2,2))
            maze.rebuild_walls()
            centre = maze.position_calculator.grid_center_to_screen
            game.ship.x, game.ship.y = centre(2,5)
            enemies = game.roster.alive()
            enemies[0].x, enemies[0].y = centre(7,5)
            if case == 'no_enemies':
                game.encounter.kill(enemies[0])
            for i, other in enumerate(enemies[1:]):
                if case == 'open_route' and i == 0:
                    other.x, other.y = centre(2,7)
                else:
                    game.encounter.kill(other)
            maze.blocks[target] = 2
            for wall in maze.walls:
                if wall.cell == target:
                    wall.hit_points = 2
        if decision in (4, 6) and not (case == 'partial' and decision == 6):
            wall = next(w for w in maze.walls if w.cell == target and w.active)
            source = {'ally':'hunter', 'enemy':'enemy'}.get(case, 'player')
            destroyed = maze.damage_wall(wall, source=source)
            assert destroyed == (decision == 6)

    shaping = CampaignShaping(**{key: 0.0 for key in TABLE.defaults if key != 'breach_block'}, breach_block=3)
    episode = fly_campaign(1, 300, spec, horizon=.2, pilot=ScriptedPilot(script), shaping=shaping)
    assert episode['rewards'].sum() == pytest.approx(points)
    assert episode['stats'].get('breach_blocks', 0) == (1 if points else 0)


@pytest.mark.parametrize('curved', [False, True])
def test_a_multi_block_breach_keeps_credit_through_a_batched_rebuild(curved):
    maze = Maze(1, MazeComplexity.EMPTY, 10, curved=curved)
    for y in range(1,9):
        maze.grid[y][4] = maze.grid[y][5] = 1
    maze.rebuild_walls()
    centre = maze.position_calculator.grid_center_to_screen
    planner = BreachPlanner()
    awarded = []

    def destroyed(cell, source):
        if source == 'player' and cell in planner.blocking_cells(maze, centre(2,5), [centre(7,5)]):
            awarded.append(cell)

    maze.on_block_destroying = destroyed
    with maze.batched_destroys():
        for cell in ((4,5), (5,5)):
            wall = next(w for w in maze.walls if w.cell == cell and w.active)
            assert maze.damage_wall(wall, damage=maze.blocks[cell], source='player')
    assert awarded == [(4,5), (5,5)]
    assert planner.blocking_cells(maze, centre(2,5), [centre(7,5)]) == frozenset()
