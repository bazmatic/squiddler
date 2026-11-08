import json
import math
from dataclasses import replace

import pygame
import pytest

import config
from tests.hunter_fixture import HunterRig
from entities.projectile import Projectile
from hunter.duel import Combatant, DuelArena, DuelResult, build_duel, duel_spawns
from hunter.model import NEUTRAL, HunterSettings
from hunter.neural.controller import NeuralController
from hunter.controller import HunterController
from hunter.neural.policy import random_policy
from hunter.senses import FEATURE_NAMES
from hunter.settings_for import hunter_settings_for
from maze.config import MazeComplexity
from maze.generator import Maze

SETTINGS = replace(HunterSettings(), indestructible=False, health=3)


class ScriptedRules:
    """Stands in for HunterController: asks for an observation each tick, then flies the given action."""

    status = 'active'

    def __init__(self, action=NEUTRAL):
        self.action, self.seen, self.closed = action, [], False

    def controller(self, settings):
        return self                               # its own pilot

    def tick(self, observe, running=True):
        self.seen.append(json.loads(observe(0.0, 0).state_json))
        return self.action

    def close(self):
        self.closed = True


class ScriptedNeural:
    kind = 'neural'
    status = 'active'

    def __init__(self, action=NEUTRAL):
        self.action, self.seen, self.closed = action, [], False

    def controller(self, settings):
        return self

    def tick(self, senses_factory, running=True):
        self.seen.append(senses_factory((1, 0, 0), 0.0))
        return self.action

    def close(self):
        self.closed = True


def duel_creature(pos, settings):
    import creatures
    from hunter.build import PilotSetup
    return creatures.build('allied_hunter', pos, pilot_setup=PilotSetup(settings, None, None), angle=0)


@pytest.fixture
def maze():
    pygame.init()
    pygame.display.set_mode((320, 240))
    return Maze(1, complexity=MazeComplexity.EMPTY, grid_size=8)


def cell(maze, col, row):
    return maze.position_calculator.grid_center_to_screen(col, row)


@pytest.fixture
def arena(maze):
    a = duel_creature(cell(maze, 2, 3), SETTINGS)
    b = duel_creature(cell(maze, 5, 3), SETTINGS)
    rules, vibes = ScriptedRules(), ScriptedNeural()
    duel = DuelArena(maze, [Combatant('rules', a, rules), Combatant('vibes', b, vibes)], timeout_seconds=60.0)
    duel.scripted = (rules, vibes)
    return duel


def shot_on(hunter, owner):
    shot = Projectile((hunter.x, hunter.y), 0.0, source='hunter')
    shot.is_enemy = True
    return shot


def test_each_pilot_is_shown_the_other_hunter_as_an_enemy_and_only_its_bullets(arena):
    rules, vibes = arena.scripted
    rules_hunter, vibes_hunter = arena.sides[0].hunter, arena.sides[1].hunter
    arena.shots['vibes'].append(shot_on(rules_hunter, 'vibes'))        # a Vibes bullet near the rules hunter
    arena.shots['rules'].append(shot_on(rules_hunter, 'rules'))        # the rules hunter's own bullet
    arena.update(0.0)
    contacts = rules.seen[0]['visible_contacts']
    assert len(contacts) == 1 and contacts[0]['allegiance'] == 'enemy'
    assert len(rules.seen[0]['visible_projectiles']) == 1               # its own bullet is not a threat to it
    assert vibes.seen[0][FEATURE_NAMES.index('enemies_present')] == 1.0


def test_a_hunters_shot_damages_the_opponent(arena):
    rules_hunter, vibes_hunter = arena.sides[0].hunter, arena.sides[1].hunter
    arena.shots['rules'].append(shot_on(vibes_hunter, 'rules'))         # rules fires at Vibes
    arena.update(0.0)
    assert vibes_hunter.health == SETTINGS.health - 1
    assert rules_hunter.health == SETTINGS.health
    assert arena.shots['rules'] == []                                   # the bullet is spent


def test_a_hunters_own_shot_never_damages_it_even_at_point_blank_range(arena):
    rules_hunter, vibes_hunter = arena.sides[0].hunter, arena.sides[1].hunter
    arena.shots['vibes'].append(shot_on(vibes_hunter, 'vibes'))         # Vibes' shot on top of Vibes
    arena.shots['rules'].append(shot_on(rules_hunter, 'rules'))         # and the same for the rules hunter
    arena.update(0.0)
    assert vibes_hunter.health == SETTINGS.health and rules_hunter.health == SETTINGS.health


def test_a_destroyed_hunter_ends_the_duel_once_and_everything_stops(arena):
    rules, vibes = arena.scripted
    rules_hunter, vibes_hunter = arena.sides[0].hunter, arena.sides[1].hunter
    vibes_hunter.health = 1
    arena.shots['rules'].append(shot_on(vibes_hunter, 'rules'))
    arena.update(0.0)
    assert arena.result == DuelResult('rules', 'destroyed') and not vibes_hunter.active
    result, elapsed, ticks = arena.result, arena.elapsed, len(rules.seen)
    for _ in range(3):
        arena.update(1.0)
    assert arena.result is result and arena.elapsed == elapsed and len(rules.seen) == ticks


def test_both_destroyed_in_the_same_frame_is_a_draw(arena):
    rules_hunter, vibes_hunter = arena.sides[0].hunter, arena.sides[1].hunter
    rules_hunter.health = vibes_hunter.health = 1
    arena.shots['rules'].append(shot_on(vibes_hunter, 'rules'))
    arena.shots['vibes'].append(shot_on(rules_hunter, 'vibes'))
    arena.update(0.0)
    assert arena.result == DuelResult(None, 'draw')


def test_the_duel_times_out_as_a_draw(maze):
    a, b = duel_creature(cell(maze, 2, 3), SETTINGS), duel_creature(cell(maze, 5, 3), SETTINGS)
    duel = DuelArena(maze, [Combatant('rules', a, ScriptedRules()), Combatant('vibes', b, ScriptedNeural())],
                     timeout_seconds=1.0)
    duel.update(30.0)
    assert duel.result is None
    duel.update(30.0)
    assert duel.result == DuelResult(None, 'timeout')


def test_bullets_stop_at_walls_and_do_not_survive_their_lifetime(arena):
    far = Projectile((arena.maze.walls[0].start[0], arena.maze.walls[0].start[1]), 0.0, source='hunter')
    arena.shots['rules'].append(far)
    far.lifetime = 1
    arena.update(1.0)
    assert not arena.shots['rules'] or not arena.shots['rules'][0].active
    assert far not in arena.shots['rules']


def test_a_burst_is_dropped_once_a_pilot_is_not_active_as_in_the_game(arena):
    rules, _ = arena.scripted
    rules.status = 'coasting'
    hunter, opponent = arena.sides[0].hunter, arena.sides[1].hunter
    opponent.x, opponent.y = cell(arena.maze, 4, 3)                                        # within a bullet's reach
    hunter.angle = math.degrees(math.atan2(opponent.y - hunter.y, opponent.x - hunter.x))   # opponent in its sights
    hunter.burst_remaining, hunter.fire_remaining = 3, 0.0
    arena.update(1.0)
    assert hunter.burst_remaining == 0 and arena.shots['rules'] == []


def test_close_closes_both_controllers(arena):
    arena.close()
    assert all(c.closed for c in arena.scripted)


def test_the_spawns_are_far_apart_but_within_sight_of_each_other(maze):
    from entities.ship import Ship
    sight = 700.0
    a, b = duel_spawns(maze, Ship(maze.start_pos), sight)
    assert 0.5 * sight < math.dist(a, b) <= sight


def test_the_spawns_are_never_close_in_an_ordinary_maze(maze):
    import random
    from entities.ship import Ship
    sight = 700.0
    for seed in range(25):
        a, b = duel_spawns(maze, Ship(maze.start_pos), sight, random.Random(seed))
        assert math.dist(a, b) >= 0.6 * sight


def test_in_a_dense_maze_the_spawns_still_have_a_clear_line_of_sight():
    import random
    from entities.ship import Ship
    from hunter.visibility import local_segments, visible_point
    pygame.init()
    pygame.display.set_mode((320, 240))
    random.seed(27)
    dense = Maze(27, complexity=MazeComplexity.EXTREME, grid_size=32)
    sight = 700.0
    a, b = duel_spawns(dense, Ship(dense.start_pos), sight)
    walls = local_segments(a, [(w.start, w.end) for w in dense.walls if w.active], sight)
    assert visible_point(a, b, walls, sight) and math.dist(a, b) <= sight
    assert math.dist(a, b) > 0.4 * sight                    # and not simply side by side


def test_build_duel_makes_a_destructible_rules_hunter_and_a_distinct_neural_one(maze):
    from entities.ship import Ship
    player = Ship(maze.start_pos)
    duel = build_duel(maze, player, rules_worker=object(), policy=random_policy(0), health=5,
                      timeout_seconds=120.0, decision_frames=6)
    rules, vibes = duel.sides
    assert (rules.name, vibes.name) == ('rules', 'vibes')
    assert isinstance(rules.flight.controller, HunterController) and isinstance(vibes.flight.controller, NeuralController)
    assert rules.hunter.health == vibes.hunter.health == 5
    assert not rules.hunter.settings.indestructible and not vibes.hunter.settings.indestructible
    assert rules.hunter.wall_energy == vibes.hunter.wall_energy == 0      # walls cost neither hunter in the duel
    assert rules.hunter.settings.turn_rate_multiplier == hunter_settings_for('rules').turn_rate_multiplier
    assert vibes.hunter.settings.turn_rate_multiplier == config.SETTINGS.game.neuralTurnRate
    assert rules.hunter.body.colours['color'] != vibes.hunter.body.colours['color']                      # you can tell them apart
    assert rules.hunter.body.colours['color'] == (120, 255, 90)                            # the class, so normal hunters, is unchanged
    toward = math.degrees(math.atan2(vibes.hunter.y - rules.hunter.y, vibes.hunter.x - rules.hunter.x)) % 360
    assert abs((rules.hunter.angle - toward + 180) % 360 - 180) < 1.0    # they start facing each other
    assert HunterSettings().indestructible is True                       # the default hunter is still unkillable


def test_ship_death_keeps_animating_after_the_duel_result(arena):
    victim = arena.sides[0].hunter
    victim.die()
    arena.result = DuelResult('vibes', 'destroyed')
    remaining = victim.death_timer
    elapsed = arena.elapsed
    arena.update(4)
    assert victim.death_timer == remaining - 4
    assert arena.elapsed == elapsed
