"""The Belemnite, and the blasts it makes when it dies: what they reach, what they do, and how they chain."""
import contextlib
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pygame
import pytest

import creatures
import enemy_kinds
import level_config
from creatures.traits.combat import Explodes
from entities.belemnite import BelemniteBody
from entities.explosion import Blast, Explosions, push, reaches
from entities.projectile import Projectile
from entities.ship import Ship
from game_handlers.collision_handler import CollisionHandler
from game_handlers.spawn_manager import make_enemy
from scoring.system import ScoringSystem
from tests.creature_parts import encounter, world
from tests.test_hunter_integration import game  # noqa: F401 - pytest fixture
from tests.test_layout_play import real_level_config

TRAITS = dict(enemy_kinds.kind('belemnite').traits)
HIT_POINTS = TRAITS['takesHits']['hitPoints']
BLAST_REACH, BLAST_FUSE = TRAITS['explodes']['reach'], TRAITS['explodes']['fuse']


def make(pos=(500.0, 500.0)):
    return creatures.build('belemnite', pos, 10)


def explodes(belemnite):
    return next(t for t in belemnite.traits if isinstance(t, Explodes))


def handler():
    scoring = ScoringSystem()
    scoring.start_level(1)
    return CollisionHandler(Mock(), scoring, Mock())


def shoot(handler, enemies, at, source='player'):
    """A friendly shot at `at`, met by the enemies."""
    return shoot_with(handler, enemies, Projectile(at, 0.0, source=source))


def go_off(handler, enemies, ship=None, hunter=None, maze=None, frames=60):
    """Burn every fuse and apply every blast, as the game does each frame, until nothing is left lit."""
    for _ in range(frames):
        for blast in handler.explosions.update(1.0):
            handler.handle_blast(blast, ship, hunter, [e for e in enemies if e.active],
                                 lambda shot: shoot_with(handler, enemies, shot), maze)


def shoot_with(handler, enemies, shot):
    """The shot offered to the enemies through an Encounter whose world reports to `handler`."""
    stand_in = world(collision_handler=handler, scoring=handler.scoring, sound_manager=handler.sound_manager)
    return encounter(*enemies, world_=stand_in).shoot(shot)


# --- the blast itself ---

def test_a_blast_reaches_circles_that_overlap_it_and_pushes_them_away_less_at_its_edge():
    blast = Blast((0.0, 0.0), 50.0, push=10.0)
    assert reaches(blast, (55.0, 0.0), 6.0) and not reaches(blast, (60.0, 0.0), 6.0)
    near, far = push(blast, (10.0, 0.0)), push(blast, (0.0, 40.0))
    assert near == pytest.approx((8.0, 0.0)) and far == pytest.approx((0.0, 2.0))
    assert push(blast, (80.0, 0.0)) == (0.0, 0.0)


def test_a_blast_goes_off_when_its_fuse_has_burnt_and_leaves_a_fireball():
    explosions = Explosions()
    explosions.trigger(Blast((0.0, 0.0), 50.0, fuse=8.0))
    explosions.trigger(None)
    assert explosions.update(5.0) == [] and explosions.pending
    [blast] = explosions.update(4.0)
    assert blast.radius == 50.0 and len(explosions.fireballs) == 1
    for _ in range(60):
        explosions.update(1.0)
    assert not explosions.pending


def test_a_fireball_draws_and_leaves_the_screen_opaque():
    pygame.init()
    screen = pygame.Surface((300, 300), pygame.SRCALPHA)
    screen.fill((0, 0, 0, 255))
    explosions = Explosions()
    explosions.trigger(Blast((150.0, 150.0), 60.0))
    explosions.update(1.0)
    for _ in range(4):
        explosions.draw(screen)
        explosions.update(4.0)
    assert screen.get_at((150, 150))[:3] != (0, 0, 0)
    assert all(screen.get_at((x, y))[3] == 255 for x in range(0, 300, 7) for y in range(0, 300, 7))


# --- the Belemnite ---

def test_it_is_a_kind_a_level_can_place_made_like_any_other_enemy():
    assert 'belemnite' in enemy_kinds.kinds() and 'belemnite' not in enemy_kinds.hunters()
    belemnite = make_enemy('belemnite', (300.0, 300.0), 10, None)
    assert isinstance(belemnite.body, BelemniteBody) and belemnite.type == 'belemnite'


def chase(enemies, target, frames):
    ship = Ship(target)
    ship.shield_active = True
    stand_in = world(ship=ship)
    enc = encounter(*enemies, world_=stand_in)
    for _ in range(frames):
        enc.step(1.0)
    return stand_in.projectiles


def test_it_chases_point_first_with_a_jellys_brain_and_never_fires():
    belemnite = make((200.0, 300.0))
    projectiles = chase([belemnite], (600.0, 300.0), 60)
    assert belemnite.x > 230.0 and projectiles == []
    assert abs((belemnite.angle + 180) % 360 - 180) < 10      # facing the way it swims: at the target


def test_belemnites_never_overlap_one_another_or_the_jellies():
    crowd = [make((200.0 + 3 * i, 300.0 + 2 * i)) for i in range(4)] + [creatures.build('aggressive', (205.0, 305.0), 10)]
    for _ in range(120):
        chase(crowd, (600.0, 300.0), 1)
        assert all(math.dist(a.get_pos(), b.get_pos()) >= a.radius + b.radius - 0.5
                   for i, a in enumerate(crowd) for b in crowd[i + 1:])


def test_it_takes_three_hits_and_lights_a_blast_that_counts_for_whoever_killed_it():
    h = handler()
    belemnite = make()
    assert shoot(h, [belemnite], belemnite.get_pos()) and shoot(h, [belemnite], belemnite.get_pos())
    assert belemnite.active and not h.explosions.pending
    shoot(h, [belemnite], belemnite.get_pos(), source='hunter')
    assert not belemnite.active and h.level_kills == 1
    [blast] = h.explosions.update(BLAST_FUSE)
    assert blast.source == 'hunter' and blast.pos == belemnite.get_pos()
    assert blast.radius == belemnite.radius * BLAST_REACH


def test_one_belemnite_sets_off_the_next_and_the_chain_counts_for_the_player():
    h = handler()
    reach = make().radius * BLAST_REACH
    chain = [make((100.0 + i * reach * 0.8, 300.0)) for i in range(4)]
    out_of_reach = make((100.0 + 3 * reach * 0.8 + reach * 3, 300.0))
    for _ in range(HIT_POINTS):
        shoot(h, [chain[0]], chain[0].get_pos())
    go_off(h, chain + [out_of_reach])
    assert not any(b.active for b in chain) and out_of_reach.active
    assert h.level_kills == 4 and h.scoring.enemies_destroyed == 4


def test_a_blast_kills_and_shoves_other_enemies_it_reaches():
    pygame.init()
    h = handler()
    urchin = creatures.build('static', (540.0, 500.0))
    belemnite = make()
    blast = explodes(belemnite).blast(belemnite, 'player')
    h.handle_blast(blast, None, None, [urchin], lambda shot: shoot_with(h, [urchin], shot))
    assert not urchin.active and h.level_kills == 1


def test_the_player_in_reach_is_shoved_and_pays_a_collision_unless_shielded():
    h = handler()
    ship = Ship((520.0, 500.0))
    ship.shield_active = False
    h.handle_blast(Blast((500.0, 500.0), 60.0, push=8.0), ship, None, [], lambda shot: False)
    assert ship.vx > 0 and h.scoring.enemy_collisions == 1
    shielded = Ship((520.0, 500.0))
    shielded.shield_active = True
    h.handle_blast(Blast((500.0, 500.0), 60.0, push=8.0), shielded, None, [], lambda shot: False)
    assert shielded.vx > 0 and h.scoring.enemy_collisions == 1
    far = Ship((700.0, 500.0))
    far.shield_active = False
    h.handle_blast(Blast((500.0, 500.0), 60.0), far, None, [], lambda shot: False)
    assert far.vx == 0 and h.scoring.enemy_collisions == 1


def test_a_blast_hits_each_wall_block_it_reaches_once():
    face = lambda cell, start, end: SimpleNamespace(active=True, cell=cell, get_segment=lambda: (start, end))
    near = [face((1, 1), (510.0, 480.0), (510.0, 520.0)), face((1, 1), (510.0, 480.0), (550.0, 480.0))]
    far = face((5, 5), (900.0, 0.0), (900.0, 40.0))
    hit = []
    maze = SimpleNamespace(spatial_grid=SimpleNamespace(get_nearby_walls=lambda pos, r: [*near, far]),
                           damage_wall=lambda wall, impact, *, source=None: hit.append((wall.cell, impact, source)) or False,
                           batched_destroys=contextlib.nullcontext)
    CollisionHandler._blast_walls(Blast((500.0, 500.0), 40.0, wall_hits=1), maze)
    assert hit == [((1, 1), (510.0, 500.0), 'enemy')]
    hit.clear()
    handler().handle_blast(Blast((500.0, 500.0), 40.0, wall_hits=0), None, None, [], lambda shot: False, maze)
    assert hit == []


def test_it_draws_point_first():
    pygame.init()
    surface = pygame.Surface((200, 200))
    belemnite = creatures.build('belemnite', (100.0, 100.0))
    belemnite.angle = 0.0
    belemnite.draw(surface)
    ahead = sum(1 for x in range(101, 125) if surface.get_at((x, 100))[:3] != (0, 0, 0))
    assert ahead > belemnite.radius        # the shell's point runs out ahead of it


# --- in a level ---

LAYOUT = {'walls': [], 'start': [1, 1], 'exit': [9, 9],
          'enemies': [{'type': 'belemnite', 'cell': [7, 2]}, {'type': 'belemnite', 'cell': [2, 7]}], 'beacon': None}


def test_a_layout_places_them_and_the_exit_counts_them(game, monkeypatch):  # noqa: F811
    real_level_config(monkeypatch)
    monkeypatch.setattr(level_config, 'load_level_config', lambda level: {
        'seed': 1, 'maze': {'complexity': 'empty', 'grid_size': 11}, 'layout': LAYOUT} if level == 3 else None)
    game.level = 3
    game.start_level()
    assert level_config.get_level_belemnite_count(3) == 2
    assert [e.type for e in game.roster.all()] == ['belemnite', 'belemnite'] and game.enemy_hunters == []
    assert game.roster.counts() == {'belemnite': 2}


def test_the_game_sets_off_a_belemnite_it_kills(game, monkeypatch):  # noqa: F811
    real_level_config(monkeypatch)
    monkeypatch.setattr(level_config, 'load_level_config', lambda level: {
        'seed': 1, 'maze': {'complexity': 'empty', 'grid_size': 11}, 'layout': LAYOUT} if level == 3 else None)
    game.level = 3
    game.start_level()
    belemnite = game.roster.of_kind('belemnite')[0]
    for _ in range(HIT_POINTS):
        game._hit_enemies(Projectile(belemnite.get_pos(), 0.0, source='player'))
    assert not belemnite.active
    for _ in range(20):
        game._update_explosions(1.0)
    assert game.collision_handler.explosions.fireballs


def test_a_level_file_without_a_layout_can_ask_for_them(game, monkeypatch):  # noqa: F811
    real_level_config(monkeypatch)
    monkeypatch.setattr(level_config, 'load_level_config', lambda level: {
        'seed': 1, 'maze': {'complexity': 'empty', 'grid_size': 12},
        'enemies': {'static': 0, 'patrol': 0, 'aggressive': 0, 'belemnite': 3}} if level == 3 else None)
    game.level = 3
    game.start_level()
    assert [e.type for e in game.roster.all()] == ['belemnite'] * 3
