"""Squids and flighthouses soak up several shots; babies still pop in one."""
from unittest.mock import Mock

import config
import creatures
import enemy_kinds
from entities.command_recorder import CommandRecorder
from entities.projectile import Projectile
from game_handlers.collision_handler import CollisionHandler
from tests.creature_parts import encounter, world

POS = (400.0, 300.0)
SQUIDDLER_HIT_POINTS = dict(enemy_kinds.kind('replay').traits)['takesHits']['hitPoints']


def shoot(handler, squid, crystals=None):
    """A player's shot at a Creature, through an Encounter whose world reports to `handler`."""
    projectile = Projectile(POS, 0.0)
    projectile.source = 'player'
    stand_in = world(collision_handler=handler, scoring=handler.scoring, sound_manager=handler.sound_manager,
                     powerup_crystals=crystals if crystals is not None else [])
    return encounter(squid, world_=stand_in).shoot(projectile)


def make_handler():
    return CollisionHandler(Mock(), Mock(), CommandRecorder())


def make_squid():
    return creatures.build('replay', POS, 1, CommandRecorder())


def test_squid_takes_several_hits():
    assert SQUIDDLER_HIT_POINTS == 5
    squid = make_squid()
    encounter(squid).step(1.0)
    assert squid.hit_points == squid.max_hit_points == SQUIDDLER_HIT_POINTS


def test_squid_survives_until_its_last_hit_point():
    handler = make_handler()
    squid = make_squid()
    for _ in range(SQUIDDLER_HIT_POINTS - 1):
        assert shoot(handler, squid)  # The shot is still used up
        assert squid.active
        assert not squid.is_dying
    assert shoot(handler, squid)
    assert not squid.active
    assert squid.is_dying


def test_kill_is_only_credited_on_the_final_hit(monkeypatch):
    monkeypatch.setattr(config, 'POWERUP_CRYSTAL_SPAWN_CHANCE', 1.0)
    handler = make_handler()
    squid = make_squid()
    crystals = []
    for _ in range(SQUIDDLER_HIT_POINTS - 1):
        shoot(handler, squid, crystals)
    assert handler.sound_manager.play_enemy_destroy.call_count == 0
    assert handler.scoring.record_enemy_destroyed.call_count == 0
    assert crystals == []
    shoot(handler, squid, crystals)
    assert handler.sound_manager.play_enemy_destroy.call_count == 1
    assert handler.scoring.record_enemy_destroyed.call_count == 1
    assert len(crystals) == 1


def test_wounded_squid_flinches_and_shows_its_damage():
    handler = make_handler()
    squid = make_squid()
    encounter(squid).step(1.0)
    assert squid.body.get_damage_fraction() == 0.0
    squid.body.is_blinking = False
    shoot(handler, squid)
    assert squid.body.is_blinking
    assert 0.0 < squid.body.get_damage_fraction() < 1.0
    wounded = squid.body.get_damage_fraction()
    shoot(handler, squid)
    assert squid.body.get_damage_fraction() > wounded


def test_baby_among_the_squids_still_dies_in_one_hit():
    handler = make_handler()
    baby = creatures.build('baby', POS, 1, CommandRecorder())
    assert shoot(handler, baby)
    assert baby.max_hit_points == 1 and not baby.active


def test_flighthouse_takes_five_hits():
    handler, flighthouse = make_handler(), creatures.build('flighthouse', POS)
    alive = []
    for _ in range(5):
        shoot(handler, flighthouse)
        alive.append(flighthouse.active)
    assert flighthouse.max_hit_points == 5 and alive == [True, True, True, True, False]
