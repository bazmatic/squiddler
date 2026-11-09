"""The Squidlet as a Creature: a small, quick squid that one hit kills, hatched from eggs and split from bosses."""
import math
from unittest.mock import Mock

import pytest

import config
import creatures
from creatures.creature import Creature
from entities.command_recorder import CommandRecorder, CommandType
from entities.squid import Squid
from tests.creature_parts import encounter, target, world

POS = (200.0, 300.0)


def thrusting_recorder():
    rec = CommandRecorder()
    for _ in range(config.REPLAY_ENEMY_WINDOW_SIZE):
        rec.record_command(CommandType.APPLY_THRUST)
    return rec


def test_the_catalogue_builds_a_squidlet():
    squidlet = creatures.build('baby', POS, 1, CommandRecorder())
    encounter(squidlet).step(1.0)
    assert isinstance(squidlet.body, Squid) and squidlet.radius == 10
    assert squidlet.hit_points == squidlet.max_hit_points == 1


@pytest.mark.parametrize('kind, share', [('baby', 0.5), ('replay', 0.45)])
def test_a_squidlet_flies_faster_than_a_squiddler(kind, share):
    squid = creatures.build(kind, POS, 1, thrusting_recorder())
    squid.angle = 0.0
    enc = encounter(squid, world_=world(ship=target((POS[0] - 2000.0, POS[1]))))
    for _ in range(120):
        enc.step(1.0)
    assert math.hypot(squid.vx, squid.vy) == pytest.approx(share * config.SHIP_MAX_SPEED * config.SHIP_FRICTION)   # friction after the cap


def test_a_hatching_egg_puts_squidlet_creatures_on_the_roster():
    egg = creatures.build('egg', POS, 1, CommandRecorder())
    enc = encounter(egg, world_=world(command_recorder=CommandRecorder()))
    enc.step(1.0)
    egg.radius = dict(egg.kind.traits)['grows']['to']
    enc.step(1.0)
    babies = [e for e in enc.roster.all() if e is not egg]
    assert babies and all(isinstance(e, Creature) and e.type == 'baby' for e in babies)


def test_a_squidlet_does_not_blink_when_rammed():
    squidlet = creatures.build('baby', POS, 1, CommandRecorder())
    ship = Mock(active=True, x=POS[0], y=POS[1], observer=False)
    ship.get_pos.return_value = POS
    ship.is_shield_active.return_value = False
    ship.check_circle_collision.return_value = True
    stand_in = world(ship=ship)
    encounter(squidlet, world_=stand_in).step(1.0)
    assert stand_in.scoring.record_enemy_collision.call_count == 1
    assert not squidlet.body.is_blinking
