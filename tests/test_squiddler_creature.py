"""The Squiddler as a Creature: it flies the player's recorded moves, faces its target when the recording is idle, and
fires where it faces when it is lined up."""
import copy
import math
from types import SimpleNamespace
from unittest.mock import Mock

import pygame
import pytest

import config
import creatures
import level_rules
from creatures.traits.combat import FlinchesWhenHit
from entities.command_recorder import CommandRecorder, CommandType
from entities.projectile import Projectile
from entities.squid import Squid
from tests.creature_parts import encounter, target, world

POS = (400.0, 300.0)


def recorder(command=None):
    """A recording full of one command (a full window), or an empty one."""
    rec = CommandRecorder()
    if command is not None:
        for _ in range(config.REPLAY_ENEMY_WINDOW_SIZE):
            rec.record_command(command)
    return rec


def squiddler(rec=None, angle=0.0, pos=POS):
    squid = creatures.build('replay', pos, 1, rec if rec is not None else recorder())
    squid.angle = angle
    return squid


def far_away():
    """A target well out of the gun's range."""
    return target((POS[0] + 2000.0, POS[1]))


def run(squid, frames, stand_in=None):
    stand_in = stand_in if stand_in is not None else world(ship=far_away())
    enc = encounter(squid, world_=stand_in)
    for _ in range(frames):
        enc.step(1.0)
    return stand_in


def test_the_catalogue_builds_a_squiddler():
    squid = squiddler()
    assert isinstance(squid.body, Squid) and squid.radius == 15
    run(squid, 1)
    assert squid.hit_points == squid.max_hit_points == 5


def test_gun_level_one_strength_is_the_old_squid_gun():
    strength = level_rules.get_enemy_strength(1)
    assert (strength.fire_interval_min, strength.fire_interval_max, strength.fire_range) == (60, 300, 400)
    late = creatures.build('replay', POS, 30, recorder())
    run(late, 1)
    assert late.gun_range == 400


def test_a_squiddler_replays_the_recording():
    turning = squiddler(recorder(CommandType.ROTATE_RIGHT))
    run(turning, 10)
    assert turning.angle == pytest.approx(10 * config.SHIP_ROTATION_SPEED)

    jetting = squiddler(recorder(CommandType.APPLY_THRUST))
    run(jetting, 10)
    assert math.hypot(jetting.vx, jetting.vy) > 0 and jetting.body.jet > 0


def test_on_no_action_it_turns_to_face_its_target():
    goal = (POS[0], POS[1] + 300.0)
    squid = squiddler(recorder(CommandType.NO_ACTION), angle=0.0)
    run(squid, 30, world(ship=target(goal)))
    bearing = math.degrees(math.atan2(goal[1] - squid.y, goal[0] - squid.x))
    assert abs((squid.angle - bearing + 180) % 360 - 180) < config.SHIP_ROTATION_SPEED


def test_it_fires_where_it_faces_when_lined_up():
    ahead = run(squiddler(angle=0.0), 1, world(ship=target((POS[0] + 200.0, POS[1]))))
    [shot] = ahead.projectiles
    assert shot.is_enemy and shot.vx > 0 and abs(shot.vy) < 1e-6
    behind = run(squiddler(angle=0.0), 30, world(ship=target((POS[0] - 200.0, POS[1]))))
    assert behind.projectiles == []


def test_before_the_window_fills_a_squiddler_only_coasts():
    squid = squiddler(recorder(), angle=0.0)
    squid.vx = 1.0
    run(squid, 30)
    assert squid.x > POS[0] and squid.angle == 0.0


def test_with_no_target_a_squiddler_turns_nowhere_and_holds_fire():
    squid = squiddler(recorder(CommandType.NO_ACTION), angle=33.0)
    stand_in = run(squid, 30, world())
    assert squid.angle == 33.0 and stand_in.projectiles == []


def test_a_squiddler_blinks_when_hit_rammed_or_bounced():
    hit = squiddler()
    enc = encounter(hit, world_=world(ship=far_away()))
    enc.step(1.0)
    enc.shoot(Projectile(hit.get_pos(), 0, source='player'))
    assert hit.active and hit.body.is_blinking

    rammed = squiddler()
    ship = Mock(active=True, x=POS[0], y=POS[1], observer=False)
    ship.get_pos.return_value = POS
    ship.is_shield_active.return_value = False
    ship.check_circle_collision.return_value = True
    run(rammed, 1, world(ship=ship))
    assert rammed.body.is_blinking

    bounced = squiddler()
    bounced.vx = 6.0
    wall = ((POS[0] + 20.0, 0.0), (POS[0] + 20.0, 600.0))
    run(bounced, 5, world(ship=far_away(), maze=SimpleNamespace(walls=[wall], spatial_grid=None)))
    assert bounced.body.is_blinking and bounced.x < POS[0] + 20.0

    plain = squiddler()
    plain.traits = tuple(t for t in plain.traits if not isinstance(t, FlinchesWhenHit))
    enc = encounter(plain, world_=world(ship=far_away()))
    enc.step(1.0)
    enc.shoot(Projectile(plain.get_pos(), 0, source='player'))
    assert plain.active and not plain.body.is_blinking


def test_a_dead_squiddler_copies_for_its_portrait():
    squid = squiddler()
    enc = encounter(squid, world_=world(ship=far_away()))
    for _ in range(5):
        enc.step(1.0)
    enc.kill(squid)
    portrait = copy.deepcopy(squid)
    assert portrait.body.tentacles and portrait.body.tentacles[0] is not squid.body.tentacles[0]
    portrait.update_death(5.0)
    screen = pygame.Surface((800, 600), pygame.SRCALPHA)
    portrait.draw_death(screen)
    assert pygame.mask.from_surface(screen, 1).count() > 0
