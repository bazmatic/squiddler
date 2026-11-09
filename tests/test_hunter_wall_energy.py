"""A mortal hunter loses energy to walls in proportion to how hard it hits them."""
from dataclasses import replace
import pytest
from tests.hunter_fixture import HunterRig
from hunter.model import HunterSettings
from maze.wall_segment import WallSegment

MORTAL = replace(HunterSettings(), indestructible=False, health=10, wall_energy=4.0)


def crash(settings, speed_share):
    """Fly a hunter into a wall at `speed_share` of its top speed; return the ship."""
    ship = HunterRig((100.0, 100.0), settings)
    ship.prev_x, ship.prev_y = 100.0, 100.0
    ship.vx = speed_share * ship.max_speed
    ship.x = 100.0 + ship.vx
    wall = WallSegment((100.0 + ship.vx * 0.5, 0.0), (100.0 + ship.vx * 0.5, 200.0), 3)
    assert ship.check_wall_collision([wall])
    return ship


def test_a_hard_crash_costs_more_energy_than_a_gentle_one():
    hard, gentle = crash(MORTAL, 1.0), crash(MORTAL, 0.25)
    assert 10 - hard.health > 10 - gentle.health > 0
    assert (10 - hard.health) / (10 - gentle.health) > 2.0


def test_a_full_speed_crash_costs_about_the_setting_per_unit_of_speed_taken():
    ship = crash(MORTAL, 1.0)
    lost = 10 - ship.health
    taken = abs(ship.vx - ship.max_speed) / ship.max_speed         # the speed the wall took, as a share of top speed
    assert lost == pytest.approx(MORTAL.wall_energy * taken, rel=1e-6)


def test_energy_is_fractional_and_a_dead_hunter_is_inactive():
    ship = crash(MORTAL, 0.5)
    assert 0 < 10 - ship.health < 10 and 10 - ship.health != int(10 - ship.health)
    weak = replace(MORTAL, health=1)
    assert crash(weak, 1.0).active is False


def test_walls_cost_nothing_when_the_hunter_is_indestructible_or_the_cost_is_zero():
    assert crash(HunterSettings(), 1.0).health == HunterSettings().health
    assert crash(replace(MORTAL, wall_energy=0.0), 1.0).health == 10


def test_a_wall_crash_is_felt_as_a_hit_even_when_it_costs_no_energy():
    ship = crash(HunterSettings(), 1.0)
    hit = ship.take_hit()
    assert hit is not None and hit[0] < 0                          # shoved back from the wall


def test_the_neural_hunter_pays_for_walls_and_the_other_pilots_do_not():
    from hunter.settings_for import hunter_settings_for
    import config
    assert hunter_settings_for('neural').wall_energy == config.SETTINGS.game.neuralWallEnergy > 0
    assert hunter_settings_for('jev').wall_energy == hunter_settings_for('rules').wall_energy == 0
