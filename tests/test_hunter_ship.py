import pytest
import config
from tests.hunter_fixture import HunterRig
from hunter.model import ACTIONS, NEUTRAL, FiringSolution, HunterSettings

MORTAL = HunterSettings(indestructible=False)
# Fire control reporting an enemy dead ahead; without it the hunter holds fire.
LINED_UP = FiringSolution(0.0, True)


@pytest.mark.parametrize('name', list(ACTIONS))
def test_simultaneous_controls(name):
    action = ACTIONS[name]
    ship = HunterRig((200,200))
    shots = ship.step(1, action, LINED_UP)
    assert (shots is not None) == action.fire
    assert ship.angle == (action.turn * config.SETTINGS.hunterShip.rotationSpeed
                          * ship.settings.turn_rate_multiplier) % 360
    assert (abs(ship.vx) + abs(ship.vy) > 0) == action.thrust
    if shots:
        assert [shot.source for shot in shots] == ['hunter']
        assert ship.step(1, action, LINED_UP) is None


def test_coasting_keeps_momentum_and_separated_hits_kill():
    ship = HunterRig((200,200), MORTAL)
    ship.vx = 2
    ship.step(1, NEUTRAL)
    assert ship.x > 200
    for remaining in (2,1,0):
        assert ship.take_damage() == (remaining == 0)
        assert ship.health == remaining
        ship.take_damage()
        assert ship.health == remaining
        ship.step(config.FPS * 0.51, NEUTRAL)
    assert not ship.active
    assert ship.step(1, ACTIONS['none_thrust_fire']) is None


def test_a_destroyed_hunter_plays_its_death_rather_than_vanishing():
    ship = HunterRig((200,200), HunterSettings(indestructible=False, health=1))
    assert ship.take_damage()
    assert ship.is_dying
    ship.update_death(ship.DEATH_DURATION)
    assert not ship.is_dying


def test_default_hunter_is_indestructible():
    ship = HunterRig((200,200))
    for _ in range(10):
        assert not ship.take_damage()
        ship.step(config.FPS * 0.51, NEUTRAL)
    assert ship.active
    assert ship.health == ship.settings.health


def test_one_trigger_pull_fires_a_burst_of_three_then_cools_down():
    ship = HunterRig((200,200))
    fire,settings = ACTIONS['none_coast_fire'],ship.settings
    shots = [frame for frame in range(1,13)
             if ship.step(1, fire if frame == 1 else NEUTRAL, LINED_UP)]
    spacing = round(settings.burst_spacing*config.FPS)
    assert shots == [1, 1+spacing, 1+2*spacing]
    # Holding the trigger starts the next burst only after the cooldown.
    ship = HunterRig((200,200))
    held = [frame for frame in range(1,31) if ship.step(1, fire, LINED_UP)]
    cooldown = round(settings.fire_interval*config.FPS)
    second = 1+2*spacing+cooldown
    assert held == [1, 1+spacing, 1+2*spacing, second, second+spacing, second+2*spacing]
    assert HunterRig((200,200)).step(1, fire, LINED_UP)[0].source == 'hunter'


def test_cancelled_burst_fires_no_further_shots():
    ship = HunterRig((200,200))
    assert ship.step(1, ACTIONS['none_coast_fire'], LINED_UP)
    ship.cancel_burst()
    assert not any(ship.step(1, NEUTRAL, LINED_UP) for _ in range(30))
