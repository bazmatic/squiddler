"""Ship effects are visible, finite, and isolated from simulation state."""
import random

import pygame
import pytest

import creatures
from entities.ship import Ship
from tests.creature_parts import encounter
from tests.test_exit_policies_game import game, start


def pixels(draw):
    surface = pygame.Surface((240, 240), pygame.SRCALPHA)
    draw(surface)
    return pygame.image.tostring(surface, 'RGBA')


def test_player_flinch_moves_only_the_rendered_hull_and_expires():
    ship = Ship((120, 120))
    ship.shield_initial_timer = 0
    before = ship.x, ship.y, ship.vx, ship.vy
    normal = pixels(ship.draw)
    ship.flinch()
    ship.impact.update(3)
    assert pixels(ship.draw) != normal
    assert (ship.x, ship.y, ship.vx, ship.vy) == before
    ship.impact.update(30)
    assert ship.impact.remaining == 0


def test_shielded_player_does_not_flinch():
    ship = Ship((120, 120))
    ship.shield_active = True
    ship.flinch()
    assert ship.impact.remaining == 0


def test_hunter_only_flinches_on_actual_damage():
    from creatures.traits.energy import damage
    hunter = creatures.build('allied_hunter', (120, 120))
    enc = encounter(hunter)
    # Energy is normally initialized by the pilot setup; exercise the real damage event here.
    hunter.indestructible = False
    hunter.health, hunter.immunity_remaining, hunter.damage_immunity = 3, 0, 0.5
    damage(hunter, enc)
    assert hunter.body.impact.remaining > 0
    hunter.body.update(hunter, 3)
    remaining = hunter.body.impact.remaining
    damage(hunter, enc)
    assert hunter.health == 2 and hunter.body.impact.remaining == remaining


@pytest.mark.parametrize('kind', ['player', 'hunter'])
def test_ship_death_animates_then_disappears_without_using_randomness(kind):
    ship = Ship((120, 120)) if kind == 'player' else creatures.build('allied_hunter', (120, 120))
    ship.die()
    state = random.getstate()
    first = pixels(ship.draw_death)
    ship.update_death(20)
    assert pixels(ship.draw_death) != first
    ship.update_death(ship.DEATH_DURATION)
    assert not ship.is_dying
    assert not any(pixels(ship.draw_death))
    assert random.getstate() == state


def test_game_over_plays_player_death_even_while_the_game_is_frozen(game):
    start(game, 1)
    game._start_game_over_sequence(__import__('time').time())
    assert not game.ship.active and game.ship.is_dying
    remaining = game.ship.death_timer
    game.update(1)
    assert game.ship.death_timer < remaining
    game.draw_game()


def test_enemy_bullet_triggers_player_flinch(game):
    from entities.projectile import Projectile
    start(game, 1)
    game.ship.shield_initial_timer = 0
    game.ship.shield_active = False
    bullet = Projectile(game.ship.get_pos(), 0, is_enemy=True)
    assert game.collision_handler.handle_projectile_ship_collision(bullet, game.ship, game.scoring)
    assert game.ship.impact.remaining > 0


def test_game_over_waits_for_the_player_death(game, monkeypatch):
    start(game, 1)
    game._start_game_over_sequence(1000.0)
    monkeypatch.setattr(game, 'clock', lambda: game.game_over_start_time + 0.5)   # half a second in
    game.update(30)
    assert game.game_over_active and game.ship.is_dying
    assert game.ship.death_timer == 30
    assert game.game_over_death_duration == 1.0
