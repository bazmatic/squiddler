"""Player inventory preserves ammo and edge-triggered controls across real game frames."""
from collections import defaultdict
from unittest.mock import Mock

import pygame
import pytest
import config
from weapons import CATALOG, WeaponInventory
from weapons.catalog import REPLACEMENT_RULESET
from entities.projectile import Projectile
from tests.test_level_arc import game


def ammo(state):
    return {view.id:view.ammo for view in state.inventory}


def test_switching_retains_ammo_and_cannot_bypass_cooldown():
    state=WeaponInventory()
    state.equip('railgun')
    state.try_fire((100,100),0)
    state.equip('scattergun')
    assert ammo(state)=={'basic':None,'railgun':11,'scattergun':20}
    assert not state.try_fire((100,100),0).projectiles
    assert state.cycle() and state.weapon_id=='basic'
    assert not state.cycle() and state.weapon_id=='basic'
    state.cycle(False)
    state.cycle()
    assert state.weapon_id=='railgun' and state.ammo==11


def test_empty_slot_is_kept_skipped_and_refilled_without_duplicates():
    state=WeaponInventory()
    state.equip('railgun')
    for _ in range(12):
        state.try_fire((100,100),0)
        state.update(100)
    assert state.weapon_id=='basic' and ammo(state)['railgun']==0
    assert not state.cycle()
    state.equip('railgun')
    assert ammo(state)=={'basic':None,'railgun':12}
    state.try_fire((100,100),0)
    state.equip('railgun')
    assert state.ammo==12 and len(state.inventory)==2


def test_shield_pickup_is_not_held_in_the_inventory(game):
    game.start_level()
    game.ship.equip_weapon('shield')
    assert ammo(game.ship.weapon)=={'basic':None}
    assert game.ship.weapon.weapon_id=='basic'


def play(game,monkeypatch):
    game.level=1
    game.start_level()
    game.state=config.STATE_PLAYING
    game.game_frozen=game.game_over_active=False
    game.level_start_time=0
    game.keys_pressed_at_start=set()
    game.player_has_moved=game.ship.game_started=True
    game.ship.shield_active=False
    game.ship.shield_initial_timer=0
    game.roster.clear()
    keys=defaultdict(bool)
    monkeypatch.setattr(pygame.key,'get_pressed',lambda:keys)
    game.input_handler=Mock(key_mappings={})
    game.input_handler.process_controller_input.return_value=[]
    game.input_handler.is_controller_shield_pressed.return_value=False
    game.input_handler.is_controller_fire_pressed.return_value=False
    return keys


def test_shield_starts_at_once_runs_a_fixed_time_and_ignores_fire_and_cycling(game,monkeypatch):
    keys=play(game,monkeypatch)
    ship=game.ship
    ship.shield_initial_timer=0
    ship.equip_weapon('railgun')
    ship.equip_weapon('shield')
    assert ship.is_shield_active() and ship.weapon.weapon_id=='railgun'
    keys[pygame.K_DOWN]=True
    game.update(1)
    assert ship.weapon.weapon_id=='basic' and ship.is_shield_active()
    keys[pygame.K_DOWN]=False
    fuel=ship.fuel
    ship.update(config.SETTINGS.powerups.shieldDurationSeconds*config.FPS-5)
    assert ship.is_shield_active() and ship.fuel==fuel
    ship.update(10)
    assert not ship.is_shield_active()


def test_shield_protects_while_it_runs(game,monkeypatch):
    play(game,monkeypatch)
    ship=game.ship
    ship.shield_initial_timer=0
    ship.shield_active=False
    ship.equip_weapon('shield')
    bullet=Projectile(ship.get_pos(),0,is_enemy=True)
    game.collision_handler.handle_projectile_ship_collision(bullet,ship,game.scoring)
    assert not ship.damaged
    ship.update(config.SETTINGS.powerups.shieldDurationSeconds*config.FPS+1)
    bullet=Projectile(ship.get_pos(),0,is_enemy=True)
    game.collision_handler.handle_projectile_ship_collision(bullet,ship,game.scoring)
    assert ship.damaged


def test_old_pickup_rules_keep_replacement_and_exclude_shield(game):
    game.combat_ruleset=REPLACEMENT_RULESET
    game.start_level()
    assert not isinstance(game.ship.weapon,WeaponInventory)
    assert 'shield' not in game.ship.weapon.catalog.drops
    game.ship.equip_weapon('railgun')
    game.ship.equip_weapon('scattergun')
    assert game.ship.weapon.weapon_id=='scattergun'


@pytest.mark.parametrize('controller',[False,True])
def test_continuing_and_retry_start_with_only_basic(game,controller):
    from game_handlers.state_handlers import LevelCompleteStateHandler
    game.level=1
    game.start_level()
    game.ship.equip_weapon('railgun')
    game.ship.fire()
    game.ship.equip_weapon('shield')
    game.level_complete_menu=Mock()
    game.level_complete_menu.get_selected_option.return_value='CONTINUE'
    handler=LevelCompleteStateHandler()
    if controller:
        game.input_handler=Mock()
        game.input_handler.is_controller_menu_confirm_pressed.return_value=True
        handler._handle_normal_controller(pygame.event.Event(pygame.JOYBUTTONDOWN,button=0),game)
    else:
        handler._handle_normal_keyboard(pygame.event.Event(pygame.KEYDOWN,key=pygame.K_RETURN),game)
    assert game.level==2 and ammo(game.ship.weapon)=={'basic':None}
    game.ship.equip_weapon('scattergun')
    game.restart_level()
    assert ammo(game.ship.weapon)=={'basic':None}


def test_k_changes_weapon_like_down(game,monkeypatch):
    keys=play(game,monkeypatch)
    ship=game.ship
    ship.shield_initial_timer=0
    ship.equip_weapon('railgun')
    keys[pygame.K_k]=True
    game.update(1)
    assert ship.weapon.weapon_id=='basic'
