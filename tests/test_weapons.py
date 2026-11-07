"""Weapon contracts exercised through state, real geometry, traits, and game integration."""
from dataclasses import replace
from unittest.mock import Mock
import random
import pygame
import pytest

import config
from weapons import CATALOG, EquippedWeapon, RULESET
from weapons.catalog import WeaponCatalog
from weapons.projectile import make_projectile
from weapons.simulation import advance
from maze.wall_segment import WallSegment
from entities.powerup_crystal import PowerupCrystal
from creatures.traits.combat import TakesHits
from tests.creature_parts import creature, encounter, world
from tests.test_level_arc import game


@pytest.mark.parametrize('weapon_id', [key for key in CATALOG.drops if key != 'shield'])
def test_replacement_ammo_and_final_volley(weapon_id):
    state = EquippedWeapon()
    state.equip(weapon_id)
    d = CATALOG[weapon_id]
    for count in range(d.capacity):
        result = state.try_fire((400,300),0)
        assert result.weapon_id == weapon_id and len(result.projectiles) == len(d.offsets)
        assert all(p.weapon_id == weapon_id for p in result.projectiles)
        assert not state.try_fire((400,300),0).projectiles
        if count < d.capacity-1:
            assert state.view.ammo == d.capacity-count-1
        state.update(d.interval)
    assert state.view.id == 'basic' and state.view.ammo is None
    assert state.try_fire((400,300),0).weapon_id == 'basic'


def test_same_weapon_replaces_ammo_and_retains_cooldown():
    state = EquippedWeapon()
    state.equip('railgun')
    shot = state.try_fire((400,300),0).projectiles[0]
    state.equip('railgun')
    assert state.view.ammo == 12 and not state.try_fire((400,300),0).projectiles
    state.equip('scattergun')
    assert state.view.ammo == 20 and state.view.cooldown == CATALOG['railgun'].interval
    assert shot.weapon_id == 'railgun'
    with pytest.raises(KeyError):
        state.equip('missing')
    assert state.view.id == 'scattergun'


def test_catalog_extension_needs_no_new_dispatch():
    d = replace(CATALOG['twin_blaster'], id='test_bolt', capacity=3)
    state = EquippedWeapon(WeaponCatalog([CATALOG['basic'],d]))
    state.equip(d.id)
    assert len(state.try_fire((400,300),0).projectiles) == 2
    assert state.view.ammo == 2
    with pytest.raises(ValueError):
        WeaponCatalog([CATALOG['basic'],d,d])


def test_twin_muzzles_are_parallel_and_scatter_fans():
    state = EquippedWeapon()
    state.equip('twin_blaster')
    a,b = state.try_fire((400,300),0).projectiles
    assert a.vx == b.vx and a.vy == b.vy and a.y != b.y
    state.update(100)
    state.equip('scattergun')
    shots = state.try_fire((400,300),0).projectiles
    assert [p.angle for p in shots] == [-20,-10,0,10,20]
    assert all(p.lifetime < CATALOG['basic'].lifetime for p in shots)


def scene(*targets, walls=()):
    w = world()
    enc = encounter(*targets, world_=w)
    enc.ready()
    w.maze.walls = list(walls)
    w.maze.damage_wall = Mock()
    return w,enc


def shot(weapon='railgun', pos=(400,300), angle=0):
    return make_projectile(pos, angle, 'player', CATALOG[weapon])


def test_rail_sweeps_targets_in_order_and_stops_at_indestructible_wall():
    first = creature([TakesHits(5)],pos=(430,300))
    second = creature([TakesHits(5)],pos=(470,300))
    behind = creature([TakesHits(5)],pos=(540,300))
    wall = WallSegment((510,250),(510,350),3,destructible=False)
    w,enc = scene(behind,second,first,walls=[wall])
    p = shot()
    advance(p,10,w.maze,enc.creatures(),enc.apply_hit,w.collision_handler.explosions.trigger)
    assert (first.hit_points,second.hit_points,behind.hit_points) == (3,3,5)
    assert not p.active
    w.maze.damage_wall.assert_called_once()


def test_rail_does_not_hit_same_creature_again_next_frame():
    target = creature([TakesHits(10)],pos=(430,300))
    w,enc = scene(target)
    p = shot()
    advance(p,1,w.maze,[target],enc.apply_hit,Mock())
    advance(p,1,w.maze,[target],enc.apply_hit,Mock())
    assert target.hit_points == 8


def test_grenade_bounces_then_explodes_once_without_wall_impact_damage():
    wall = WallSegment((440,250),(440,350),3)
    w,enc = scene(walls=[wall])
    p = shot('grenade_launcher')
    blasts=[]
    advance(p,10,w.maze,[],enc.apply_hit,blasts.append)
    assert p.vx < 0 and p.active and not blasts
    w.maze.damage_wall.assert_not_called()
    advance(p,100,w.maze,[],enc.apply_hit,blasts.append)
    assert len(blasts) == 1 and not p.active
    assert blasts[0].source == 'player' and blasts[0].weapon_id == 'grenade_launcher'
    assert blasts[0].fuse == 0 and blasts[0].damage == 2
    advance(p,100,w.maze,[],enc.apply_hit,blasts.append)
    assert len(blasts) == 1


def test_grenade_contact_and_fuse_same_step_do_not_duplicate():
    target = creature([TakesHits(10)],pos=(430,300))
    w,enc = scene(target)
    p = shot('grenade_launcher')
    p.lifetime=6
    blasts=[]
    advance(p,6,w.maze,[target],enc.apply_hit,blasts.append)
    assert len(blasts) == 1 and target.hit_points == 10


@pytest.mark.parametrize('dt',[.5,1,4,72])
def test_fuse_uses_elapsed_simulation_time(dt):
    w,enc=scene()
    p=shot('grenade_launcher')
    p.vx=p.vy=0
    blasts=[]
    for _ in range(round(72/dt)):
        advance(p,dt,w.maze,[],enc.apply_hit,blasts.append)
    assert len(blasts) == 1


def test_seeker_turn_is_bounded_and_wall_blocks_acquisition():
    target=creature([TakesHits(4)],pos=(500,360))
    w,enc=scene(target)
    p=shot('seeker')
    advance(p,1,w.maze,[target],enc.apply_hit,Mock())
    assert 0 < p.angle <= CATALOG['seeker'].turn_rate and p.target is target
    p=shot('seeker')
    w.maze.walls=[WallSegment((450,200),(450,450),3)]
    advance(p,1,w.maze,[target],enc.apply_hit,Mock())
    assert p.target is None and p.angle == 0


def test_normal_game_enables_shared_weapons_and_pickups(game):
    game.level=1
    game.start_level()
    assert game.ship.weapon.view.id == 'basic'
    pickup=PowerupCrystal(game.ship.get_pos(),'grenade_launcher')
    game.powerup_crystals=[pickup]
    game._update_powerup_crystals(1)
    assert game.ship.weapon.view.id == 'grenade_launcher' and not pickup.active
    assert game.collision_handler.level_crystals_collected == 1
    game.ship.ammo=0  # old level ammo stock cannot prevent firing
    projectiles=game.ship.fire()
    assert len(projectiles) == 1 and projectiles[0].weapon_id == 'grenade_launcher'
    game.start_level()
    assert game.ship.weapon.view.id == 'basic'


def test_a_hunter_fires_the_weapon_it_picks_up_until_the_ammo_runs_out():
    import creatures
    from hunter.build import PilotSetup
    from hunter.model import HunterSettings
    from tests.creature_parts import fire_round
    from game_handlers.collision_handler import CollisionHandler
    w=world(combat_ruleset=RULESET)
    hunter=creatures.build('allied_hunter',(400,300),pilot_setup=PilotSetup(HunterSettings(),None,None))
    hunter.traits=tuple(t for t in hunter.traits if t.NAME != 'pilot')
    enc=encounter(hunter,world_=w)
    enc.ready()
    handler=CollisionHandler(Mock(),Mock(),Mock())
    assert handler.handle_hunter_crystal_collision(hunter,PowerupCrystal(hunter.get_pos(),'scattergun'))
    volleys=[]
    for _ in range(5000):
        shots=fire_round(hunter,enc)
        if shots:
            volleys.append(shots)
        if hunter.weapon.weapon_id == 'basic':
            break
    assert len(volleys) == CATALOG['scattergun'].capacity
    assert all(len(v) == 5 and all(s.weapon_id == 'scattergun' for s in v) for v in volleys)
    # Out of ammo, it is back on its own gun
    for _ in range(100):
        shots=fire_round(hunter,enc)
        if shots:
            break
    assert shots and not hasattr(shots[0],'weapon_id')


def test_a_hunter_shield_pickup_makes_it_immune():
    import creatures
    from hunter.build import PilotSetup
    from hunter.model import HunterSettings
    from creatures.traits.energy import damage
    from game_handlers.collision_handler import CollisionHandler
    hunter=creatures.build('allied_hunter',(400,300),pilot_setup=PilotSetup(replace(HunterSettings(),indestructible=False),None,None))
    hunter.traits=tuple(t for t in hunter.traits if t.NAME != 'pilot')
    enc=encounter(hunter,world_=world(combat_ruleset=RULESET))
    enc.ready()
    health=hunter.health
    assert CollisionHandler(Mock(),Mock(),Mock()).handle_hunter_crystal_collision(hunter,PowerupCrystal(hunter.get_pos(),'shield'))
    damage(hunter,enc)
    assert hunter.health == health


def test_new_ruleset_episode_is_deterministic_with_recorded_tuning():
    from experiments import harness
    try:
        seen=[]
        first=harness.run_episode(1,7,horizon=2,combat_ruleset=RULESET,weapon_tuning=CATALOG.snapshot(),
                                  trace=lambda t,g,h: seen.append(h.weapon_pickups))
        second=harness.run_episode(1,7,horizon=2,combat_ruleset=RULESET,weapon_tuning=CATALOG.snapshot())
        assert first == second
        assert seen and all(seen)
    finally:
        harness.reset()


def test_observations_include_hunter_boost_and_pickup_lifetime():
    from tests.test_senses import sense,world as sensor_world,value
    from tests.hunter_fixture import HunterRig
    hunter=HunterRig((500,500))
    hunter.gun_upgrade_level=1
    hunter.upgrade_remaining=config.POWERUP_DURATION_SECONDS/2
    pickup=PowerupCrystal((520,500),'seeker')
    pickup.age=180
    v=sense(hunter,sensor_world(),powerups=[pickup])
    assert value(v,'power_boost') == 1
    assert value(v,'power_boost_remaining') == pytest.approx(.5)
    assert value(v,'powerup0_lifetime') == pytest.approx(.5)


def test_legacy_policy_adapter_preserves_logits_and_ignores_new_fields(tmp_path):
    import numpy as np
    from hunter.neural.policy import NeuralPolicy,random_policy
    from hunter.senses import FEATURE_NAMES,SENSE_SIZE
    policy=random_policy(1)
    width=FEATURE_NAMES.index('power_boost')
    policy.params['embed_w'][:,width:]=0
    path=tmp_path/'legacy.npz'
    params=dict(policy.params)
    params['embed_w']=params['embed_w'][:,:width]
    np.savez(path,feature_names=np.array(FEATURE_NAMES[:width]),mean=policy.mean[:width],scale=policy.scale[:width],**params)
    loaded=NeuralPolicy.load(path)
    assert loaded.legacy_observations
    x=np.ones(SENSE_SIZE,dtype=np.float32)
    expected,_=policy.step(x,policy.initial_state())
    actual,_=loaded.step(x,loaded.initial_state())
    for name in expected:
        assert np.array_equal(expected[name],actual[name])


def test_drop_selection_is_seeded_and_honours_allowlist():
    a,b=random.Random(10),random.Random(10)
    assert [CATALOG.choose_drop(a) for _ in range(30)] == [CATALOG.choose_drop(b) for _ in range(30)]
    assert {CATALOG.choose_drop(a,['railgun']) for _ in range(10)} == {'railgun'}
    with pytest.raises(ValueError):
        CATALOG.choose_drop(a,['unknown'])


def test_grenade_starts_belemnite_chain_with_weapon_attribution(game):
    from game_handlers.spawn_manager import make_enemy
    game.level=1
    game.start_level()
    game.roster.clear()
    a=make_enemy('belemnite',(600,400),1,game.command_recorder)
    b=make_enemy('belemnite',(660,400),1,game.command_recorder)
    c=make_enemy('static',(720,400),1,game.command_recorder)
    game.roster.extend([a,b,c])
    game.encounter.ready()
    a.hit_points=b.hit_points=c.hit_points=1
    projectile=shot('grenade_launcher',a.get_pos())
    game.collision_handler.explosions.trigger(projectile.detonate())
    for _ in range(30):
        game._update_explosions(1)
    assert not any(enemy.active for enemy in (a,b,c))
    assert game.collision_handler.weapon_kills['grenade_launcher'] == 3
    assert all(enemy.killed_by == 'player' for enemy in (a,b,c))


def test_neural_training_records_new_weapon_state_and_collection(tmp_path):
    import numpy as np
    from experiments import harness
    from experiments.rollout import fly_rollout
    from hunter.neural.policy import random_policy
    from experiments.student import StudentSpec
    path=tmp_path/'new.npz'
    random_policy(1).save(path)
    equipped=[False]
    def trace(t,game,hunter):
        if not equipped[0]:
            pickup=PowerupCrystal(hunter.get_pos(),'grenade_launcher')
            assert game.collision_handler.handle_hunter_crystal_collision(hunter,pickup)
            equipped[0]=True
    try:
        result=fly_rollout(1,3,StudentSpec(str(path)),horizon=2,combat_ruleset=RULESET,
                          weapon_tuning=CATALOG.snapshot(),upgrade_bonus=1,trace=trace)
        assert result['senses'].shape[0] > 0
        assert np.isfinite(result['rewards']).all()
        assert result['upgraded_fraction'] > .5
        assert result['powerups'] == 1
        assert result['weapon_pickups'] == {}
    finally:
        harness.reset()


def test_grenade_runs_through_the_real_game_loop(game):
    game.level=1
    game.start_level()
    game.state=config.STATE_PLAYING
    game.player_has_moved=True
    game.ship.game_started=True
    game.roster.clear()
    game.ship.equip_weapon('grenade_launcher')
    projectile=game.ship.fire()[0]
    game.projectiles.append(projectile)
    for _ in range(80):
        game.update(1)
    assert projectile.detonated and projectile not in game.projectiles
    assert game.collision_handler.explosions.fireballs


def rock_maze(walls, strong=()):
    from maze.config import MazeComplexity
    from maze.generator import Maze
    from level_layout import parse_layout
    layout = parse_layout({'walls': [list(c) for c in walls],
                           'strong_walls': [list(c) for c in strong],
                           'start': [1,1], 'exit': [9,9]},11)
    return Maze(1,MazeComplexity.EMPTY,11,layout=layout)


def test_rail_instantly_slices_adjacent_blocks_and_stops_at_metal():
    cells=[(3,5),(4,5),(5,5)]
    metal=(7,5)
    maze=rock_maze(cells,[metal])
    for cell in cells:
        maze.blocks[cell]=99
    for wall in maze.walls:
        if wall.cell in cells:
            wall.hit_points=99
    start=maze.position_calculator.grid_center_to_screen(2,5)
    target=creature([TakesHits(5)],pos=maze.position_calculator.grid_center_to_screen(8,5))
    w,enc=scene(target)
    p=shot('railgun',start)
    # One update crosses all three adjoining blocks, including newly exposed faces.
    advance(p,maze.cell_size_x*7/(p.definition.base_speed*p.definition.speed),
            maze,[target],enc.apply_hit,Mock())
    assert all(cell not in maze.blocks and maze.grid[cell[1]][cell[0]] == 0 for cell in cells)
    assert metal in maze.blocks and target.hit_points == 5
    assert not p.active
    assert maze.wall_generation == len(cells)


def test_rail_hits_enemy_after_destroying_wall_without_slowing():
    maze=rock_maze([(3,5)])
    start=maze.position_calculator.grid_center_to_screen(2,5)
    target=creature([TakesHits(5)],pos=maze.position_calculator.grid_center_to_screen(4,5))
    _,enc=scene(target)
    p=shot('railgun',start)
    before=p.vx
    dt=maze.cell_size_x*3/p.vx
    advance(p,dt,maze,[target],enc.apply_hit,Mock())
    assert (3,5) not in maze.blocks and target.hit_points == 3
    assert p.active and p.vx == before
    assert p.x == pytest.approx(start[0]+before*dt)


def test_grenade_explosion_damages_rock_and_preserves_indestructible_walls():
    from game_handlers.collision_handler import CollisionHandler
    maze=rock_maze([(3,5)],[(6,5)])
    handler=CollisionHandler(Mock(),Mock(),Mock())
    for cell in ((3,5),(6,5),(0,5)):
        wall=next(w for w in maze.walls if w.cell == cell)
        center=((wall.start[0]+wall.end[0])/2,(wall.start[1]+wall.end[1])/2)
        p=shot('grenade_launcher',center)
        before=maze.blocks[cell]
        handler.explosions.trigger(p.detonate())
        for blast in handler.explosions.update(1):
            handler.handle_blast(blast,None,None,[],Mock(),maze)
        assert cell in maze.blocks
        if cell == (3,5):
            assert maze.blocks[cell] == before-1
        else:
            assert maze.blocks[cell] >= 1
            assert maze.grid[cell[1]][cell[0]] == 1


def test_railgun_can_fire_again_after_half_the_previous_interval():
    state=EquippedWeapon()
    state.equip('railgun')
    assert state.try_fire((400,300),0).projectiles
    state.update(25)
    assert not state.try_fire((400,300),0).projectiles
    state.update(.5)
    assert state.try_fire((400,300),0).projectiles


def test_scattergun_pellet_hits_harder_than_basic_bolt():
    basic_target=creature([TakesHits(10)],pos=(430,300))
    scatter_target=creature([TakesHits(10)],pos=(430,300))
    for weapon,target in (('basic',basic_target),('scattergun',scatter_target)):
        w,enc=scene(target)
        p=shot(weapon)
        advance(p,4,w.maze,[target],enc.apply_hit,Mock())
    assert basic_target.hit_points == 9
    assert scatter_target.hit_points == 8


def test_grenade_triggers_on_near_miss_before_physical_contact():
    target=creature([TakesHits(10)],pos=(450,330))
    w,enc=scene(target)
    p=shot('grenade_launcher')
    blasts=[]
    advance(p,20,w.maze,[target],enc.apply_hit,blasts.append)
    assert len(blasts) == 1 and p.detonated
    assert p.lifetime > 0  # proximity, not the fuse
    distance=((p.x-target.x)**2+(p.y-target.y)**2)**.5
    assert target.radius+p.radius < distance <= target.radius+p.definition.proximity_radius+1e-6
    advance(p,20,w.maze,[target],enc.apply_hit,blasts.append)
    assert len(blasts) == 1


def test_grenade_proximity_does_not_trigger_through_wall_or_for_distant_enemy():
    for pos,walls in (((435,300),[WallSegment((420,250),(420,350),3)]),((400,380),[])):
        target=creature([TakesHits(10)],pos=pos)
        w,enc=scene(target,walls=walls)
        from utils.spatial_grid import SpatialGrid
        w.maze.spatial_grid=SpatialGrid(1000,1000,10)
        w.maze.spatial_grid.add_walls(walls)
        p=shot('grenade_launcher')
        p.vx=p.vy=0
        blasts=[]
        advance(p,1,w.maze,[target],enc.apply_hit,blasts.append)
        assert p.active and not blasts
