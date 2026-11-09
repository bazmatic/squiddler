"""Flocker mechanics, local coordination, real encounters and exact continuous replay."""
from collections import Counter
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pygame
import pytest

import creatures
import enemy_kinds
from creatures.flocker_team import Intent, get_flocker_team, muzzle, visible
from learning.adapters import flocker as adapter
from learning.adapters import flocker_scenario as scenario
from learning.policy import random_policy
from tests.creature_parts import encounter, open_maze, target, world


def wall(start=(200.,100.),end=(200.,700.)):
    from maze.wall_segment import WallSegment
    return WallSegment(start,end,hit_points=10)


def scene(positions=None, ship=None, walls=()):
    if positions is None:
        positions=((200.+enemy_kinds.kind('flocker').tuning['size']+3.,300.),)
    birds = [creatures.build('flocker',p,angle=0.) for p in positions]
    enc = encounter(*birds,world_=world(ship=ship,maze=open_maze(walls)))
    enc.ready()
    team = get_flocker_team(enc)
    team.external = True
    team.members()
    return birds,enc,team


@pytest.mark.parametrize('throttle',[0.,.25,.5,1.,False,True,2.,-1.])
def test_throttle_is_proportional_and_clamped(throttle):
    (bird,),enc,team = scene(positions=((400.,300.),))
    team.commit({0:Intent(throttle=throttle)})
    enc.step(1.)
    (full,),full_enc,full_team = scene(positions=((400.,300.),))
    full_team.commit({0:Intent(throttle=1.)})
    full_enc.step(1.)
    assert bird.vx == pytest.approx(full.vx*max(0,min(1,throttle)))
    assert bird.vy == pytest.approx(0.)


def test_head_aims_at_bounded_speed_and_bullets_leave_actual_mouth():
    (bird,),enc,team = scene(positions=((400.,300.),),ship=target((600.,300.)))
    team.commit({0:Intent(head=1.)})
    enc.step(1.)
    assert bird.flocker_head == 3.
    for _ in range(40):
        enc.step(1.)
    assert bird.flocker_head == 90.
    bird.gun_cooldown = 0
    team.commit({0:Intent(fire=True)})
    expected,heading = muzzle(bird)
    enc.step(1.)
    shot, = enc.world.projectiles
    assert shot.get_pos() == pytest.approx(expected)
    assert shot.angle == pytest.approx(heading)
    assert shot.vx == pytest.approx(0,abs=1e-8)
    assert shot.vy > 0


def test_body_rotation_carries_relative_head_heading():
    (bird,),enc,team = scene(positions=((400.,300.),))
    bird.flocker_head = 30.
    team.commit({0:Intent(turn=1.)})
    enc.step(1.)
    assert bird.flocker_head == 30.
    assert muzzle(bird)[1] == pytest.approx(bird.angle+30.)


def test_sight_respects_range_walls_and_gates():
    (bird,),enc,team = scene(ship=target((150.,300.)),walls=(wall(),))
    assert visible(enc,bird) == []
    enc.maze.walls.clear()
    assert len(visible(enc,bird)) == 1
    enc.maze.gate_segments = [wall()]
    assert visible(enc,bird) == []
    enc.world.ship = target((1000.,300.))
    enc.maze.gate_segments = []
    assert visible(enc,bird) == []


def test_gentle_perch_folds_and_holds_then_releases_on_sight():
    (bird,),enc,team = scene(walls=(wall(),))
    bird.flocker_quiet = 60
    team.commit({0:Intent(perch=1)})
    enc.step(1.)
    assert bird.flocker_perched
    assert bird.get_pos() == pytest.approx((200.+bird.radius+.5,300.))
    origin = bird.get_pos()
    for _ in range(15):
        enc.step(1.)
    assert bird.get_pos() == origin
    assert bird.body.perch_blend == 1.
    assert bird.body.flutter_remaining == 0.
    enc.world.ship = target((400.,300.))
    for _ in range(6):
        enc.step(1.)
    assert not bird.flocker_perched
    assert bird.flocker_wake_latency <= 6


def test_fast_wall_approach_bounces_and_flutters_without_hit_flash():
    (bird,),enc,team = scene(walls=(wall(),))
    bird.vx = -3.
    bird.flocker_quiet = 60
    team.commit({0:Intent(perch=1)})
    for _ in range(4):
        enc.step(1.)
    assert not bird.flocker_perched
    assert bird.body.flutter_remaining>0
    assert bird.body.flinch_remaining==0
    assert bird.vx>=0


def test_crowded_perch_and_moving_gate_cannot_attach():
    birds,enc,team = scene(positions=((206.,300.),(210.,300.)),walls=(wall(),))
    birds[0].flocker_quiet = 60
    assert not team.attach(birds[0])
    enc.maze.walls.clear()
    enc.maze.gate_segments = [wall()]
    assert not team.attach(birds[0])


def test_local_messages_arrive_next_decision_and_mute_removes_them():
    birds,enc,team = scene(positions=((400.,300.),(460.,300.)))
    team.before_frame()
    team.commit({0:Intent(message=5),1:Intent()})
    assert team.received(birds[1])=={}
    for _ in range(6):
        team.before_frame()
    assert team.received(birds[1])=={0:5}
    team.muted = True
    assert team.received(birds[1])=={}


def test_local_alert_wakes_occluded_teammate_without_exposing_target():
    # Horizontal wall occludes the perched bird but not its teammate on the target side.
    w=wall((200.,350.),(500.,350.))
    birds,enc,team=scene(positions=((215.,300.),(215.,400.)),walls=(w,))
    sleeper,scout=birds
    sleeper.flocker_quiet=60
    sleeper.y=350.-sleeper.radius-.5
    assert team.attach(sleeper)
    enc.world.ship=target((350.,450.))
    assert visible(enc,sleeper)==[] and visible(enc,scout)
    for _ in range(13):
        enc.step(1.)
    assert not sleeper.flocker_perched
    assert all(t is None for t in team.targets(sleeper))


def test_perched_actions_and_schema_preserve_continuous_throttle():
    (bird,),enc,team = scene(walls=(wall(),))
    bird.flocker_quiet=60
    assert team.attach(bird)
    x,masks,_=adapter.observe(team,bird)
    assert len(x)==len(adapter.FEATURES)
    assert not masks['throttle'].any() and masks['head'].all()
    intent=adapter.intention((.2,.35,-.4,0,0,0,2),perched=True)
    assert intent.throttle==0 and intent.perch==2
    assert adapter.intention((0,.35,0,0,0,0,0)).throttle==pytest.approx((math.tanh(.35)+1)/2)


def test_volley_requires_distinct_shooters_same_target_and_accepted_hit():
    tracker=scenario.VolleyTracker()
    target_a,target_b=object(),object()
    a,b,c=SimpleNamespace(),SimpleNamespace(),SimpleNamespace()
    tracker.launch(a,target_a,0,0,{0,1,2})
    tracker.launch(b,target_a,1,6,{0,1,2})
    tracker.launch(c,target_b,2,6,{2})
    rewards=Counter()
    tracker.settle(7,rewards,.5)
    assert sum(rewards.values())==0
    tracker.hit(a,target_b)
    tracker.settle(8,rewards,.5)
    assert sum(rewards.values())==0
    tracker.hit(b,target_a)
    tracker.settle(9,rewards,.5)
    assert sum(rewards.values())==pytest.approx(.5*2/3)
    tracker.settle(10,rewards,.5)
    assert sum(rewards.values())==pytest.approx(.5*2/3)
    assert tracker.metrics()['useful_volleys']==1


def test_volley_window_cannot_chain_and_one_shooter_is_not_a_volley():
    tracker=scenario.VolleyTracker()
    t=object()
    for i,frame in enumerate((0,6,12)):
        shot=SimpleNamespace()
        tracker.launch(shot,t,i,frame,{0,1,2})
    assert len(tracker.groups)==2
    assert tracker.metrics()['volleys']==1
    assert tracker.metrics()['volley_spreads']==[6]


@pytest.mark.parametrize('stage',[1,2,3,4,5,6])
def test_real_encounter_collects_each_curriculum_stage(stage):
    from learning.training import collect
    env=scenario.environment(horizon=.2)
    try:
        rows,events,actions=collect(env,random_policy(adapter.SPEC,0),4,dict(level=13,team_size=2,stage=stage))
        assert events['seconds']==.2 and len(rows)==2
        assert np.asarray(rows[0]['actions']).dtype.kind=='f'
        assert all(len(e['critic'][0])==adapter.CRITIC_SIZE for e in rows)
    finally:
        env.close()


def test_reinforcements_start_fresh_trajectories_and_do_not_reuse_ids():
    from learning.training import collect
    env=scenario.environment(horizon=2.2)
    try:
        rows,events,actions=collect(env,random_policy(adapter.SPEC,0),4,
                                   dict(level=13,team_size=2,stage=3,reinforcements=1))
        assert len(rows)==3 and events['team_size']==3
        assert all(rows[2]['alive'])
        assert len(rows[2]['x'])==len(rows[0]['x'])-20
        assert len(rows[2]['rewards'])==len(rows[2]['x'])
    finally:
        env.close()


@pytest.mark.parametrize('radius',[9.6,12.])
def test_exact_replay_roundtrips_fractional_actions(tmp_path,radius):
    from learning.training import collect
    from learning.replay import fly
    from shutil import copyfile
    opponent=Path('assets/neural/hunter.npz')
    copyfile(opponent,tmp_path/'opponent.npz')
    manifest=scenario.metadata(SimpleNamespace(horizon=.3,opponent=tmp_path/'opponent.npz'))
    manifest['creature']['tuning']['size'] = radius
    manifest['training'] = {}
    (tmp_path/'config.json').write_text(json.dumps(dict(task='flocker_team',manifest=manifest)))
    env=scenario.environment(horizon=.3,opponent=tmp_path/'opponent.npz',flocker_radius=radius)
    case=dict(level=13,team_size=2,stage=4,seed=7)
    try:
        _,events,actions=collect(env,random_policy(adapter.SPEC,0),7,{k:v for k,v in case.items() if k!='seed'})
    finally:
        env.close()
    (tmp_path/'episodes').mkdir()
    (tmp_path/'episodes'/'iter0000.json').write_text(json.dumps(dict(jobs=[case],episodes=[events],actions=[actions])))
    rows=fly(tmp_path,0)
    assert rows[0]['seconds']==.3


def test_checkpoint_rejects_another_task(tmp_path):
    from learning.adapters.jelly import SPEC
    path=tmp_path/'jelly.npz'
    random_policy(SPEC,0).save(path)
    with pytest.raises(ValueError):
        adapter.load_policy(path)


def test_curriculum_requires_two_evaluated_passing_blocks():
    state=dict(stage=1,passing_blocks=0)
    state=scenario.advance_curriculum(state,{'flocker_validation_passed':True})
    assert state['stage']==1 and state['passing_blocks']==1
    state=scenario.advance_curriculum(state,{})
    assert state['passing_blocks']==1
    assert scenario.advance_curriculum(state,{'flocker_validation_passed':True})['stage']==2
    assert scenario.advance_curriculum(state,{'flocker_validation_passed':False})['passing_blocks']==0


def test_real_policy_batch_can_update_with_fractional_controls():
    torch=pytest.importorskip('torch')
    from learning.training import collect,build_network
    from learning.ppo import update_team
    env=scenario.environment(horizon=.3)
    try:
        p=random_policy(adapter.SPEC,0)
        episodes,_,_=collect(env,p,4,dict(level=13,team_size=2,stage=3))
        net=build_network(p,adapter.CRITIC_SIZE)
        stats=update_team(net,torch.optim.Adam(net.parameters(),lr=.0001),episodes,np.random.default_rng(0),epochs=1)
        assert all(math.isfinite(float(v)) for v in stats.values())
    finally:
        env.close()


def test_game_driver_and_training_decisions_produce_identical_flights():
    from learning.training import collect
    p=random_policy(adapter.SPEC,2)
    env=scenario.environment(horizon=1.2)
    try:
        _,expected,_=collect(env,p,8,dict(level=13,team_size=3,stage=3),greedy=True)
        env.reset(8,level=13,team_size=3,stage=3)
        env.team.external=False
        env.team.driver=adapter.FlockerDriver(p)
        while True:
            result=env.step({})
            if result.terminated or result.truncated:
                break
        assert result.events==expected
    finally:
        env.close()


def test_stale_alerts_expire_without_relay():
    birds,enc,team=scene(positions=((400.,300.),(460.,300.)))
    team.alerts={0:(0,birds[0].get_pos())}
    team.frames=60
    assert team.alerted(birds[1])
    team.frames=61
    assert not team.alerted(birds[1])


def test_perch_pose_and_head_render_on_all_wall_orientations():
    (bird,),enc,team=scene(positions=((200.,200.),))
    bird.body.perch_blend=1.
    surface=pygame.Surface((400,400),pygame.SRCALPHA)
    for angle in (0,45,90,180,270):
        for head in (-90,0,90):
            bird.angle,bird.flocker_head=angle,head
            bird.draw(surface)
    assert surface.get_bounding_rect().width>0


def test_dead_agents_drop_signals_and_reinforcements_get_new_ids():
    birds,enc,team=scene(positions=((400.,300.),(460.,300.)))
    team.messages={0:(3,birds[0].get_pos())}
    team.alerts={0:(0,birds[0].get_pos())}
    enc.kill(birds[0])
    assert 0 not in team.messages and 0 not in team.alerts
    c=creatures.build('flocker',(600.,300.))
    enc.roster.add(c)
    enc.ready()
    assert [o.agent_id for o in team.members()]==[1,2]


def test_promotion_refuses_unpassed_or_changed_weights(tmp_path):
    from tools.promote_flocker import promote
    (tmp_path/'evaluation.json').write_text(json.dumps(dict(deployment=dict(passed=False))))
    with pytest.raises(ValueError,match='deployment gate'):
        promote(tmp_path,tmp_path/'published.npz')
    random_policy(adapter.SPEC,0).save(tmp_path/'best.npz')
    (tmp_path/'evaluation.json').write_text(json.dumps(dict(deployment=dict(passed=True),checkpoint_sha256='wrong')))
    with pytest.raises(ValueError,match='evaluated checkpoint'):
        promote(tmp_path,tmp_path/'published.npz')
    assert not (tmp_path/'published.npz').exists()


def test_real_projectiles_credit_one_useful_volley_and_actual_damage():
    env=scenario.environment(horizon=1.)
    try:
        env.reset(4,level=13,team_size=2,stage=4,moving=False)
        # Isolate emission and hit attribution from cave geometry in this encounter.
        env.world.maze=open_maze()
        env.team.maze=env.world.maze
        env.targets[0].x,env.targets[0].y=600.,300.
        env.targets[0].vx=env.targets[0].vy=0.
        for c,y in zip(env.team.members(),(280.,320.)):
            c.x,c.y=400.,y
            c.prev_x,c.prev_y=c.get_pos()
            c.angle=0.
            c.flocker_head=math.degrees(math.atan2(300-y,200))
            c.gun_cooldown=0.
        actions={i:(0.,-20.,0.,1,1,0,0) for i in (0,1)}
        rewards=0.
        for _ in range(6):
            result=env.step(actions)
            rewards+=result.reward
        assert result.events['shots']==2 and result.events['hits']==2
        assert env.targets[0].health==6
        assert result.events['useful_volleys']==1
        assert result.events['participation_sum']==1.
        assert sum(env.agent_hits.values())==2
        assert rewards>2*env.rewards['hit_reward']-2*env.rewards['shot_cost']
    finally:
        env.close()


def test_a_new_flocker_can_die_before_its_first_team_decision():
    bird=creatures.build('flocker',(400.,300.))
    enc=encounter(bird)
    enc.ready()
    assert not hasattr(bird,'agent_id')
    enc.kill(bird)
    assert not bird.active


def test_cli_defaults_flocker_bootstrap_to_zero():
    from tools.ppo_neural import build_parser
    parser=build_parser()
    assert parser.parse_args(['--task','flocker_team']).bootstrap_episodes==0
    assert parser.parse_args(['--task','hunter']).bootstrap_episodes==8


@pytest.mark.parametrize('stage',range(2,7))
def test_curriculum_retention_cycles_through_all_earlier_stages(stage):
    cases=scenario.training_cases(0,96,stage)
    assert sum(c['stage']==stage for c in cases)==72
    assert {c['stage'] for c in cases if c['stage']<stage}==set(range(1,stage))


def test_volley_promotion_requires_a_volley_that_actually_hits(monkeypatch,tmp_path):
    monkeypatch.setattr(scenario,'_baselines',{})
    monkeypatch.setattr(scenario,'scripted_report',lambda *a,**kw:{'summary':{}})
    args=SimpleNamespace(horizon=60.,opponent='unused',out=str(tmp_path))
    summary=dict(volleys=5,participation=1.,median_launch_spread=1.,useful_volleys_per_minute=0.)
    assert not scenario.validation_evidence(summary,dict(stage=4),args)['flocker_validation_passed']
    summary['useful_volleys_per_minute']=1.
    assert scenario.validation_evidence(summary,dict(stage=4),args)['flocker_validation_passed']


def test_deployment_compares_useful_volley_rates_when_trials_end_early(monkeypatch,tmp_path):
    trained={'episodes':[dict(seconds=20.,hits=4,useful_volleys=2,survival_fraction=1.,cohesion=.8)]}
    baseline={'episodes':[dict(seconds=60.,hits=4,useful_volleys=2,survival_fraction=1.,cohesion=.8)]}
    monkeypatch.setattr(scenario,'scripted_report',lambda *a,**kw:baseline)
    args=SimpleNamespace(horizon=60.,opponent='unused',out=str(tmp_path))
    report=scenario.deployment_report(dict(trained=trained,scripted=baseline),args)
    assert report['passed']
    assert report['comparisons']['bounded']['useful_volleys_per_minute']['lower_95']>0


def test_pursuit_alone_cannot_pass_the_flocking_gate(monkeypatch,tmp_path):
    monkeypatch.setattr(scenario,'_baselines',{})
    baseline=dict(approaches_per_minute=1.,cohesion=.8,bunched_fraction=.02)
    monkeypatch.setattr(scenario,'scripted_report',lambda *a,**kw:{'summary':baseline})
    args=SimpleNamespace(horizon=60.,opponent='unused',out=str(tmp_path))
    summary=dict(approaches_per_minute=2.,cohesion=.3,bunched_fraction=.01)
    assert not scenario.validation_evidence(summary,dict(stage=1),args)['flocker_validation_passed']
    summary['cohesion']=.8
    assert scenario.validation_evidence(summary,dict(stage=1),args)['flocker_validation_passed']


def test_regroup_drills_remain_target_free_and_are_retained():
    for stage in range(1,7):
        cases=scenario.training_cases(0,96,stage)
        pursuit=[c for c in cases if c['stage']==1]
        assert any(c['regroup'] for c in pursuit)
        assert any(not c['regroup'] for c in pursuit)
    env=scenario.environment(horizon=.2)
    try:
        env.reset(4,level=13,team_size=2,stage=1,regroup=True)
        assert not any(t.active for t in env.targets)
        result=env.step({i:(0.,0.,0.,0,0,0,0) for i in (0,1)})
        assert not any(t.active for t in env.targets)
        assert result.events.get('approaches',0)==0
    finally:
        env.close()


def test_first_stage_checkpoint_selection_prefers_a_cohesive_flock_to_lone_pursuit():
    scattered=dict(approaches_per_minute=20.,cohesion=.2)
    flocking=dict(approaches_per_minute=10.,cohesion=.8)
    assert scenario.curriculum_selection_score(flocking,dict(stage=1))>scenario.curriculum_selection_score(scattered,dict(stage=1))


@pytest.mark.parametrize('level',scenario.TRAIN_LEVELS)
def test_flocking_drills_spawn_a_compact_nonoverlapping_group(level):
    env=scenario.environment(horizon=.1)
    try:
        env.reset(431,level=level,team_size=8,stage=1,compact_group=True)
        birds=env.team.members()
        distances=[math.dist(a.get_pos(),b.get_pos()) for i,a in enumerate(birds) for b in birds[i+1:]]
        assert len(birds)==8
        assert min(distances)>12.
        assert max(distances)<=120.
        assert all(env.world.maze.is_open(c.get_pos(),c.radius+3) for c in birds)
    finally:
        env.close()


def test_target_reports_are_delayed_local_snapshots_and_do_not_grant_sight():
    birds,enc,team=scene(positions=((215.,300.),(215.,400.)),walls=(wall((200.,350.),(500.,350.)),))
    listener,scout=birds
    enc.world.ship=target((350.,450.))
    team.prepare_decision()
    assert team.target_report(listener) is None
    team.frames=6
    team.prepare_decision()
    report=team.target_report(listener)
    assert report['position']==(350.,450.)
    assert report['age_frames']==6
    assert team.targets(listener)==[None]*4
    x,masks,_=adapter.observe(team,listener)
    assert x[adapter.FEATURES.index('report_present')]==1.
    assert not masks['fire'][1]
    enc.world.ship.x=450.
    assert team.target_report(listener)['position']==(350.,450.)
    listener.x=1000.
    assert team.target_report(listener) is None
    listener.x=215.
    team.muted=True
    assert team.target_report(listener) is None


def test_target_reports_expire_and_are_not_rebroadcast_by_listeners():
    birds,enc,team=scene(positions=((215.,300.),(215.,400.)),walls=(wall((200.,350.),(500.,350.)),))
    listener,scout=birds
    enc.world.ship=target((350.,450.))
    team.prepare_decision();team.frames=6;team.prepare_decision()
    assert team.target_report(listener)
    enc.world.ship.active=False
    team.frames=12;team.prepare_decision()
    assert listener.agent_id not in team._sightings
    team.frames=78;team.prepare_decision()
    assert team.target_report(listener) is None


def test_old_flocker_weights_transfer_without_changing_existing_behaviour(tmp_path):
    from dataclasses import replace
    old_spec=replace(adapter.SPEC,features=adapter.FEATURES[:-len(adapter.REPORT)],version=2)
    old=random_policy(old_spec,seed=7)
    path=tmp_path/'old.npz';old.save(path)
    new=adapter.load_transfer_policy(path)
    x=np.random.default_rng(4).normal(size=len(old_spec.features)).astype(np.float32)
    before,state_before=old.step(x,old.initial_state())
    after,state_after=new.step(np.concatenate((x,np.ones(len(adapter.REPORT),np.float32))),new.initial_state())
    np.testing.assert_allclose(state_before,state_after,atol=1e-6)
    for head in before:
        np.testing.assert_allclose(before[head],after[head],atol=1e-6)
    assert new.spec==adapter.SPEC


@pytest.mark.parametrize('stage,strength',[(5,.5),(6,1.)])
def test_combat_opponent_is_the_real_neural_hunter_with_only_bullet_strength_changed(stage,strength):
    import config
    from hunter.settings_for import hunter_settings_for
    env=scenario.environment(horizon=.2)
    try:
        env.reset(900,level=15,team_size=4,stage=stage)
        ship=env.ships[0];settings=hunter_settings_for('neural')
        traits={t.NAME:t for t in ship.traits}
        assert ship.settings==settings
        assert ship.flight.controller.kind=='neural'
        assert ship.flight.controller.decision_frames==config.SETTINGS.game.neuralDecisionFrames
        assert ship.max_speed==config.SETTINGS.hunterShip.maxSpeed
        assert ship.thrust_force==config.SETTINGS.hunterShip.thrustForce*settings.thrust_multiplier
        assert ship.current_rotation_speed==config.SETTINGS.hunterShip.rotationSpeed*settings.turn_rate_multiplier
        assert ship.health==settings.health
        assert ship.indestructible==settings.indestructible
        assert traits['burstFire'].rounds==settings.burst_size
        assert ship.fire_interval==settings.fire_interval
        assert traits['gun'].shot_strength==strength
    finally:
        env.close()


def test_weaker_training_bullets_really_do_half_damage_to_flockers():
    from entities.projectile import Projectile
    for stage,remaining in ((5,.5),(6,0.)):
        env=scenario.environment(horizon=.2)
        try:
            env.reset(900,level=15,team_size=2,stage=stage)
            env.world.maze=open_maze()
            bird=env.team.members()[0];bird.x,bird.y=400.,300.
            shot=Projectile(bird.get_pos(),0.,source='hunter')
            shot.vx=shot.vy=0.
            env.world.projectiles=[shot]
            env.fly_shots(env.event_rows())
            assert bird.hit_points==remaining
            assert bird.active==(stage==5)
        finally:
            env.close()


def test_aim_accuracy_counts_both_landed_shots_when_immunity_blocks_second_damage():
    from entities.projectile import Projectile
    env=scenario.environment(horizon=.2,rewards={'hit_reward':2.})
    try:
        env.reset(900,level=15,team_size=2,stage=5)
        env.world.maze=open_maze()
        ship=env.ships[0];ship.x,ship.y=600.,300.;ship.vx=ship.vy=0.
        shots=[]
        for bird in env.team.members():
            shot=Projectile(ship.get_pos(),0.,is_enemy=True,source='enemy')
            shot.vx=shot.vy=0.;shot.actor_id=bird.agent_id;shots.append(shot)
        env.world.projectiles=shots
        env.counts['shots']=2
        rows=env.event_rows();health=ship.health
        env.fly_shots(rows)
        rewards=env.paid(rows)
        events=env.events();summary=scenario.metrics([events])
        assert all(not shot.active for shot in shots)
        assert health-ship.health==1
        assert events['hits']==1 and events['landed_hits']==2
        assert events['non_damaging_hits']==1
        assert summary['accuracy']==.5
        assert summary['aim_accuracy']==1.
        assert sum(rewards.values())==2.
    finally:
        env.close()


def test_aiming_drills_include_hunter_speed_targets_and_visible_starts():
    cases=[c for c in scenario.training_cases(0,96,3) if c['stage']==3]
    assert {c['target_speed'] for c in cases}>={0.,1.,2.,3.,4.}
    env=scenario.environment(horizon=.2)
    try:
        env.reset(431,level=13,team_size=8,stage=3,moving=True,target_speed=4.,visible_start=True)
        target=env.targets[0]
        assert math.hypot(target.vx,target.vy)==pytest.approx(4.)
        assert any(target in visible(env.enc,c) for c in env.team.members())
    finally:
        env.close()


def test_aiming_refresher_cannot_graduate_at_twenty_percent(monkeypatch,tmp_path):
    monkeypatch.setattr(scenario,'_baselines',{})
    baseline=dict(accuracy=.2,hits_per_minute=10.)
    monkeypatch.setattr(scenario,'scripted_report',lambda *a,**kw:{'summary':baseline})
    args=SimpleNamespace(horizon=60.,opponent='unused',out=str(tmp_path))
    summary=dict(hits=5,accuracy=.25,hits_per_minute=12.)
    state=dict(stage=3,minimum_aim_accuracy=.4)
    assert not scenario.validation_evidence(summary,state,args)['flocker_validation_passed']
    summary['accuracy']=.5
    assert scenario.validation_evidence(summary,state,args)['flocker_validation_passed']


def aim_inputs(x,slot=0):
    return {name:float(x[adapter.FEATURES.index(f'target{slot}_{name}')])
            for name in ('aim_sin','aim_cos','lead_present','lead_sin','lead_cos','body_vx','body_vy')}


def test_aim_error_is_shown_directly_and_changes_sign_with_the_head():
    (bird,),enc,team=scene(positions=((400.,300.),),ship=target((600.,300.)))
    aimed=aim_inputs(adapter.observe(team,bird)[0])
    assert aimed['aim_sin']==pytest.approx(0.,abs=.01) and aimed['aim_cos']==pytest.approx(1.,abs=.001)
    bird.flocker_head=10.
    left=aim_inputs(adapter.observe(team,bird)[0])['aim_sin']
    bird.flocker_head=-10.
    right=aim_inputs(adapter.observe(team,bird)[0])['aim_sin']
    assert left<-.15 and right>.15 and left==pytest.approx(-right,abs=.02)


def test_lead_error_points_ahead_of_a_crossing_target_and_agrees_when_it_stands_still():
    ship=target((600.,300.))
    (bird,),enc,team=scene(positions=((400.,300.),),ship=ship)
    still=aim_inputs(adapter.observe(team,bird)[0])
    assert still['lead_present']==1. and still['lead_sin']==pytest.approx(still['aim_sin'],abs=.01)
    ship.vx,ship.vy=0.,4.
    moving=aim_inputs(adapter.observe(team,bird)[0])
    assert moving['lead_present']==1. and moving['lead_sin']>.1       # the shot must go towards +y, the target's side
    assert moving['aim_sin']==pytest.approx(still['aim_sin'],abs=.01)


def test_lead_is_absent_beyond_gun_range_but_aim_error_is_still_shown():
    (bird,),enc,team=scene(positions=((400.,300.),),ship=target((900.,300.)))
    seen=aim_inputs(adapter.observe(team,bird)[0])
    assert seen['lead_present']==0. and seen['aim_cos']>.99


def test_target_velocity_is_also_given_in_the_birds_own_frame():
    ship=target((400.,500.))
    ship.vx,ship.vy=8.,0.
    (bird,),enc,team=scene(positions=((400.,300.),),ship=ship)
    bird.angle=90.
    seen=aim_inputs(adapter.observe(team,bird)[0])
    assert (seen['body_vx'],seen['body_vy'])==pytest.approx((0.,-1.),abs=1e-6)


def test_checkpoints_without_aim_inputs_keep_their_behaviour_when_they_are_added(tmp_path):
    from dataclasses import replace
    new_names=('aim_sin','aim_cos','lead_present','lead_sin','lead_cos','body_vx','body_vy')
    keep=[i for i,n in enumerate(adapter.FEATURES) if not any(n.endswith('_'+s) and n.startswith('target') for s in new_names)]
    old=random_policy(replace(adapter.SPEC,features=tuple(adapter.FEATURES[i] for i in keep),version=3),seed=5)
    path=tmp_path/'old.npz';old.save(path)
    new=adapter.load_transfer_policy(path)
    x=np.random.default_rng(1).normal(size=len(adapter.FEATURES)).astype(np.float32)
    before,_=old.step(x[keep],old.initial_state())
    after,_=new.step(x,new.initial_state())
    for head in before:
        np.testing.assert_allclose(before[head],after[head],atol=1e-6)


def land_shots(env,count):
    from entities.projectile import Projectile
    bird=env.team.members()[0]
    rows=env.event_rows()
    for _ in range(count):
        t=env.targets[0]
        shot=Projectile(t.get_pos(),0.,is_enemy=True,source='enemy')
        shot.actor_id=bird.agent_id
        env.world.projectiles.append(shot)
        env.fly_shots(rows)
    return env.paid(rows)


def test_aiming_drill_targets_take_any_number_of_hits_and_stay_where_they_are():
    env=scenario.environment(horizon=1.)
    try:
        env.reset(3,level=13,team_size=2,stage=3,target_speed=0.,visible_start=True)
        start=env.targets[0].get_pos()
        rewards=land_shots(env,20)
        assert env.counts['hits']==20 and env.counts.get('kills',0)==0
        assert env.targets[0].active and env.targets[0].get_pos()==start
        assert sum(rewards.values())==pytest.approx(20*env.rewards['hit_reward'])
    finally:
        env.close()


def test_other_stages_still_kill_and_replace_a_target_after_eight_hits():
    env=scenario.environment(horizon=1.)
    try:
        env.reset(3,level=13,team_size=2,stage=4)
        land_shots(env,8)
        assert env.counts['kills']==1 and env.targets[0].generation==1
    finally:
        env.close()


def incoming_fire(bird,dx,vx):
    from entities.projectile import Projectile
    shot=Projectile((bird.x+dx,bird.y),0.,source='hunter')
    shot.vx,shot.vy=vx,0.
    return shot


def threat_inputs(x,slot=0):
    return {name:float(x[adapter.FEATURES.index(f'threat{slot}_{name}')]) for name in adapter.THREAT}


def test_a_bird_sees_fire_coming_at_it_but_not_fire_moving_away():
    (bird,),enc,team=scene(positions=((400.,300.),),ship=target((600.,300.)))
    bird.angle=0.
    enc.world.projectiles=[incoming_fire(bird,150.,-8.)]
    coming=threat_inputs(adapter.observe(team,bird)[0])
    assert coming['present']==1. and coming['miss']<.1 and 0<coming['time']<1 and coming['vx']<0
    enc.world.projectiles=[incoming_fire(bird,150.,8.)]
    assert threat_inputs(adapter.observe(team,bird)[0])['present']==0.
    enc.world.projectiles=[incoming_fire(bird,150.,-8.)]
    enc.world.projectiles[0].y+=200.
    assert threat_inputs(adapter.observe(team,bird)[0])['miss']==1.


def test_being_in_the_line_of_fire_and_being_hit_cost_only_the_bird_concerned():
    rewards={k:0. for k in scenario.flocker_rewards.DEFAULTS}
    rewards.update(threat_penalty=.05,damage_penalty=2.)
    env=scenario.environment(horizon=1.,rewards=rewards)
    try:
        env.reset(900,level=15,team_size=2,stage=3,moving=False)
        env.world.maze=open_maze();env.team.maze=env.world.maze
        a,b=env.team.members()
        a.x,a.y,b.x,b.y=400.,300.,400.,600.
        a.prev_x,a.prev_y,b.prev_x,b.prev_y=a.x,a.y,b.x,b.y
        a.vx=a.vy=b.vx=b.vy=0.
        shot=incoming_fire(a,30.,-8.)
        shot.creature_damage=.5
        env.world.projectiles=[shot]
        result=env.step({i:(0.,-20.,0.,0,0,0,0) for i in (0,1)})
        assert a.hit_points==.5
        assert result.agent_rewards[a.agent_id]==pytest.approx(-.05-2.*.5,abs=1e-6)
        assert result.agent_rewards[b.agent_id]==0.
    finally:
        env.close()


def test_combat_opens_against_a_weaker_hunter_but_held_out_trials_face_the_full_one():
    cases=[c for i in range(36) for c in scenario.training_cases(i,12,5) if c['stage']==5]
    assert {(c['hunter_strength'],c['hunter_health']) for c in cases}==set(scenario.HUNTER_RAMP)
    assert all('hunter_strength' not in c for c in scenario.validation_cases()+scenario.evaluation_cases(5))
    env=scenario.environment(horizon=.2)
    try:
        env.reset(900,level=15,team_size=2,stage=5,hunter_strength=.125,hunter_health=4)
        assert env.ships[0].health==4
        env.world.maze=open_maze()
        bird=env.team.members()[0];bird.x,bird.y=400.,300.
        shot=incoming_fire(bird,0.,0.)
        env.world.projectiles=[shot]
        env.fly_shots(env.event_rows())
        assert bird.hit_points==pytest.approx(.875)
    finally:
        env.close()


def test_combat_gate_can_demand_more_survival_than_the_scripted_team(monkeypatch):
    monkeypatch.setitem(scenario._baselines,(5,60.,'o'),dict(hits_per_minute=1.,survival_fraction=.1))
    summary=dict(hits=5,hits_per_minute=2.,survival_fraction=.3)
    args=SimpleNamespace(horizon=60.,opponent='o')
    state=dict(stage=5,passing_blocks=0)
    assert scenario.validation_evidence(summary,state,args)['flocker_validation_passed']
    assert not scenario.validation_evidence(summary,dict(state,minimum_survival=.5),args)['flocker_validation_passed']


def test_a_flocker_that_has_picked_a_target_can_still_be_copied_for_its_death_portrait():
    import copy
    import threading
    (bird,),enc,team=scene(positions=((400.,300.),),ship=target((600.,300.)))
    ship=target((600.,300.))
    ship.sound=threading.Lock()          # a live ship carries unpicklable audio handles
    bird.flocker_target=ship
    portrait=copy.deepcopy(bird)
    assert portrait.flocker_target is None and portrait.get_pos()==bird.get_pos()
