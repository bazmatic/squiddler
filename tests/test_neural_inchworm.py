"""Inchworm inference, motor semantics, curriculum and replay. No optimizer runs."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from learning.adapters import inchworm as adapter
from learning.adapters import inchworm_scenario as scenario
from learning.inchworm_rewards import DEFAULTS
from learning.policy import random_policy
from physics.inchworm import InchwormCommand


def test_observation_is_finite_body_relative_and_has_versioned_shape():
    env = scenario.environment(horizon=.2)
    try:
        a = env.reset(12,level=0,stage=3).observations[0]
        b = env.reset(12,level=1,stage=3).observations[0]
        assert a.shape == (len(adapter.FEATURES),) == (83,)
        assert np.isfinite(a).all()
        assert b == pytest.approx(a,abs=1e-5)
        target = env.prey
        wall = type(env.world.maze.walls[0])((env.worm.x-50,(env.worm.y+target.y)/2),
                                            (env.worm.x+50,(env.worm.y+target.y)/2),1)
        # Explicit occlusion uses a wall perpendicular to the current line of sight.
        mid = ((env.worm.worm_front[0]+target.x)/2,(env.worm.worm_front[1]+target.y)/2)
        dx,dy = target.x-env.worm.worm_front[0],target.y-env.worm.worm_front[1]
        wall.start,wall.end = (mid[0]-dy,mid[1]+dx),(mid[0]+dy,mid[1]-dx)
        env.world.maze.walls.append(wall)
        x,masks,_ = adapter.observe(env.worm,env.enc)
        assert x[adapter.FEATURES.index('target_present')] == 0
        assert not masks['bite'][1]
    finally:
        env.close()


def test_motor_conversion_clips_through_tanh_and_has_no_twist_output():
    command = adapter.intention((100.,-100.,0.,.2,-.2,1,0,1))
    assert command.muscles[:3] == (1.,-1.,0.)
    assert command.rear_grip and not command.front_grip and command.bite
    assert len(adapter.HEADS) == 8 and all('twist' not in n for n,_ in adapter.HEADS)
    for bad in ((0.,)*7,(float('nan'),)*8,(0.,)*5+(2,0,0)):
        with pytest.raises(ValueError):
            adapter.intention(bad)


def test_policy_round_trip_rejects_wrong_task_and_bad_normalization(tmp_path):
    policy = random_policy(adapter.SPEC,2,hidden=8,embed=4)
    path = tmp_path/'weights.npz'
    policy.save(path)
    loaded = adapter.load_policy(path)
    x = np.zeros(len(adapter.FEATURES),np.float32)
    assert loaded.choose(x,loaded.initial_state())[0] == policy.choose(x,policy.initial_state())[0]
    policy.mean[0] = np.nan
    policy.save(path)
    with pytest.raises(ValueError,match='normalization'):
        adapter.load_policy(path)
    from learning.adapters.flocker import SPEC as FLOCKER_SPEC
    random_policy(FLOCKER_SPEC).save(path)
    with pytest.raises(ValueError,match='schema'):
        adapter.load_policy(path)


def test_game_loads_once_with_independent_memory_and_external_override(tmp_path,monkeypatch):
    import creatures
    from tests.creature_parts import encounter,world,open_maze
    path = tmp_path/'policy.npz'
    random_policy(adapter.SPEC,1,hidden=8,embed=4).save(path)
    monkeypatch.setenv('INCHWORM_POLICY',str(path))
    a,b = creatures.build('inchworm',(400.,200.)),creatures.build('inchworm',(500.,200.))
    enc = encounter(a,b,world_=world(maze=open_maze()))
    enc.ready()
    assert a.worm_driver.policy is b.worm_driver.policy
    assert a.worm_driver.state is not b.worm_driver.state
    a.worm_command = InchwormCommand()
    enc.step(1.)
    assert not a.worm_driver.state.any()  # an external controller retains priority
    assert b.worm_driver.state.any()
    enc.close()
    assert a.worm_driver is None and b.worm_driver is None


def test_missing_selected_policy_logs_and_falls_back(monkeypatch,caplog):
    from tests.test_inchworm_creature import setting
    monkeypatch.setenv('INCHWORM_POLICY','/no/such/inchworm.npz')
    c,enc = setting()
    assert c.worm_driver is None
    enc.step(1.)
    assert c.worm_front_grip
    assert 'using reflex controller' in caplog.text
    enc.close()


@pytest.mark.parametrize('stage',scenario.CURRICULUM)
def test_each_lesson_collects_finite_trajectories_without_learning(stage):
    from learning.training import collect
    env = scenario.environment(horizon=.2)
    policy = random_policy(adapter.SPEC,2,hidden=8,embed=4)
    try:
        trajectories,events,trace = collect(env,policy,12,dict(level=0,stage=stage))
        assert len(trajectories)==1
        assert 1 <= len(trace) <= 2 if stage == 7 else len(trace)==2
        assert len(trajectories[0]['actions'][0])==8
        assert np.isfinite(trajectories[0]['x']).all()
        assert np.isfinite(trajectories[0]['rewards']).all()
        assert trajectories[0]['terminated'] == bool(events.get('curl_anchor_lost',False))
        with pytest.raises(RuntimeError):
            env.step({0:(0.,)*8})
    finally:
        env.close()


def test_curriculum_requires_two_held_out_passes_and_keeps_prior_lessons():
    state = scenario.initial_curriculum(None)
    assert scenario.advance_curriculum(state,dict(success=1.)) == state
    evidence = scenario.validation_evidence(dict(success=1.,survived=1.),state,None)
    state = scenario.advance_curriculum(state,evidence)
    assert state == dict(stage=1,passing_blocks=1)
    for _ in range(scenario.PASSES_REQUIRED[1]-1):
        assert state['stage']==1
        state = scenario.advance_curriculum(state,evidence)
    assert state['stage']==6
    hopping = scenario.validation_evidence(dict(success=1.,survived=1.,airborne_fraction=.7),scenario.initial_curriculum(None),None)
    assert not hopping['inchworm_validation_passed']      # a crawl that is mostly airborne does not pass
    slow = scenario.validation_evidence(dict(success=1.,survived=1.,seconds=scenario.CRAWL_MAX_SECONDS+10),scenario.initial_curriculum(None),None)
    assert not slow['inchworm_validation_passed']         # nor does a slow one
    cases = [c for i in range(4) for c in scenario.training_cases(i,24,5,curriculum=dict(stage=5))]
    assert {c['stage'] for c in cases} == set(scenario.CURRICULUM)   # practice rotates through every earlier lesson
    assert {c['level'] for c in cases}.isdisjoint(c['level'] for c in scenario.validation_cases())
    assert {c['seed'] for c in scenario.evaluation_cases()}.isdisjoint(c['seed'] for c in scenario.validation_cases())
    assert {c["stage"] for c in scenario.training_cases(0,40,1,bootstrap=True)} == set(scenario.DEMONSTRATED)   # Gap has no scripted demonstration


def test_recorded_actions_replay_exactly_and_rule_changes_are_rejected(tmp_path):
    from learning import replay
    from learning.training import collect
    env = scenario.environment(horizon=.3)
    case = dict(seed=18,level=4,stage=5)
    try:
        _,events,trace = collect(env,random_policy(adapter.SPEC,1,hidden=8,embed=4),
                               case['seed'],dict(level=case['level'],stage=case['stage']))
    finally:
        env.close()
    manifest = scenario.metadata(SimpleNamespace(horizon=.3))
    manifest['training'] = {}
    config = dict(task='inchworm',manifest=manifest)
    (tmp_path/'config.json').write_text(json.dumps(config))
    (tmp_path/'episodes').mkdir()
    (tmp_path/'episodes/iter0000.json').write_text(json.dumps(dict(jobs=[case],episodes=[dict(events,**case)],actions=[trace])))
    result, = replay.fly(tmp_path,0)
    assert result['return'] == pytest.approx(events['return_total'])
    config['manifest']['scenario_version'] += 1
    (tmp_path/'config.json').write_text(json.dumps(config))
    with pytest.raises(ValueError,match='rules'):
        replay.open_environment(tmp_path)


def test_preset_and_task_defaults_are_consistent_and_do_not_launch(monkeypatch):
    from tools.ppo_neural import build_parser
    from learning.tasks import task
    preset = json.loads((Path(__file__).resolve().parents[1]/'config/training/inchworm.json').read_text())
    args = build_parser().parse_args(['--task','inchworm'])
    for key,value in task('inchworm').fresh_defaults.items():
        assert getattr(args,key) == value == preset[key]
    overridden = build_parser().parse_args(['--task','inchworm','--episodes','2','--bootstrap-episodes=0'])
    assert overridden.episodes==2 and overridden.bootstrap_episodes==0
    assert task('inchworm').adapter().USES_OPPONENT is False
    assert 'inchworm' in task('inchworm').label.lower()
    from tools.trainer.settings import applies_to
    assert applies_to('inchworm_bite_reward') == ('inchworm',)


def test_deployment_requires_all_lessons_and_matched_cases():
    assert not scenario.deployment_report({'trained':{'episodes':[]},'scripted':{'episodes':[]}},None)['passed']
    a = dict(stage=1,level=1,seed=1,success=1.)
    with pytest.raises(ValueError,match='match'):
        scenario.compare({'episodes':[a]},{'episodes':[dict(a,seed=2)]})


def test_cli_dispatch_keeps_planned_horizon_without_running_training(monkeypatch):
    import learning.run
    from tools.ppo_neural import main
    seen = []
    def record_only(args):
        seen.append(args)
        return dict(paused=False,last='not-started')
    monkeypatch.setattr(learning.run,'train',record_only)
    main(['--task','inchworm'])
    assert seen[-1].horizon == 120.
    main(['--task','inchworm','--horizon=2'])
    assert seen[-1].horizon == 2.


def test_game_inference_and_external_training_actions_share_cadence_through_stun():
    a,b = scenario.environment(horizon=2.),scenario.environment(horizon=2.)
    policy = random_policy(adapter.SPEC,3,hidden=8,embed=4)
    try:
        a.reset(12); b.reset(12)
        a.worm.worm_driver = adapter.InchwormDriver(policy,seed=12)
        rng = np.random.default_rng(np.random.SeedSequence([12,0]))
        state = policy.initial_state()
        for decision in range(12):
            if decision == 2:
                a.worm.worm_stun = b.worm.worm_stun = 10.
            x,masks,_ = adapter.observe(b.worm,b.enc)
            action,state = policy.choose(x,state,masks,greedy=False,rng=rng)
            b.step({0:action})
            for _ in range(adapter.DECISION_FRAMES):
                a.enc.step(1.)
            assert a.worm.get_pos() == pytest.approx(b.worm.get_pos(),abs=1e-7)
            assert a.worm.worm_driver.state == pytest.approx(state,abs=1e-7)
    finally:
        a.close(); b.close()


def test_sustained_crawl_goals_and_short_drills_fit_the_platform():
    import math
    env = scenario.environment()
    distances = []
    try:
        for seed in range(32):
            env.reset(seed, level=0, stage=1)
            distances.append(env.distance())
        assert any(60 <= d <= 90 for d in distances)
        assert any(180 <= d <= 300 for d in distances)
        assert all(60 <= d <= 90 or 180 <= d <= 300 for d in distances)
        for case in scenario.validation_cases()+scenario.evaluation_cases():
            if case['stage'] != 1:
                continue
            env.reset(**case)
            assert 180 <= env.distance() <= 300
            goal = env.world.worm_goal
            along = sum((goal[i]-env.origin[i])*env.forward[i] for i in range(2))
            assert abs(along)+60 < env.arenas[str(case['level'])]['half_length']
        env.reset(12, level=0, stage=2)
        assert env.distance() < 100  # Turning remains its own short drill.
        assert env.horizon == 120
    finally:
        env.close()


def test_turn_targets_stay_within_reach_of_the_first_joint_and_include_near_ones():
    env = scenario.environment(horizon=1.)
    try:
        radii,heights = [],[]
        for seed in range(40):
            env.reset(seed,level=0,stage=2)
            links = env.worm.worm_physics.links
            pivot = [(links[0].position[i]+links[1].position[i])/2 for i in range(2)]
            dx,dy = env.world.worm_goal[0]-pivot[0],env.world.worm_goal[1]-pivot[1]
            behind = -(dx*env.forward[0]+dy*env.forward[1])
            above = dx*env.normal[0]+dy*env.normal[1]
            assert behind > 0 and above >= -1e-6
            assert math.degrees(math.atan2(above,behind)) <= scenario.TURN_MAX_ANGLE+1e-6
            radii.append(math.hypot(dx,dy)); heights.append(above)
        assert scenario.TURN_REACH[0]-1e-6 <= min(radii) and max(radii) <= scenario.TURN_REACH[1]+1e-6
        assert min(radii) < 25 and max(radii) > 45        # near curls and far swings both occur
        assert sum(h < 15 for h in heights) > len(heights)*.5  # mostly low targets
    finally:
        env.close()


def test_about_face_goal_is_behind_the_worm_weighted_near():
    env = scenario.environment(horizon=1.)
    try:
        for seed in range(20):
            env.reset(seed,level=seed%4,stage=8)
            c = env.worm.get_pos(); goal = env.world.worm_goal
            behind = -((goal[0]-c[0])*env.forward[0]+(goal[1]-c[1])*env.forward[1])
            assert 45 <= behind <= 200
        distances = []
        for seed in range(100,160):
            env.reset(seed,stage=8); distances.append(math.dist(env.worm.get_pos(),env.world.worm_goal))
        assert min(distances) >= 90 and max(distances) > 180 and min(distances) < 110
        assert scenario.CURRICULUM.index(8) == scenario.CURRICULUM.index(2)+1
    finally:
        env.close()


@pytest.mark.parametrize('origin,offset,grounded,success', [
    ('head', 13., True, True), ('head', 14., True, False),
    ('head', 0., False, False), ('centre', 0., True, False),
])
def test_about_face_arrival_uses_head_and_requires_a_planted_grip(monkeypatch,origin,offset,grounded,success):
    env = scenario.environment(horizon=.1)
    try:
        env.reset(12,stage=8)
        if grounded:
            env.worm.worm_command = InchwormCommand(rear_grip=True)
            env.enc.step(1.)
            assert env.worm.worm_rear_grip
        pos = env.worm.worm_front if origin == 'head' else env.worm.get_pos()
        env.world.worm_goal = (pos[0]+offset,pos[1])
        monkeypatch.setattr(env.enc,'step',lambda dt: None)
        result = env.step({0:(0.,)*8})
        assert env.success == success
        assert result.terminated == success
        if success:
            assert env.frames == 1
    finally:
        env.close()


def test_about_face_progress_tracks_head_motion(monkeypatch):
    from learning.inchworm_rewards import DEFAULTS
    env = scenario.environment(horizon=1/60,rewards={**dict.fromkeys(DEFAULTS,0.),'progress_reward':1.})
    try:
        env.reset(12,stage=8)
        head = env.worm.worm_front
        env.world.worm_goal = (head[0]+20.,head[1])
        monkeypatch.setattr(env.enc,'step',lambda dt: setattr(env.worm,'worm_front',(head[0]+2.,head[1])))
        result = env.step({0:(0.,)*8})
        assert result.reward == pytest.approx(2.)
        assert env.distance() == pytest.approx(18.)
        # A goal between centre and head is behind the head, despite being ahead of the centre.
        env.world.worm_goal = (head[0]-2.,head[1])
        assert env.target_alignment() == 0.
    finally:
        env.close()


def test_recorded_turn_that_previously_stalled_finishes_with_bounded_substeps(monkeypatch):
    from physics.inchworm import InchwormPhysics
    recorded = json.loads((Path(__file__).parent/'fixtures/inchworm-stall-actions.json').read_text())
    original = InchwormPhysics._twist_reflex
    calls = 0
    def counted(self, dt):
        nonlocal calls
        calls += 1
        assert calls < 10000, 'physics substeps stopped advancing the episode'
        return original(self, dt)
    monkeypatch.setattr(InchwormPhysics, '_twist_reflex', counted)
    env = scenario.environment(horizon=45.)
    try:
        env.reset(recorded['seed'],level=recorded['level'],stage=recorded['stage'])
        for action in recorded['actions']:
            calls = 0
            env.step({0:action})
        assert np.isfinite(env.snapshot().critic).all()
    finally:
        env.close()


@pytest.mark.parametrize('heading,movement,expected', [
    (0.,1.,6.), (math.pi,1.,0.), (math.pi/3,1.,3.),
    (math.pi/2,1.,0.), (0.,-1.,-6.), (math.pi,-1.,-6.), (0.,0.,0.),
])
def test_progress_rewards_head_first_approach_and_always_penalizes_retreat(monkeypatch,heading,movement,expected):
    from physics.inchworm import InchwormPhysics
    from learning.inchworm_rewards import DEFAULTS
    env = scenario.environment(horizon=.1,rewards={**dict.fromkeys(DEFAULTS,0.),'progress_reward':1.})
    try:
        env.reset(12,level=0,stage=1)
        env.world.worm_goal = (env.worm.x+100,env.worm.y)
        monkeypatch.setattr(InchwormPhysics,'heading',property(lambda self:heading))
        def move(dt):
            env.worm.x += movement
        monkeypatch.setattr(env.enc,'step',move)
        transition = env.step({0:(0.,)*8})
        assert transition.reward == pytest.approx(expected,abs=1e-9)
    finally:
        env.close()


def test_turning_in_place_at_end_cannot_reward_previous_backward_progress(monkeypatch):
    from physics.inchworm import InchwormPhysics
    from learning.inchworm_rewards import DEFAULTS
    env = scenario.environment(horizon=.1,rewards={**dict.fromkeys(DEFAULTS,0.),'progress_reward':1.})
    heading = math.pi
    try:
        env.reset(12,level=0,stage=1)
        env.world.worm_goal = (env.worm.x+100,env.worm.y)
        monkeypatch.setattr(InchwormPhysics,'heading',property(lambda self:heading))
        def move(dt):
            nonlocal heading
            env.worm.x += 1.
            if env.frames == 5:
                heading = 0.
        monkeypatch.setattr(env.enc,'step',move)
        assert env.step({0:(0.,)*8}).reward == pytest.approx(0.,abs=1e-9)
    finally:
        env.close()


def test_rear_and_curl_are_gated_before_turning_and_retained_later():
    state = scenario.initial_curriculum(None)
    visited = [state['stage']]
    for expected in scenario.CURRICULUM[1:]:
        evidence = scenario.validation_evidence(dict(success=1.,survived=1.),state,None)
        for _ in range(scenario.PASSES_REQUIRED.get(state['stage'],2)):
            assert state['stage'] != expected
            state = scenario.advance_curriculum(state,evidence)
        assert state['stage'] == expected
        visited.append(expected)
    assert visited == list(scenario.CURRICULUM)
    cases = scenario.training_cases(3,24,2)
    assert sum(c['stage']==2 for c in cases) >= len(cases)//2          # the current lesson keeps at least half
    assert {c['stage'] for c in cases} == {1,6,7,2}                     # and the earlier ones are all still practised


@pytest.mark.parametrize('stage',(6,7))
@pytest.mark.parametrize('level',(0,4))
def test_supported_reach_and_backward_curl_are_physically_achievable(stage,level):
    env = scenario.environment(horizon=6.)
    try:
        env.reset(12,level=level,stage=stage)
        goal = env.world.worm_goal
        initial_tail = env.worm.worm_physics.tail
        along = sum((goal[i]-initial_tail[i])*env.forward[i] for i in range(2))
        assert (along > 0) if stage == 6 else (along < 0)
        assert not env.success
        while not env.finished:
            env.step({0:scenario.teacher_action(env,0,None)})
        assert env.success and env.contact_frames >= 1
        assert env.distance() <= 12
        if stage == 7:
            assert env.worm.worm_rear_grip
        if stage == 7:
            assert env.worm.worm_physics.grips[0] is env.curl_anchor
            assert not env.curl_anchor_lost
    finally:
        env.close()


@pytest.mark.parametrize('gap,success',[(11.9,True),(12.,True),(12.1,False)])
def test_first_rear_lesson_only_requires_one_frame_near_target(monkeypatch,gap,success):
    env = scenario.environment(horizon=.1)
    try:
        env.reset(12,level=0,stage=6)
        head = env.worm.worm_front
        env.world.worm_goal = (head[0]+gap,head[1])
        env.worm.worm_rear_grip = False
        env.worm.worm_front_grip = False
        monkeypatch.setattr(env.enc,'step',lambda dt:None)
        transition = env.step({0:(0.,)*8})
        assert env.success is success
        assert transition.terminated is success
        if success:
            assert env.frames == 1
            assert env.contact_frames == 1
    finally:
        env.close()


def test_validation_uses_the_deployed_action_distribution():
    from learning.run import evaluate
    seen = []
    def pool(jobs):
        seen.extend(jobs)
        return [(None,{},[]) for job in jobs]
    args = SimpleNamespace(task='inchworm',opponent='',horizon=.1)
    evaluate(pool,args,'unused.npz',[dict(seed=12,level=4,stage=6)])
    assert seen[0]['mode'] == 'sample'
    # Deterministic inference remains available for explicit diagnostic comparisons.
    seen.clear()
    evaluate(pool,args,'unused.npz',[dict(seed=12,level=4,stage=6)],mode='greedy')
    assert seen[0]['mode'] == 'greedy'


def test_other_tasks_keep_deterministic_evaluation_by_default(monkeypatch):
    from learning.run import evaluate
    import learning.tasks
    definition = SimpleNamespace(adapter=lambda:SimpleNamespace(metrics=lambda rows:{}))
    monkeypatch.setattr(learning.tasks,'task',lambda name:definition)
    seen=[]
    def pool(jobs):
        seen.extend(jobs)
        return [(None,{},[]) for job in jobs]
    evaluate(pool,SimpleNamespace(task='other',opponent='',horizon=.1),'unused.npz',[dict(seed=1)])
    assert seen[0]['mode']=='greedy'


@pytest.mark.parametrize('gap,success',[(11.9,True),(12.,True),(12.1,False)])
def test_curl_success_is_one_reach_with_original_anchor(monkeypatch,gap,success):
    env = scenario.environment(horizon=.1)
    try:
        env.reset(12,level=0,stage=7)
        assert env.curl_anchor is not None
        head = env.worm.worm_front
        env.world.worm_goal = (head[0]+gap,head[1])
        monkeypatch.setattr(env.enc,'step',lambda dt:None)
        transition = env.step({0:(0.,)*5+(1.,0.,0.)})
        assert env.success is success
        if success:
            assert transition.terminated and env.frames == 1
    finally:
        env.close()


@pytest.mark.parametrize('replant',[False,True])
def test_curl_cannot_release_or_replant_the_starting_anchor_even_at_target(monkeypatch,replant):
    env = scenario.environment(horizon=.1)
    try:
        env.reset(12,level=0,stage=7)
        p = env.worm.worm_physics
        original = env.curl_anchor
        p._release_foot(0)
        if replant:
            p._try_foot(0)
            assert p.grips[0] is not None and p.grips[0] is not original
        env.world.worm_goal = env.worm.worm_front
        monkeypatch.setattr(env.enc,'step',lambda dt:None)
        transition = env.step({0:(0.,)*5+(1.,0.,0.)})
        assert not env.success and env.curl_anchor_lost
        assert transition.terminated and env.frames == 1
        with pytest.raises(RuntimeError,match='episode ended'):
            env.step({0:(0.,)*8})
    finally:
        env.close()


@pytest.mark.parametrize('heading,travel,expected',[(0.,-1.,-6.),(0.,1.,0.),(math.pi,-1.,0.),(math.pi,1.,-6.)])
def test_tail_first_translation_is_penalized_in_either_facing_direction(monkeypatch,heading,travel,expected):
    from physics.inchworm import InchwormPhysics
    from creatures.traits.inchworm import snapshot
    from learning.inchworm_rewards import DEFAULTS
    env=scenario.environment(horizon=.1,rewards={**dict.fromkeys(DEFAULTS,0.),'backward_cost':1.})
    try:
        env.reset(12,stage=1)
        env.backward_owed = 0.  # This check starts after the initial allowance is spent.
        monkeypatch.setattr(InchwormPhysics,'heading',property(lambda self:heading))
        def move(dt):
            env.worm.worm_physics.translate((travel,0.))
            snapshot(env.worm)
        monkeypatch.setattr(env.enc,'step',move)
        result=env.step({0:(0.,)*8})
        assert result.reward == pytest.approx(expected)
        assert result.events['backward_distance'] == pytest.approx(-expected)
    finally:env.close()




@pytest.mark.parametrize('stage,start,end,sign',[(8,0.,math.pi,1),(8,math.pi,0.,-1),(2,0.,math.pi,0)])
def test_about_face_pays_for_turning_toward_the_goal_and_only_there(monkeypatch,stage,start,end,sign):
    from physics.inchworm import InchwormPhysics
    from learning.inchworm_rewards import DEFAULTS
    env=scenario.environment(horizon=.1,rewards={**dict.fromkeys(DEFAULTS,0.),'turn_reward':1.})
    try:
        env.reset(12,stage=stage)
        heading=[start]
        monkeypatch.setattr(InchwormPhysics,'heading',property(lambda self:heading[0]))
        monkeypatch.setattr(env.enc,'step',lambda dt:heading.__setitem__(0,end))
        reward=env.step({0:(0.,)*8}).reward
        assert (reward > .5, reward < -.5) == ((True,False) if sign > 0 else (False,True) if sign < 0 else (False,False))
    finally:env.close()


def test_a_rolled_worm_sees_and_commands_its_bends_mirrored():
    from learning.adapters.inchworm import FEATURES, bend_signs, intention, observe
    env=scenario.environment(horizon=1.)
    try:
        env.reset(12,stage=1)
        p=env.worm.worm_physics
        p.step(InchwormCommand((.5,-.5,.5,-.5,.5)),20.)
        upright,_,_=observe(env.worm,env.enc)
        p.roll=[math.pi]*6
        assert bend_signs(p) == (-1.,)*5
        rolled,_,_=observe(env.worm,env.enc)
        bends=[i for i,n in enumerate(FEATURES) if n.startswith(('angle','rate','activation'))]
        assert rolled[bends] == pytest.approx(-upright[bends])
        assert intention((1.,)*5+(0,0,0),bend_signs(p)).muscles == pytest.approx((-math.tanh(1.),)*5)
        p.roll=[0.]*6
        assert intention((1.,)*5+(0,0,0),bend_signs(p)).muscles == pytest.approx((math.tanh(1.),)*5)
    finally:env.close()


@pytest.mark.parametrize('stage,labelled',[(8,True),(1,False)])
def test_only_about_face_asks_the_scripted_controller_what_it_would_do_in_visited_states(stage,labelled):
    from learning.policy import random_policy
    from learning.training import collect
    env=scenario.environment(horizon=3.)
    try:
        assert env.teacher_weight > 0
        episodes,_,_=collect(env,random_policy(scenario.SPEC,0),12,dict(stage=stage),
                             labeller=lambda e,i:scenario.label_action(e,i,None))
        labels=episodes[0]['teacher']
        assert len(labels) == len(episodes[0]['x'])
        assert all((l is not None and len(l) == 9 and l[5] in (0,1) and l[6] in (0,1)) == labelled for l in labels)
    finally:env.close()


@pytest.mark.parametrize('idle_seconds',[0.,10.])
def test_the_about_face_expert_wins_even_when_the_worm_has_sat_idle_first(idle_seconds):
    """Labels must be right for the states a learner reaches, including one that stalled before it tried."""
    from learning.adapters.inchworm import action_for, bend_signs
    from physics.inchworm import InchwormCommand
    env=scenario.environment(horizon=60.)
    try:
        env.reset(40,stage=8)
        p=env.worm.worm_physics
        while not env.finished:
            command=(InchwormCommand(tuple(p.activation),True,True) if env.frames/60 < idle_seconds
                     else scenario.posture_expert(env))
            env.step({0:action_for(command,bend_signs(p))})
        assert env.success
    finally:env.close()


def test_a_rear_segment_that_swings_back_and_forth_is_not_walking_backward(monkeypatch):
    from physics.inchworm import InchwormPhysics
    from creatures.traits.inchworm import snapshot
    from learning.inchworm_rewards import DEFAULTS
    env=scenario.environment(horizon=.1,rewards={**dict.fromkeys(DEFAULTS,0.),'backward_cost':1.})
    try:
        env.reset(12,stage=1)
        monkeypatch.setattr(InchwormPhysics,'heading',property(lambda self:0.))
        swings=iter([1.,-1.]*3)
        def move(dt):
            env.worm.worm_physics.translate((next(swings),0.))
            snapshot(env.worm)
        monkeypatch.setattr(env.enc,'step',move)
        result=env.step({0:(0.,)*8})
        assert result.reward == pytest.approx(0.) and result.events['backward_distance'] == pytest.approx(0.)
    finally:env.close()


def test_retreat_is_free_until_it_exceeds_the_forward_travel_credited_against_it(monkeypatch):
    from physics.inchworm import InchwormPhysics
    from creatures.traits.inchworm import snapshot
    from learning.inchworm_rewards import DEFAULTS
    env=scenario.environment(horizon=.3,rewards={**dict.fromkeys(DEFAULTS,0.),'backward_cost':1.})
    try:
        env.reset(12,stage=1)
        env.backward_owed = 0.  # Check replenishment after spending the initial allowance.
        monkeypatch.setattr(InchwormPhysics,'heading',property(lambda self:0.))
        steps=iter([2.]*6+[-2.]*12)                     # 12 px forward, then 24 px back, in three decisions
        def move(dt):
            env.worm.worm_physics.translate((next(steps),0.))
            snapshot(env.worm)
        monkeypatch.setattr(env.enc,'step',move)
        rewards=[env.step({0:(0.,)*8}).reward for _ in range(3)]
        assert rewards == pytest.approx([0.,0.,-12.])    # the back swing repays the credit, then 12 px is walked back
    finally:env.close()


@pytest.mark.parametrize('heading', [0., math.pi])
def test_initial_backward_allowance_is_bounded_in_both_directions(monkeypatch, heading):
    from physics.inchworm import InchwormPhysics
    from creatures.traits.inchworm import snapshot
    from learning.inchworm_rewards import DEFAULTS
    env = scenario.environment(horizon=.4, rewards={**dict.fromkeys(DEFAULTS, 0.), 'backward_cost': 1.})
    try:
        env.reset(12, stage=1)
        monkeypatch.setattr(InchwormPhysics, 'heading', property(lambda self: heading))
        def move(dt):
            env.worm.worm_physics.translate((-2.*math.cos(heading), 0.))
            snapshot(env.worm)
        monkeypatch.setattr(env.enc, 'step', move)
        rewards = [env.step({0: (0.,)*8}).reward for _ in range(4)]
        # 48 px of sustained retreat spends the 30 px allowance, then pays for 18 px.
        assert rewards == pytest.approx([0., 0., -6., -12.])
        assert env.backward_distance == pytest.approx(18.)
    finally:
        env.close()


def test_scripted_about_face_first_strides_do_not_pay_backward_penalty():
    env = scenario.environment(horizon=120.)
    try:
        env.reset(2000000800, level=4, stage=8)
        while not env.finished:
            env.step({0: scenario.teacher_action(env, 0, None)})
        assert env.success
        assert env.backward_distance == pytest.approx(0.)
    finally:
        env.close()


def test_a_rolled_worm_sees_the_surface_on_the_same_side_as_an_upright_one(monkeypatch):
    """After the turn the body frame is mirrored; the surface must still appear on the side the belly faces."""
    from physics.inchworm import InchwormPhysics
    from learning.adapters.inchworm import FEATURES, observe
    env=scenario.environment(horizon=1.)
    try:
        env.reset(12,stage=1)
        p=env.worm.worm_physics
        upright,_,_=observe(env.worm,env.enc)
        p.roll=[math.pi]*6
        monkeypatch.setattr(InchwormPhysics,'heading',property(lambda self:math.pi))
        # Turned about to face the opposite way: the world is the same, so forward-facing parts swap, sides do not.
        rolled,_,_=observe(env.worm,env.enc)
        for name in ('rear_normal_side','front_normal_side'):
            assert rolled[FEATURES.index(name)] == pytest.approx(upright[FEATURES.index(name)],abs=1e-5)
    finally:env.close()


def test_a_brief_phase_of_the_expert_is_not_drowned_out_by_a_long_one():
    """Releasing the rear lasts a few decisions between long stretches of other phases; weighting every decision
    equally taught it far too slowly."""
    import torch
    from learning.adapters.inchworm import CRITIC_SIZE, SPEC
    from learning.policy import random_policy
    from learning.ppo import update_team
    from learning.training import build_network, collect
    policy=random_policy(SPEC,0)
    env=scenario.environment(horizon=3.)
    try:
        base=collect(env,policy,12,dict(stage=8))[0][0]
    finally:env.close()
    n=len(base['x']); rare=range(n-3,n)
    def release_probability_after_training(with_phases):
        episode=dict(base,teacher=[(0.,)*5+((1.,1.,0.) if t not in rare else (0.,1.,0.))+((float(t in rare),) if with_phases else ())
                                   for t in range(n)])
        torch.manual_seed(0)
        net=build_network(policy,CRITIC_SIZE)
        optimiser=torch.optim.Adam(net.parameters(),lr=3e-3)
        rng=np.random.default_rng(0)
        for _ in range(3):
            update_team(net,optimiser,[episode],rng,teacher_weight=5.,teacher_heads=('rear_grip',))
        x=torch.as_tensor(np.asarray(base['x']),dtype=torch.float32)[None]
        critic=torch.as_tensor(np.asarray(base['critic']),dtype=torch.float32)[None]
        with torch.no_grad():
            probabilities=torch.softmax(net(x,critic)[0]['rear_grip'][0],-1)[:,0]
        return float(probabilities[list(rare)].mean())
    assert release_probability_after_training(True) > release_probability_after_training(False)+.3


def test_crawl_pays_more_for_arriving_sooner():
    def arrival_reward(frames):
        env = scenario.environment(horizon=120.,rewards=dict(DEFAULTS))
        env.reset(1,stage=1,level=4)
        env.world.worm_goal = tuple(env.worm.get_pos())   # the worm is already on the marker
        env.frames = frames
        for _ in range(60):
            transition = env.step({0:(0.,)*5+(1.,1.,0.)})   # plant both feet
            if env.finished:
                break
        assert env.success
        env.close()
        return transition.reward
    assert arrival_reward(600) > arrival_reward(5400) + 1.


def test_crawl_goals_sit_on_uneven_ground_that_can_still_be_crawled():
    rough = [seed for seed in range(40) if len(scenario.InchwormEnvironment.terrain(seed,1,350.)) > 2]
    assert rough and len(rough) < 40                      # some episodes are uneven, some stay level
    assert all(len(scenario.InchwormEnvironment.terrain(seed,6,350.)) == 2 for seed in range(40))   # other drills are unchanged
    for seed in rough[:3]:
        env = scenario.environment(horizon=120.,scripted=True,rewards=dict(DEFAULTS))
        env.reset(seed,stage=1,level=4)
        heights = [h for _,h in env.heights]
        assert max(heights)-min(heights) > 4              # genuinely bumpy
        while not env.finished:
            env.step({0:(0.,)*8})
        assert env.success                                # the marker is reachable over the bumps
        env.close()


def test_scripted_about_face_walks_on_soon_after_the_turn():
    env = scenario.environment(horizon=60.,scripted=True,rewards=dict(DEFAULTS))
    env.reset(500,stage=8,level=4)
    p = env.worm.worm_physics
    turned = start = None
    for decision in range(400):
        env.step({0:(0.,)*8})
        stretch = min(1.,math.dist(p.head,p.tail)/(6*p.length))
        distance = math.dist(p.head,env.world.worm_goal)
        if turned is None and env.goal_facing() > .8 and stretch > .6 and p.grips[0] is None:
            turned,start = decision,distance
        if turned is not None and distance < start-10:
            break
    env.close()
    assert turned is not None
    assert decision-turned < 40          # about 4 s; waiting out a fixed hold took nearly 6 s


def test_the_expert_teaches_the_turn_and_the_rear_foot_down_but_not_the_walk():
    env = scenario.environment(horizon=60.,scripted=True,rewards=dict(DEFAULTS))
    env.reset(500,stage=8,level=4)
    labels = []
    while not env.finished:
        labels.append(scenario.label_action(env,0,None))
        env.step({0:(0.,)*8})
    env.close()
    assert env.success
    rear = [n for n,_ in scenario.SPEC.heads].index('rear_grip')
    taught = [l for l in labels if l is not None]
    assert any(not math.isnan(l[0]) for l in taught)                    # the curl, landing and release are taught fully
    walking = [l for l in taught if math.isnan(l[0])]
    assert walking and all(l[rear] == 1. and sum(not math.isnan(v) for v in l[:8]) == 1 for l in walking)   # after the flip: only "rear foot on"
    assert all(l is None for l in labels[-20:])                         # and once it has landed, nothing: the walk is the policy's own


def test_crawl_charges_a_head_pointing_at_the_ground_but_not_a_level_one():
    def one_step_reward(cost,head_angle=None):
        env = scenario.environment(horizon=60.,rewards=dict(DEFAULTS,end_pitch_cost=cost))
        env.reset(1,stage=1,level=0)
        if head_angle is not None:
            env.worm.worm_physics.links[-1].angle = head_angle
        reward = env.step({0:(0.,)*8}).reward
        env.close()
        return reward
    level = one_step_reward(10.)-one_step_reward(0.)
    upright = one_step_reward(10.,math.pi/2)-one_step_reward(0.,math.pi/2)
    assert abs(level) < 1e-6                 # a head lying along the ground costs nothing
    assert upright < -.02                    # pointing straight up or down at it does


def test_every_batch_after_crawl_keeps_a_quarter_of_its_episodes_on_crawl():
    for stage in (6,8,5):
        for iteration in range(5):
            cases = scenario.training_cases(iteration,24,stage,curriculum=dict(stage=stage))
            assert sum(c['stage'] == 1 for c in cases) >= len(cases)//4
            assert sum(c['stage'] == stage for c in cases) >= len(cases)//2     # the current lesson still gets half
    assert {c['stage'] for c in scenario.training_cases(0,24,1,curriculum=dict(stage=1))} == {1}     # Crawl itself is unchanged


CROSSING_SEED = 2000000906    # one of the seeds 2000000900-11 where the scripted stride crosses the gap (and quickly)


def test_gap_lesson_puts_the_goal_on_a_second_island_that_a_worm_can_sometimes_cross():
    for seed in range(2000000900,2000000912):
        env = scenario.environment(horizon=60.,scripted=True,rewards=dict(DEFAULTS))
        env.reset(seed,stage=9,level=4)
        first,second = env.islands_built
        gap = second[0][0]-first[-1][0]
        assert 14. <= gap <= 42.                                   # a real gap, within what a front foot can span
        walls = env.world.maze.walls
        assert len(walls) == 8                                      # two solid blocks: top, both end faces and underside each
        assert max(math.dist(w.start,w.end) for w in walls) > 100. and any(abs(math.dist(w.start,w.end)-scenario.GAP_BLOCK) < 1e-6 for w in walls)
        goal_along = (env.world.worm_goal[0]-env.origin[0])*env.forward[0]+(env.world.worm_goal[1]-env.origin[1])*env.forward[1]
        assert goal_along > second[0][0]                           # the marker is on the far island
        assert env.worm.get_pos() != env.world.worm_goal and env.distance() > 100.
        env.close()
    # Solvable: even the scripted stride (deterministic) makes it on some seeds. Flying all twelve took ~27 s.
    env = scenario.environment(horizon=60.,scripted=True,rewards=dict(DEFAULTS))
    env.reset(CROSSING_SEED,stage=9,level=4)
    while not env.finished:
        env.step({0:(0.,)*8})
    assert env.success
    env.close()
    assert 9 in scenario.CURRICULUM and 9 not in scenario.DEMONSTRATED


def test_the_bundled_inchworm_brain_loads_and_drives_a_game_worm(monkeypatch):
    from tests.test_inchworm_creature import setting
    monkeypatch.delenv('INCHWORM_POLICY')                      # as in the game: no override, so the bundled brain
    c,enc = setting()
    assert c.worm_driver is not None
    gripping = 0
    for _ in range(120):
        enc.step(1.)
        gripping += bool(c.worm_front_grip or c.worm_rear_grip)
    # The brain samples its actions, so any single frame may catch both ends lifted mid-stride (this used to flake
    # about 1 run in 3). A worm the brain is really driving holds the wall for most of the run (~95 of 120 frames).
    assert c.active and c.worm_driver.state.any() and gripping > 60
    enc.close()
