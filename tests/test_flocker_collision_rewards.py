"""Explicit wall impact credit is local, once per event and absent for gentle perches."""
from learning import flocker_rewards
from learning.adapters.flocker_scenario import environment
from tests.creature_parts import open_maze
from tests.test_neural_flocker import wall


def collision_scene(penalty=.1,stage=3):
    rewards={k:0. for k in flocker_rewards.DEFAULTS}
    rewards['wall_collision_penalty']=penalty
    env=environment(horizon=.3,rewards=rewards)
    env.reset(4,level=13,team_size=2,stage=stage,moving=False)
    env.world.maze=open_maze((wall(),))
    env.team.maze=env.world.maze
    env.obstacles=None
    a,b=env.team.members()
    a.x,a.y=200.+a.radius+3.,300.
    a.prev_x,a.prev_y=a.get_pos()
    a.angle=0.
    a.vx,a.vy=-3.,0.
    b.x,b.y=500.,600.
    b.prev_x,b.prev_y=b.get_pos()
    b.vx=b.vy=0.
    actions={i:(0.,-20.,0.,0,0,0,0) for i in (0,1)}
    return env,a,b,actions


def test_wall_penalty_is_charged_once_to_the_colliding_bird():
    env,a,b,actions=collision_scene()
    try:
        result=env.step(actions)
        assert result.events['hard_landings']==1
        assert result.agent_rewards[a.agent_id]==-.1
        assert result.agent_rewards[b.agent_id]==0.
        assert result.reward==-.1
        assert result.events['wall_collision_cost']==.1
        result=env.step(actions)
        assert result.reward==0.
        assert result.events['wall_collision_cost']==.1
        assert result.events['agent_wall_collision_costs']=={a.agent_id:.1}
    finally:
        env.close()


def test_gentle_perching_has_no_wall_penalty():
    env,a,b,actions=collision_scene(stage=2)
    try:
        a.vx=0.
        a.flocker_quiet=60
        actions[a.agent_id]=(0.,-20.,0.,0,0,0,1)
        result=env.step(actions)
        assert a.flocker_perched
        assert result.events['hard_landings']==0
        assert result.events['wall_collision_cost']==0.
        assert result.reward==0.
    finally:
        env.close()


def test_collision_credit_is_retained_when_the_bird_dies_later_in_the_step():
    env,a,b,actions=collision_scene()
    try:
        def die_after_contact(scene):
            if a.active and a.flocker_hard_landings:
                scene.enc.kill(a)
        env.render_frames=die_after_contact
        result=env.step(actions)
        assert not a.active
        assert result.agent_rewards[a.agent_id]==-.1
        assert result.events['agent_wall_collision_costs']=={a.agent_id:.1}
    finally:
        env.close()


def test_old_manifests_and_settings_keep_zero_collision_cost():
    assert flocker_rewards.from_manifest({'rewards':{'hit_reward':1.}})['wall_collision_penalty']==0.
    assert flocker_rewards.recovered_settings({'manifest':{}})['flocker_wall_collision_penalty']==0.
    assert flocker_rewards.DEFAULTS['wall_collision_penalty']==.1


def test_smaller_spacing_allows_close_flocking_without_overlap():
    env,a,b,actions=collision_scene()
    try:
        a.x,a.y=500.,400.
        b.x,b.y=530.,400.
        a.vx=a.vy=b.vx=b.vy=0.
        env.rewards['bunch_penalty']=.01
        env.rewards['bunch_distance']=22.
        assert env.step(actions).reward==0.
        env.rewards['bunch_distance']=35.
        assert env.step(actions).reward<0.
    finally:
        env.close()


def test_contact_only_allows_close_flight_but_penalizes_touching_and_overlap():
    import pytest
    for factor,expected in ((1.001,0.),(1.1,0.),(1.,-.02),(.9,-.02),(0.,-.02)):
        env,a,b,actions=collision_scene()
        try:
            a.x,a.y=500.,400.
            distance=(a.radius+b.radius)*factor
            b.x,b.y=500.+distance,400.
            a.vx=a.vy=b.vx=b.vy=0.
            env.rewards.update(contact_only=1.,bunch_penalty=.01,bunch_distance=100.)
            result=env.step(actions)
            assert result.reward==pytest.approx(expected)
            # Even exact overlap remains possible: no physical separation or bounce.
            assert b.x-a.x==pytest.approx(distance)
            assert (a.vx,a.vy,b.vx,b.vy)==(0.,0.,0.,0.)
        finally:
            env.close()


def test_old_recordings_retain_the_proximity_rule():
    assert flocker_rewards.from_manifest({'rewards':{'bunch_distance':22.}})['contact_only']==0.
    assert flocker_rewards.DEFAULTS['contact_only']==1.


def test_overlapping_birds_can_fly_through_each_other_without_bouncing():
    env,a,b,actions=collision_scene()
    try:
        a.x,a.y=500.,400.
        b.x,b.y=506.,400.
        a.angle,b.angle=0.,180.
        a.vx,a.vy=1.,0.
        b.vx,b.vy=-1.,0.
        env.rewards.update(contact_only=1.,bunch_penalty=.01)
        result=env.step(actions)
        assert a.x>b.x
        assert a.vx>0 and b.vx<0
        assert result.reward<0.
    finally:
        env.close()


def test_separated_bird_earns_regrouping_credit_without_an_adversary():
    env,a,b,_=collision_scene()
    try:
        env.world.ship=env.world.hunter=None
        env.world.additional_targets=[]
        env.targets=[]
        a.x,a.y=400.,400.
        b.x,b.y=650.,400.
        a.vx,a.vy=2.,0.
        b.vx=b.vy=0.
        env.rewards['cohesion_reward']=.002
        toward=env.event_rows()
        env.sample_motion(toward)
        a.vx=-2.
        away=env.event_rows()
        env.sample_motion(away)
        toward,away=env.paid(toward),env.paid(away)
        assert toward[a.agent_id]>away[a.agent_id]
        assert toward[a.agent_id]>0.
        assert toward[b.agent_id]==0.
    finally:
        env.close()


def test_regrouping_credit_does_not_pull_birds_into_a_wall_or_toward_unseen_birds():
    env,a,b,_=collision_scene()
    try:
        env.rewards['cohesion_reward']=.02
        env.targets=[]
        env.world.additional_targets=[]
        a.x,a.y=150.,300.
        b.x,b.y=450.,300.
        a.vx,a.vy=2.,0.
        b.vx=b.vy=0.
        credit=env.event_rows()
        env.sample_motion(credit)
        assert env.paid(credit)[a.agent_id]==0.
        env.world.maze=open_maze()
        b.x=750.
        env.sample_motion(credit)
        assert env.paid(credit)[a.agent_id]==0.
    finally:
        env.close()


def test_nearby_cohesion_rewards_velocity_alignment_not_just_proximity():
    import pytest
    env,a,b,_=collision_scene()
    try:
        env.world.maze=open_maze()
        env.targets=[]
        env.world.additional_targets=[]
        a.x,a.y=400.,400.
        b.x,b.y=440.,400.
        env.rewards['cohesion_reward']=.02
        a.vx,a.vy=2.,0.
        a.angle,b.angle=0.,180.  # Actual travel direction matters, not body angle.
        credits=[]
        for velocity in ((1.,0.),(1.,1.),(0.,2.),(-2.,0.),(0.,0.)):
            b.vx,b.vy=velocity
            credit=env.event_rows()
            env.sample_motion(credit)
            credits.append(env.paid(credit)[a.agent_id])
        assert credits[0]==pytest.approx(.02/6)
        assert credits[0]>credits[1]>credits[2]
        assert credits[2:]==[0.,0.,0.]
        a.vx=a.vy=0.
        b.vx=2.
        credit=env.event_rows()
        env.sample_motion(credit)
        assert env.paid(credit)[a.agent_id]==0.
    finally:
        env.close()


def test_following_an_occluded_target_report_earns_pursuit_progress():
    env,a,b,actions=collision_scene()
    try:
        a.x,a.y=150.,300.
        b.x,b.y=250.,300.
        a.angle=0.;a.vx=2.;a.vy=0.
        b.vx=b.vy=0.
        t=env.targets[0];t.x,t.y=350.,300.;t.vx=t.vy=0.
        a.flocker_target=None
        env.team.alerts={b.agent_id:(env.frames,b.get_pos(),t.get_pos(),(0.,0.))}
        env.team._sightings={}
        env.team._prepared_frame=env.frames
        env.rewards['progress_reward']=.01
        result=env.step(actions)
        assert result.agent_rewards[a.agent_id]>0.
    finally:
        env.close()


def test_stationary_bird_does_not_earn_progress_from_a_target_approaching_it():
    env,a,b,actions=collision_scene()
    try:
        env.world.maze=open_maze();env.team.maze=env.world.maze
        a.x,a.y=400.,300.;a.vx=a.vy=0.
        b.x,b.y=900.,600.;b.vx=b.vy=0.
        t=env.targets[0];t.x,t.y=600.,300.;t.vx,t.vy=-1.,0.
        a.flocker_target=t
        env.rewards['progress_reward']=.01
        result=env.step(actions)
        assert result.agent_rewards[a.agent_id]==0.
    finally:
        env.close()


def test_waypoint_teleport_does_not_create_a_false_progress_penalty():
    import pytest
    env,a,b,actions=collision_scene(stage=1)
    try:
        env.world.maze=open_maze();env.team.maze=env.world.maze
        a.x,a.y=400.,300.;a.vx=a.vy=0.
        b.x,b.y=900.,600.;b.vx=b.vy=0.
        t=env.targets[0];t.x,t.y=480.,300.;t.vx=t.vy=0.
        a.flocker_target=t
        env.rewards['progress_reward']=.01
        result=env.step(actions)
        assert result.events['approaches']==1
        assert result.agent_rewards[a.agent_id]==pytest.approx(.5)
        assert result.events.get('pursuit_progress_reward',0)==0.
    finally:
        env.close()


def test_aiming_progress_rewards_a_correct_head_turn_without_firing():
    env,a,b,actions=collision_scene()
    try:
        env.world.maze=open_maze();env.team.maze=env.world.maze
        a.x,a.y=400.,300.;b.x,b.y=1000.,600.
        a.angle=0.;a.flocker_head=30.
        a.vx=a.vy=b.vx=b.vy=0.
        target=env.targets[0];target.x,target.y=600.,300.;target.vx=target.vy=0.
        env.rewards['aim_progress_reward']=.2
        actions[a.agent_id]=(0.,-20.,-20.,0,0,0,0)
        result=env.step(actions)
        assert result.agent_rewards[a.agent_id]>0.
        assert result.events.get('shots',0)==0
    finally:
        env.close()


def test_target_motion_alone_does_not_earn_aiming_progress():
    env,a,b,actions=collision_scene()
    try:
        env.world.maze=open_maze();env.team.maze=env.world.maze
        a.x,a.y=400.,300.;b.x,b.y=1000.,600.
        a.angle=0.;a.flocker_head=30.
        a.vx=a.vy=b.vx=b.vy=0.
        target=env.targets[0];target.x,target.y=600.,300.;target.vx,target.vy=0.,1.
        env.rewards['aim_progress_reward']=.2
        assert env.step(actions).agent_rewards[a.agent_id]==0.
    finally:
        env.close()


def test_aiming_geometry_leads_a_moving_target_and_rejects_an_impossible_shot():
    import math
    import config
    import pytest
    from learning.flocker_aim import intercept_point
    from creatures.flocker_team import muzzle
    env,a,b,_=collision_scene()
    try:
        env.world.maze=open_maze()
        a.x,a.y=400.,300.;a.angle=a.flocker_head=0.
        t=env.targets[0];t.x,t.y=600.,300.;t.vx,t.vy=0.,1.
        point=intercept_point(a,t,env.world.maze)
        assert point[1]>t.y
        origin,_=muzzle(a)
        time=(point[1]-t.y)/t.vy
        assert math.dist(origin,point)==pytest.approx(config.PROJECTILE_SPEED*time)
        t.vx,t.vy=config.PROJECTILE_SPEED*2.,0.
        assert intercept_point(a,t,env.world.maze) is None
    finally:
        env.close()


def test_head_wiggling_cannot_farm_aiming_progress_against_a_static_target():
    import pytest
    env,a,b,actions=collision_scene()
    try:
        env.world.maze=open_maze();env.team.maze=env.world.maze
        a.x,a.y=400.,300.;b.x,b.y=1000.,600.
        a.angle=0.;a.flocker_head=30.
        a.vx=a.vy=b.vx=b.vy=0.
        t=env.targets[0];t.x,t.y=600.,300.;t.vx=t.vy=0.
        env.rewards['aim_progress_reward']=.2
        actions[a.agent_id]=(0.,-20.,20.,0,0,0,0)
        away=env.step(actions).agent_rewards[a.agent_id]
        actions[a.agent_id]=(0.,-20.,-20.,0,0,0,0)
        toward=env.step(actions).agent_rewards[a.agent_id]
        assert away<0 and toward>0
        assert away+toward==pytest.approx(0.,abs=1e-9)
    finally:
        env.close()
