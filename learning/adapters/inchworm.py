"""Versioned body-relative senses and eight motor outputs for one inchworm."""
import math
import zipfile
import numpy as np

from creatures.traits.rooted import wall_between
from learning.policy import PolicySpec, RecurrentPolicy
from physics.inchworm import InchwormCommand, MUSCLES
from utils import get_closest_point_on_line
from weapons.geometry import segment_contact

DECISION_FRAMES = 6
# Gameplay and evaluation must execute the same learned action distribution.
INFERENCE_ACTION_MODE = 'sample'
SIGHT = 300.
MOTOR_NAMES = tuple(f'bend{i}' for i in range(5))
HEADS = tuple((n, 2) for n in MOTOR_NAMES) + (('rear_grip', 2), ('front_grip', 2), ('bite', 2))
FEATURES = tuple(f'{group}{i}' for group, count in (('angle',5),('rate',5),('activation',5),
                   ('roll_sin',6),('roll_cos',6)) for i in range(count))
FEATURES += ('velocity_forward','velocity_side','angular_velocity','rear_grip','front_grip',
             'rear_load','front_load') + tuple(f'contact{i}' for i in range(6))
FEATURES += ('grabbing','effort','bite_charge','stun')
FEATURES += tuple('previous_'+n for n,_ in HEADS)
FEATURES += tuple(f'{end}_ray{i}' for end in ('rear','front') for i in range(8))
FEATURES += tuple(f'{end}_{name}' for end in ('rear','front') for name in ('surface_distance','normal_forward','normal_side'))
FEATURES += ('target_present','target_forward','target_side','target_vforward','target_vside','target_radius',
             'goal_present','goal_forward','goal_side')
FEATURES += ('aim_cos','aim_sin','aim_speed')    # where the body's motion points relative to the goal, and how fast it moves
SPEC = PolicySpec('inchworm', FEATURES, HEADS, version=1, continuous=MOTOR_NAMES)
CRITIC_SIZE = len(FEATURES)  # No hidden world state is needed for these single-agent drills.


def visible_target(c, enc):
    target = c.worm_prey if c.worm_prey is not None and c.worm_prey.active else enc.target_entity_for(c)
    if (target is None or not target.active or math.dist(c.get_pos(), target.get_pos()) > SIGHT
            or wall_between(c.worm_front, target.get_pos(), enc.maze)):
        return None
    return target


def bend_signs(p):
    """+1 per joint while its links lie belly-down, -1 once the twist reflex has rolled them over.

    A rolled worm bends the opposite way to arch. Mirroring the five bends by these signs, in what the policy
    sees and in what it commands, makes a worm that has turned around look and act like one that has not,
    so the crawl it already knows carries over.
    """
    return tuple(1. if math.cos((p.roll[i]+p.roll[i+1])/2) >= 0 else -1. for i in range(MUSCLES))


def body_side(p):
    """+1 belly-down in the body frame, -1 once the worm has rolled over: then the surface lies on the other side."""
    return 1. if sum(math.cos(r) for r in p.roll) >= 0 else -1.


def observe(c, enc):
    p = c.worm_physics
    o = p.observe()
    signs = bend_signs(p)
    side = body_side(p)
    f = p.heading
    cf, sf = math.cos(f), math.sin(f)
    def local(dx, dy, scale):
        return ((dx*cf+dy*sf)/scale, side*(-dx*sf+dy*cf)/scale)
    x = [v*(signs[i] if group in ('joint_angles','joint_rates','activation') else 1.)
         for group in ('joint_angles','joint_rates','activation','roll_sin','roll_cos') for i,v in enumerate(o[group])]
    x += [o['velocity'][0], side*o['velocity'][1], side*o['angular_velocity'], *o['grips'], *o['grip_loads'], *o['contacts'],
          o['grabbing'], o['effort'], c.worm_bite/max(1., c.worm_bite_interval), c.worm_stun/30.]
    previous = getattr(c, 'worm_last_command', InchwormCommand())
    x += [*(m*sign for m,sign in zip(previous.muscles,signs)), previous.rear_grip, previous.front_grip, previous.bite]
    grid = getattr(enc.maze, 'spatial_grid', None)
    walls = list(grid.get_nearby_walls(c.get_pos(), SIGHT+p.length*4) if grid else enc.maze.walls)
    walls += list(getattr(enc.maze, 'gate_segments', ()))
    walls = [w for w in walls if w.active and w.start != w.end]
    for origin in (p.tail, p.head):
        for i in range(8):
            angle = f+side*math.tau*i/8
            delta = (math.cos(angle)*SIGHT, math.sin(angle)*SIGHT)
            hits = [hit[0] for w in walls if (hit := segment_contact(origin, delta, w.start, w.end, .1)) is not None]
            x.append(min(hits, default=1.))
    for origin in (p.tail, p.head):
        feet = [get_closest_point_on_line(origin,w.start,w.end) for w in walls]
        foot = min(feet, key=lambda v: math.dist(origin,v), default=None)
        if foot is None:
            x += [1., 0., 0.]
        else:
            dx, dy = origin[0]-foot[0], origin[1]-foot[1]
            distance = max(1e-8, math.hypot(dx,dy))
            x += [min(1., distance/p.reach), *local(dx,dy,distance)]
    target = visible_target(c, enc)
    if target is None:
        x += [0.]*6
    else:
        x += [1., *local(target.x-c.x,target.y-c.y,SIGHT),
              *local(target.vx-c.vx,target.vy-c.vy,8.), target.radius/50.]
    goal = getattr(enc.world, 'worm_goal', None)
    if goal is None and target is not None:
        goal = target.get_pos()
    x += [1., *local(goal[0]-c.x,goal[1]-c.y,SIGHT)] if goal is not None else [0.]*3
    vx, vy = p.velocity
    speed = math.hypot(vx,vy)
    if goal is None or speed < 1e-6:
        x += [0.]*3
    else:        # the angle between the centre's velocity and the goal: a whip lets go when it is near zero
        off = math.atan2(goal[1]-p.centre[1],goal[0]-p.centre[0])-math.atan2(vy,vx)
        x += [math.cos(off), side*math.sin(off), speed]
    masks = {name:np.ones(size,bool) for name,size in HEADS}
    if target is None:
        masks['bite'][1] = False
    if c.worm_stun > 0:
        for name in MOTOR_NAMES:
            masks[name][:] = False
        for name in ('rear_grip','front_grip','bite'):
            masks[name][:] = (True,False)
    return np.clip(np.asarray(x,np.float32),-5.,5.), masks, ()


def intention(action, signs=(1.,)*5):
    if len(action) != len(HEADS) or not np.isfinite(action).all():
        raise ValueError('inchworm action must contain eight finite values')
    if any(v not in (0,1) for v in action[5:]):
        raise ValueError('inchworm grip and bite actions must be binary')
    return InchwormCommand(tuple(math.tanh(float(v))*sign for v,sign in zip(action[:5],signs)), *(bool(v) for v in action[5:]))


def action_for(command, signs=(1.,)*5):
    return (*tuple(float(np.arctanh(np.clip(v*sign,-.999,.999))) for v,sign in zip(command.muscles,signs)),
            int(command.rear_grip), int(command.front_grip), int(command.bite))


def enabled(x, action, masks):
    return [bool(masks[n].any()) if n in MOTOR_NAMES else bool(masks[n].sum()>1) for n,_ in HEADS]


def load_transfer_policy(path):
    """A checkpoint of an older observation layout, carried over: the features it already had keep their weights and
    normalisation, and each new feature starts with zero weights, so the migrated worm acts exactly as the old one did
    until training finds a use for the new inputs."""
    try:
        old = RecurrentPolicy.load(path)
    except zipfile.BadZipFile as exc:
        raise ValueError(f'damaged inchworm checkpoint: {exc}') from exc
    if old.spec == SPEC:
        return old
    if old.spec.task != SPEC.task or old.spec.heads != SPEC.heads or old.spec.continuous != SPEC.continuous:
        raise ValueError('checkpoint task or observation/action schema does not match')
    where = {name: i for i, name in enumerate(old.spec.features)}
    embed = np.zeros((old.params['embed_w'].shape[0], len(FEATURES)), np.float32)
    mean, scale = np.zeros(len(FEATURES), np.float32), np.ones(len(FEATURES), np.float32)
    for i, name in enumerate(FEATURES):
        if name in where:
            embed[:, i], mean[i], scale[i] = old.params['embed_w'][:, where[name]], old.mean[where[name]], old.scale[where[name]]
    return RecurrentPolicy(dict(old.params, embed_w=embed), mean, scale, SPEC)


def load_policy(path):
    policy = load_transfer_policy(path)
    if not np.isfinite(policy.mean).all() or not np.isfinite(policy.scale).all():
        raise ValueError('nonfinite inchworm normalization')
    return policy


class InchwormDriver:
    """Independent memory and sampling stream; same action distribution as training.

    An explicit episode seed reproduces agent zero's training actions. Gameplay
    defaults to a fresh independent stream; recorded actions remain replayable.
    """
    def __init__(self, policy, seed=None):
        self.policy = policy
        self.rng = np.random.default_rng(None if seed is None else np.random.SeedSequence([seed,0]))
        self.state = policy.initial_state()
        self.remaining = 0.
        self.command = InchwormCommand()

    def advance(self, c, enc, dt):
        if self.remaining <= 1e-8:
            x, masks, _ = observe(c, enc)
            action, self.state = self.policy.choose(x,self.state,masks,
                                                   greedy=INFERENCE_ACTION_MODE=='greedy',rng=self.rng)
            self.command = intention(action, bend_signs(c.worm_physics))
            self.remaining = DECISION_FRAMES
        self.remaining -= dt
        return self.command
