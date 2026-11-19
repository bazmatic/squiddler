"""Versioned body-relative senses and seven motor outputs for one sea snake.

Every length is expressed in the snake's own radius and every speed in lengths per frame, so a snake whose body
proportions change still reads the world the same way.
"""
import math
import zipfile
import numpy as np

from creatures.traits.rooted import wall_between
from learning.policy import PolicySpec, RecurrentPolicy
from physics.sea_snake import MUSCLES, SeaSnakeCommand
from utils import get_closest_point_on_line
from weapons.geometry import segment_contact

DECISION_FRAMES = 6
# Gameplay and evaluation must execute the same learned action distribution.
INFERENCE_ACTION_MODE = 'sample'
MAX_RATE = .1           # radians per frame of relative joint rotation counted as full speed
SIGHT_RADII = 10.       # how far it sees, in snake radii
MOTOR_NAMES = tuple(f'bend{i}' for i in range(MUSCLES))
HEADS = tuple((n, 2) for n in MOTOR_NAMES)
FEATURES = tuple(f'{group}{i}' for group in ('angle', 'rate', 'activation') for i in range(MUSCLES))
FEATURES += ('velocity_forward', 'velocity_side') + tuple(f'contact{i}' for i in range(MUSCLES+1))
FEATURES += tuple('previous_'+n for n, _ in HEADS)
FEATURES += tuple(f'{end}_ray{i}' for end in ('tail', 'head') for i in range(8))
FEATURES += ('surface_distance', 'surface_forward', 'surface_side')
FEATURES += ('target_present', 'target_sin', 'target_cos', 'target_distance', 'root_pinned')
SPEC = PolicySpec('sea_snake', FEATURES, HEADS, version=1, continuous=MOTOR_NAMES)
CRITIC_SIZE = len(FEATURES)


def visible_target(c, enc, sight, scent=False):
    """The prey it can sense. `scent` (the game, not training) lets it sense prey beyond sight range, by bearing only."""
    target = enc.target_entity_for(c)
    if (target is None or not target.active or (math.dist(c.get_pos(), target.get_pos()) > sight and not scent)
            or wall_between(c.snake_physics.head, target.get_pos(), enc.maze)):
        return None
    return target


def observe(c, enc, scent=False):
    p = c.snake_physics
    o = p.observe()
    sight = SIGHT_RADII*p.radius
    f = p.heading
    cf, sf = math.cos(f), math.sin(f)
    def local(dx, dy):
        return (dx*cf+dy*sf, -dx*sf+dy*cf)
    x = [*o['joint_angles'], *(v/MAX_RATE for v in o['joint_rates']), *o['activation'],
         *o['velocity'], *(float(v) for v in o['contacts'])]
    # snake_last_command is only updated when the driver decides (stale while an external command overrides).
    previous = getattr(c, 'snake_last_command', None) or SeaSnakeCommand()
    x += previous.muscles
    grid = getattr(enc.maze, 'spatial_grid', None)
    walls = list(grid.get_nearby_walls(p.centre, sight+p.length*4) if grid else enc.maze.walls)
    walls += list(getattr(enc.maze, 'gate_segments', ()))
    walls = [w for w in walls if w.active and w.start != w.end]
    for origin in (p.tail, p.head):
        for i in range(8):
            angle = f+math.tau*i/8
            delta = (math.cos(angle)*sight, math.sin(angle)*sight)
            hits = [hit[0] for w in walls if (hit := segment_contact(origin, delta, w.start, w.end, .1)) is not None]
            x.append(min(hits, default=1.))
    centre = p.centre
    foot = min((get_closest_point_on_line(centre, w.start, w.end) for w in walls),
               key=lambda v: math.dist(centre, v), default=None)
    if foot is None:
        x += [1., 0., 0.]
    else:
        dx, dy = centre[0]-foot[0], centre[1]-foot[1]
        distance = max(1e-8, math.hypot(dx, dy))
        x += [min(1., distance/sight), *(v/distance for v in local(dx, dy))]
    target = visible_target(c, enc, sight, scent)
    if target is None:
        x += [0.]*4
    else:
        dx, dy = target.x-centre[0], target.y-centre[1]
        forward, side = local(dx, dy)
        bearing = math.atan2(side, forward)
        x += [1., math.sin(bearing), math.cos(bearing), min(1., math.hypot(dx, dy)/sight)]
    x.append(0.)      # the slot the trained brain was built with for a pinned root, which no snake has now
    masks = {name: np.ones(size, bool) for name, size in HEADS}
    return np.clip(np.asarray(x, np.float32), -5., 5.), masks, ()


def intention(action, signs=None):
    if len(action) != len(HEADS) or not np.isfinite(action).all():
        raise ValueError('sea snake action must contain seven finite values')
    return SeaSnakeCommand(tuple(math.tanh(float(v)) for v in action))


def action_for(command):
    return tuple(float(np.arctanh(np.clip(v, -.999, .999))) for v in command.muscles)


def enabled(x, action, masks):
    return [bool(masks[n].any()) for n, _ in HEADS]


def load_policy(path):
    try:
        policy = RecurrentPolicy.load(path, SPEC)
        if not np.isfinite(policy.mean).all() or not np.isfinite(policy.scale).all():
            raise ValueError('nonfinite sea snake normalization')
        return policy
    except zipfile.BadZipFile as exc:
        raise ValueError(f'damaged sea snake checkpoint: {exc}') from exc


class SeaSnakeDriver:
    """Independent memory and sampling stream; same action distribution as training."""
    def __init__(self, policy, seed=None):
        self.policy = policy
        self.rng = np.random.default_rng(None if seed is None else np.random.SeedSequence([seed, 0]))
        self.state = policy.initial_state()
        self.remaining = 0.
        self.command = SeaSnakeCommand()

    def advance(self, c, enc, dt):
        if self.remaining <= 1e-8:
            x, masks, _ = observe(c, enc, scent=True)
            action, self.state = self.policy.choose(x, self.state, masks,
                                                   greedy=INFERENCE_ACTION_MODE == 'greedy', rng=self.rng)
            self.command = intention(action)
            # Only updated when the driver decides (stale while an external command overrides).
            c.snake_last_command = self.command
            self.remaining = DECISION_FRAMES
        self.remaining -= dt
        return self.command
