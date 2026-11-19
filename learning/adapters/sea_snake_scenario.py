"""One sea snake hunting a target: an open arena with a still target, then a moving one, then walls in the way.

Every layout length is in the snake's own radius (SNAKE_RADII below), so the lessons keep their shape if the snake is
drawn bigger or smaller. A trained snake ships when it reaches the target and swims like a snake doing it: a travelling
wave of bends, graded by learning.sea_snake_wriggle. The scripted controller (creatures.sea_snake_control), steering
itself every frame as in the game, is evaluated on the same held-out cases for information and to calibrate that grade.
"""
from dataclasses import asdict
import hashlib
import math
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import numpy as np

from learning.environment import Snapshot, Transition
from learning.adapters.sea_snake import (SPEC, CRITIC_SIZE, DECISION_FRAMES, INFERENCE_ACTION_MODE, observe, intention,
                                         action_for, enabled, load_policy)  # noqa: F401 (the adapter's interface)
from learning import sea_snake_rewards, sea_snake_wriggle as wriggle
from learning.sea_snake_rewards import TABLE

INITIAL_STAGE = 1
USES_OPPONENT = False
EVALUATION_MODES = ('trained', 'scripted')
EVALUATION_ACTION_MODE = INFERENCE_ACTION_MODE
IMITATION_HEADS = tuple(n for n, _ in SPEC.heads)
IMITATION_EPOCHS = 4
STAGES = ('open', 'moving', 'maze')
CURRICULUM = (1, 2, 3)
SUCCESS_GATES = {1: .9, 2: .8, 3: .7}
PASSES_REQUIRED = 2          # held-out validation blocks in a row before the next lesson
# The scripted controller's mean wriggle on the validation cases is .944 open, .950 moving, .830 maze (a tail-only flap
# scores about .2-.3); the gate is about 80% of the lowest, so a trained snake must wriggle nearly as well as the
# scripted travelling wave, with its whole body.
WRIGGLE_GATE = .65
CENTRE = (676., 439.)        # the middle of the 1352x878 game screen
ARENA_HALF = 13.             # the arena is a square box this many snake radii from its centre to each wall
SNAKE_MARGIN = 3.5           # the snake's centre starts at least this far from a wall (it is 5.6 radii long)
TARGET_MARGIN = 1.5
OPEN_DISTANCE = (5., 10.)    # open lessons: the target starts this far from the snake's centre
TARGET_SPEED = .03           # moving lesson: radii per frame (the scripted snake swims about .1)
TARGET_TURN = .08            # radians of random heading drift per frame
BAFFLE_GAP = (3.5, 5.)       # maze: each wall across the arena leaves one gap this wide
TRAIN_LEVELS = (0, 1, 2, 3)
VALIDATION_LEVELS = (4, 5, 6, 7)
HOLDOUT_LEVELS = (8, 9, 10, 11)
# A level turns the whole layout by a quarter turn and may mirror it; the square box stays on screen either way.
ARENAS = {str(i): dict(angle=90.*(i % 4), mirrored=bool(i//4 % 2), half_radii=ARENA_HALF) for i in range(12)}


def snake_radius():
    import enemy_kinds
    return float(enemy_kinds.kind('sea_snake').tuning['size'])


class Target:
    """What the snake hunts: a ship-sized point the Encounter treats as the player's ship. Never shielded."""
    observer = False

    def __init__(self, pos, radius):
        self.x, self.y = pos
        self.radius = radius
        self.active = True

    def get_pos(self):
        return (self.x, self.y)

    def is_shield_active(self):
        return False


class Stings:
    """Stands in for the game's scoring: a sting is the snake's 'enemy collision' with the ship."""
    def __init__(self):
        self.count = 0

    def record_enemy_collision(self):
        self.count += 1


class SeaSnakeEnvironment:
    spec = SPEC
    adapter = sys.modules[__name__]

    def __init__(self, horizon=20., arenas=None, scripted=False, rewards=None, opponent=None, muted=False):
        if not math.isfinite(horizon) or horizon <= 0:
            raise ValueError('sea snake horizon must be finite and positive')
        self.horizon, self.arenas = horizon, arenas or ARENAS
        self.scripted = scripted
        self.rewards = sea_snake_rewards.validate(rewards or {})
        self.world = None
        self.render_frames = None

    def layout(self, rng, stage):
        """(snake centre, snake heading, target, baffles, gaps) in the arena's own radii: along/out coordinates.

        Each gap is the middle of the opening in one baffle, in the order the snake meets them."""
        heading = rng.uniform(-math.pi, math.pi)
        inner = ARENA_HALF-SNAKE_MARGIN
        if stage in (1, 2):
            while True:
                snake = (rng.uniform(-inner, inner), rng.uniform(-inner, inner))
                bearing, distance = rng.uniform(-math.pi, math.pi), rng.uniform(*OPEN_DISTANCE)
                target = (snake[0]+math.cos(bearing)*distance, snake[1]+math.sin(bearing)*distance)
                if max(abs(v) for v in target) <= ARENA_HALF-TARGET_MARGIN:
                    return snake, heading, target, [], []
        snake = (rng.uniform(-ARENA_HALF+SNAKE_MARGIN, -ARENA_HALF+SNAKE_MARGIN+2.), rng.uniform(-inner, inner))
        target = (rng.uniform(ARENA_HALF-4., ARENA_HALF-2.), rng.uniform(-inner, inner))
        alongs = [rng.uniform(-1., 1.)] if rng.random() < .5 else [rng.uniform(-3.5, -2.5), rng.uniform(2.5, 3.5)]
        baffles, gaps = [], []
        for along in alongs:
            # Where the straight line to the target would cross: the gap is never there, so the wall is in the way.
            crossing = snake[1]+(target[1]-snake[1])*(along-snake[0])/(target[0]-snake[0])
            gap = rng.uniform(*BAFFLE_GAP)
            while True:
                middle = rng.uniform(-ARENA_HALF+gap/2+1., ARENA_HALF-gap/2-1.)
                if abs(middle-crossing) > gap/2+1.:
                    break
            baffles += [((along, -ARENA_HALF), (along, middle-gap/2)), ((along, middle+gap/2), (along, ARENA_HALF))]
            gaps.append((along, middle))
        return snake, heading, target, baffles, gaps

    def reset(self, seed, level=0, stage=1):
        import creatures
        from creatures.encounter import Encounter
        from creatures.sea_snake_control import SeaSnakeController
        from game_handlers.enemy_roster import EnemyRoster
        from maze.wall_segment import WallSegment
        import config
        self.close()
        if stage not in CURRICULUM:
            raise ValueError('sea snake stage must be 1..3')
        self.seed, self.level, self.stage = seed, level, stage
        self.rng = rng = random.Random(seed)
        arena = self.arenas[str(level)]
        angle = math.radians(arena['angle'])
        flip = -1. if arena['mirrored'] else 1.
        self.forward, self.normal = (math.cos(angle), math.sin(angle)), (-math.sin(angle)*flip, math.cos(angle)*flip)
        r = self.unit = snake_radius()

        def point(along, out):
            return (CENTRE[0]+(self.forward[0]*along+self.normal[0]*out)*r,
                    CENTRE[1]+(self.forward[1]*along+self.normal[1]*out)*r)
        self.point = point
        snake, heading, target, baffles, self.gaps = self.layout(rng, stage)
        h = arena['half_radii']
        corners = [(-h, -h), (h, -h), (h, h), (-h, h)]
        outlines = list(zip(corners, corners[1:]+corners[:1]))+baffles
        walls = [WallSegment(point(*a), point(*b), 1, destructible=False) for a, b in outlines]
        maze = SimpleNamespace(walls=walls, gate_segments=[], spatial_grid=None)
        self.stings = Stings()
        self.target = Target(point(*target), float(config.SHIP_SIZE))
        self.world = SimpleNamespace(ship=self.target, hunter=None, maze=maze, roster=EnemyRoster(), scoring=self.stings,
            sea_snake_policy=None, additional_targets=[], projectiles=[], powerup_crystals=[],
            sound_manager=SimpleNamespace(play_enemy_destroy=lambda *a: None),
            collision_handler=SimpleNamespace(drop_pickup=lambda *a: None, record_kill=lambda *a: None),
            jev_beacon_allowed=lambda: False, command_recorder=None)
        world_heading = math.atan2(self.forward[1]*math.cos(heading)+self.normal[1]*math.sin(heading),
                                   self.forward[0]*math.cos(heading)+self.normal[0]*math.sin(heading))
        self.snake = c = creatures.build('sea_snake', point(*snake), angle=math.degrees(world_heading))
        c.agent_id = 0
        self.world.roster.add(c)
        self.enc = Encounter(self.world.roster, self.world)
        self.enc.ready()
        self.target_heading = rng.uniform(-math.pi, math.pi)
        self.teacher = SeaSnakeController()
        self.frames = self.stung = self.resets = self.wall_frames = 0
        self.effort_sum = self.return_total = 0.
        self.success = self.finished = False
        self.wriggle = wriggle.Recorder()
        self.previous_rates = None
        self.action_trace = []
        return self.snapshot()

    def snapshot(self):
        x, masks, _ = observe(self.snake, self.enc)
        live = self.snake.active
        return Snapshot({0: x} if live else {}, {0: masks} if live else {}, x, {0: x})

    def local(self, point):
        """A screen point in the arena's own radii: (along, out)."""
        dx, dy = point[0]-CENTRE[0], point[1]-CENTRE[1]
        return ((dx*self.forward[0]+dy*self.forward[1])/self.unit, (dx*self.normal[0]+dy*self.normal[1])/self.unit)

    def path_distance(self, point):
        """Radii a head at this point still has to swim to the edge of the target: straight in the open, and through
        the middle of every baffle gap it has not yet passed in the maze, so a head pressed against a baffle on the
        straight line is no closer than one heading for the gap. Progress pays its change, so it stays potential-based."""
        here, goal = self.local(point), self.local(self.target.get_pos())
        total = 0.
        for gap in self.gaps:           # the target lies beyond every baffle (larger along)
            if here[0] < gap[0]:
                total += math.dist(here, gap)
                here = gap
        total += math.dist(here, goal)
        return max(0., total-self.target.radius/self.unit)

    def distance(self):
        """Radii from the head to the edge of the target, along the way through any baffle gaps."""
        return self.path_distance(self.snake.snake_physics.head)

    def move_target(self):
        """Moving lesson: the target wanders at a steady pace and turns back off the arena's walls."""
        self.target_heading += self.rng.gauss(0., TARGET_TURN)
        step = TARGET_SPEED*self.unit
        x, y = self.target.x+math.cos(self.target_heading)*step, self.target.y+math.sin(self.target_heading)*step
        local = ((x-CENTRE[0])*self.forward[0]+(y-CENTRE[1])*self.forward[1],
                 (x-CENTRE[0])*self.normal[0]+(y-CENTRE[1])*self.normal[1])
        if max(abs(v) for v in local)/self.unit > ARENA_HALF-TARGET_MARGIN:
            # Head back toward the middle of the arena instead of leaving it.
            self.target_heading = math.atan2(CENTRE[1]-self.target.y, CENTRE[0]-self.target.x)
            return
        self.target.x, self.target.y = x, y

    def step(self, actions):
        if self.finished:
            raise RuntimeError('step after sea snake episode ended')
        c = self.snake
        if self.scripted:
            # The baseline: the game's own scripted controller, steering every frame as it does in play.
            action = (0.,)*len(SPEC.heads)
            c.snake_command = None
        else:
            if set(actions) != {0}:
                raise ValueError('sea snake episode expects an action for agent 0')
            action = tuple(actions[0])
            c.snake_command = c.snake_last_command = intention(action)
        self.action_trace.append({0: tuple(float(v) for v in action)})
        # The counts each reward term pays on (learning.sea_snake_rewards); paid together once the decision ends.
        events = TABLE.row()
        count = min(DECISION_FRAMES, max(0, math.ceil(self.horizon*60)-self.frames))
        for _ in range(count):
            before, resets_before = self.distance(), c.snake_resets
            if self.stage == 2:
                self.move_target()
            self.enc.step(1.)
            self.frames += 1
            resets = c.snake_resets-resets_before
            self.resets += resets
            events['resets'] += resets
            events['seconds'] += 1/60.
            if not c.active:
                break
            p = c.snake_physics
            after = self.distance()
            events['progress_radii'] += before-after
            angles, rates, extent, forward = (wriggle.bends(p), wriggle.bend_rates(p), wriggle.extension(p),
                                              wriggle.forward_speed(p))
            # A rebuilt body starts from rest: its first frame is not a twitch.
            previous = rates if resets or self.previous_rates is None else self.previous_rates
            for key, value in wriggle.frame_terms(angles, rates, previous, extent, forward, after < before).items():
                events[key] += value
            self.previous_rates = rates
            self.wriggle.add(angles, extent, forward)
            effort = sum(abs(a) for a in p.activation)/len(p.activation)
            self.effort_sum += effort
            events['effort_seconds'] += effort/60.
            pressed = any(p.surface_contacts)
            self.wall_frames += pressed
            events['wall_seconds'] += pressed/60.
            if self.stings.count > self.stung:
                self.stung = self.stings.count
                events['stings'] += 1
                self.success = True
            if self.render_frames is not None:
                self.render_frames(self)
            if self.success:
                break
        reward = float(TABLE.pay([events], self.rewards)[0])
        self.return_total += reward
        terminated = self.success or not c.active
        truncated = not terminated and self.frames >= math.ceil(self.horizon*60)
        self.finished = terminated or truncated
        return Transition(self.snapshot(), reward, terminated, truncated, self.events(), {0: reward})

    def events(self):
        horizon_frames = math.ceil(self.horizon*60)
        # How snake-like the whole episode was; graded once it is over (learning.sea_snake_wriggle.score).
        graded = {'wriggle' if k == 'wriggle' else 'wriggle_'+k: v
                  for k, v in self.wriggle.score().items()} if self.finished else {}
        return dict(seconds=self.frames/60., success=float(self.success), stings=self.stung,
                    distance=self.distance() if self.snake.active else 0., effort=self.effort_sum/max(1, self.frames),
                    wall_fraction=self.wall_frames/max(1, self.frames), resets=self.resets,
                    survived=float(self.snake.active), return_total=self.return_total, stage=self.stage,
                    # 1 for a sting at once, falling to 0 for one at the horizon or none.
                    sting_speed=float(self.success)*(1.-self.frames/horizon_frames), **graded)

    def close(self):
        if self.world is not None:
            self.enc.close()
            self.world = None


def teacher_action(env, agent, action):
    """The scripted controller's command for this decision, as the action that produces it (bootstrap imitation)."""
    target = env.target.get_pos() if env.target.active else None
    return action_for(env.teacher.command(env.snake.snake_physics, DECISION_FRAMES, target))


def environment(**options):
    return SeaSnakeEnvironment(**options)


def environment_options(manifest):
    return dict(rewards=sea_snake_rewards.validate(manifest.get('rewards', {})))


def metadata(args):
    import enemy_kinds
    import pymunk
    root = Path(__file__).resolve().parents[2]
    files = ('physics/sea_snake.py', 'creatures/sea_snake_control.py', 'creatures/traits/sea_snake.py',
             'learning/adapters/sea_snake.py', 'learning/adapters/sea_snake_scenario.py',
             'learning/sea_snake_wriggle.py')
    return dict(task=SPEC.task, scenario_version=4, spec=SPEC.metadata(), horizon=args.horizon,
                progress_rule='head_path_through_baffle_gap_middles',
                wriggle=dict(score='geometric_mean_of_curvature_wave_rhythm_swim_times_spread', gate=WRIGGLE_GATE,
                             spread_share=wriggle.SPREAD_SHARE,
                             shaping='wave_and_curvature_paid_only_while_swimming_forward_and_closing'),
                deployment=dict(success=DEPLOY_SUCCESS, sample_tolerance=SAMPLE_TOLERANCE),
                lessons=dict(zip(CURRICULUM, STAGES)), curriculum_order=CURRICULUM, success_gates=SUCCESS_GATES,
                passes_required=PASSES_REQUIRED, success='first_sting_of_the_target', units='snake_radii',
                layout=dict(arena_half=ARENA_HALF, snake_margin=SNAKE_MARGIN, target_margin=TARGET_MARGIN,
                            open_distance=OPEN_DISTANCE, target_speed=TARGET_SPEED, target_turn=TARGET_TURN,
                            baffle_gap=BAFFLE_GAP),
                action_selection=dict(training='sample', validation=EVALUATION_ACTION_MODE,
                                      gameplay=INFERENCE_ACTION_MODE),
                reference='scripted_controller_every_frame_for_information',
                decision_frames=DECISION_FRAMES, opponent_sha256=None, arenas=ARENAS,
                creature=asdict(enemy_kinds.kind('sea_snake')), rewards=sea_snake_rewards.from_args(args),
                pymunk=pymunk.version, physics_fps=60,
                implementation_sha256={p: hashlib.sha256((root/p).read_bytes()).hexdigest() for p in files})


def training_cases(iteration, count, stage, bootstrap=False, curriculum=None):
    current = curriculum['stage'] if curriculum else stage
    earlier = CURRICULUM[:CURRICULUM.index(current)]

    def practice_stage(j):
        """A quarter of every batch rotates through the earlier lessons; the rest is the current one."""
        return earlier[(iteration+j//4) % len(earlier)] if earlier and j % 4 == 0 else current
    return [dict(level=TRAIN_LEVELS[j % 4], seed=10000+iteration*count+j+(500000000 if bootstrap else 0),
                 stage=CURRICULUM[j % len(CURRICULUM)] if bootstrap else practice_stage(j)) for j in range(count)]


def validation_cases():
    return [dict(level=level, seed=2000000000+stage*100+j*4+k, stage=stage)
            for stage in CURRICULUM for j, level in enumerate(VALIDATION_LEVELS) for k in range(3)]


def evaluation_cases():
    return [dict(level=level, seed=3000000000+stage*100+j*4+k, stage=stage)
            for stage in CURRICULUM for j, level in enumerate(HOLDOUT_LEVELS) for k in range(3)]


def metrics(rows):
    return {key: float(np.mean([r.get(key, 0.) for r in rows])) if rows else 0.
            for key in ('success', 'sting_speed', 'distance', 'seconds', 'effort', 'wall_fraction', 'resets',
                        'survived', 'return_total', 'wriggle', *('wriggle_'+c for c in wriggle.COMPONENTS))}


def selection_score(summary):
    return (summary['success']*10+summary['wriggle']*5+summary['sting_speed']*2-summary['distance']*.1
            -summary['resets']*5)


def _paired(a, b, key):
    if len(a['episodes']) != len(b['episodes']):
        raise ValueError('sea snake comparison requires paired cases')
    values = []
    for x, y in zip(a['episodes'], b['episodes']):
        if any(x[k] != y[k] for k in ('level', 'seed', 'stage')):
            raise ValueError('sea snake comparison cases do not match')
        values.append(x[key]-y[key])
    if not values:
        return dict(mean=0., lower_95=-1., upper_95=1.)
    samples = np.random.default_rng(0).choice(values, (2000, len(values))).mean(axis=1)
    return dict(mean=float(np.mean(values)), lower_95=float(np.quantile(samples, .025)),
                upper_95=float(np.quantile(samples, .975)))


def compare(a, b):
    """Paired difference in success (trained minus baseline) over the same cases."""
    return _paired(a, b, 'success')


def promote(stage, summary):
    return stage


def initial_curriculum(args):
    return dict(stage=1, passing_blocks=0)


def validation_evidence(summary, state, args, deadline=math.inf):
    # A lesson is passed only by reaching the target while swimming like a snake.
    passed = (summary['success'] >= SUCCESS_GATES[state['stage']] and summary.get('survived', 1.) == 1.
              and summary.get('wriggle', 0.) >= WRIGGLE_GATE)
    return dict(sea_snake_validation_passed=passed)


def advance_curriculum(state, summary):
    state = dict(state)
    if 'sea_snake_validation_passed' not in summary:
        return state
    state['passing_blocks'] = state['passing_blocks']+1 if summary['sea_snake_validation_passed'] else 0
    index = CURRICULUM.index(state['stage'])
    if state['passing_blocks'] >= PASSES_REQUIRED and index+1 < len(CURRICULUM):
        state.update(stage=CURRICULUM[index+1], passing_blocks=0)
    return state


def curriculum_key(state):
    return STAGES[state['stage']-1]


def curriculum_validation_cases(state):
    return [c for c in validation_cases() if c['stage'] == state['stage']]


def curriculum_evaluation_cases(state):
    return evaluation_cases()


def curriculum_selection_score(summary, state):
    return selection_score(summary)


# Deployment: it must reach the target often enough in every lesson, and swim like a snake doing it. It need not beat
# the scripted controller at anything; the scripted numbers are reported for information only.
DEPLOY_SUCCESS = {1: .8, 2: .8, 3: .5}
SAMPLE_TOLERANCE = 1/12        # one episode of a lesson's 12 held-out cases


def lesson_report(rows):
    summary = metrics(rows)
    return {k: summary[k] for k in ('success', 'wriggle', *('wriggle_'+c for c in wriggle.COMPONENTS), 'seconds')}


def deployment_report(reports, args, deadline=math.inf):
    """Ship only a policy that, in every lesson on the held-out cases, reaches the target at least DEPLOY_SUCCESS of
    the time (less one episode's worth for the small sample) and wriggles at least WRIGGLE_GATE on average."""
    trained, scripted = reports['trained'], reports['scripted']
    rows = trained['episodes']
    per_lesson, scripted_lessons = {}, {}
    for s in CURRICULUM:
        name = STAGES[s-1]
        report = lesson_report([r for r in rows if r['stage'] == s])
        report.update(reaches=bool(report['success'] >= DEPLOY_SUCCESS[s]-SAMPLE_TOLERANCE),
                      wriggles=bool(report['wriggle'] >= WRIGGLE_GATE), cases=sum(r['stage'] == s for r in rows))
        per_lesson[name] = report
        scripted_lessons[name] = lesson_report([r for r in scripted['episodes'] if r['stage'] == s])
    covered = all(d['cases'] for d in per_lesson.values())
    reaches = covered and all(d['reaches'] for d in per_lesson.values())
    wriggles = covered and all(d['wriggles'] for d in per_lesson.values())
    return dict(passed=bool(reaches and wriggles), reaches_target=bool(reaches), wriggles_like_a_snake=bool(wriggles),
                success_gates=DEPLOY_SUCCESS, sample_tolerance=SAMPLE_TOLERANCE, wriggle_gate=WRIGGLE_GATE,
                per_lesson=per_lesson, scripted_for_information=scripted_lessons,
                trained=metrics(rows), scripted=metrics(scripted['episodes']), automatic_install=False)
