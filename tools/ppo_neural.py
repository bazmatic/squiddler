"""Recurrent PPO fine-tuning of the neural student: python -m tools.ppo_neural (the loop is in Task 3)."""
import argparse
import functools
import os
import sys
import time
from pathlib import Path
import numpy as np
from hunter.neural.policy import HEAD_SIZES, InvalidPolicy, NeuralPolicy
from hunter.senses import SENSE_SIZE, OBSERVATION_VERSION
from tools.train_neural import _load_into, to_numpy_policy

HEADS = ('turn', 'thrust', 'fire')
REWARD_SCALE = 0.1            # keeps returns near 1 so one learning rate suits the policy and value heads


def _net_class(features=SENSE_SIZE):
    import torch
    from learning.models import network_class
    base = network_class(features, HEAD_SIZES)

    class PolicyValueNet(base):
        def __init__(self, hidden, embed):
            super().__init__(hidden, embed)
            self.value = torch.nn.Linear(hidden, 1)
            torch.nn.init.zeros_(self.value.weight)
            torch.nn.init.zeros_(self.value.bias)

        def forward(self, x, state=None):
            h, _ = self.gru(torch.tanh(self.embed(x)), state)
            return {k: head(h) for k, head in self.heads.items()}, self.value(h).squeeze(-1)
    return PolicyValueNet


def build_net(policy):
    """A torch net holding `policy`'s weights, plus a zeroed value head (training only)."""
    net = _net_class(policy.params['embed_w'].shape[1])(policy.hidden, policy.params['embed_w'].shape[0])
    _load_into(net, policy)
    return net


from learning.ppo import pack, joint_logp, gae, _kl_and_entropy, improve


def export(net, policy, path):
    """Save the net as a game policy file (the value head is left behind)."""
    exported = to_numpy_policy(net, policy.mean, policy.scale)
    exported.feature_names = policy.feature_names
    exported.save(path)


ROOT = Path(__file__).resolve().parents[1]
PPO_TASKS = ('hunter', 'campaign_hunter')     # the tasks this loop trains; the rest train in learning.run
DEFAULT_START = ROOT / 'assets' / 'neural' / 'hunter.npz'
DEFAULT_OUT = ROOT / '.astrovoid-local' / 'neural' / 'rl'
TRAIN_SEEDS = 100             # seeds 300-399
DEFAULT_HORIZON = 90.0        # seconds of game time per training episode unless the shaping says otherwise


def make_jobs(iteration, count, weights, decision_frames, turn_rate, levels=None, task='hunter'):
    """One job per episode, cycling evenly through the selected levels (default: the tune set). Another task's jobs
    name it after the student."""
    from experiments.run import TUNE_LEVELS
    from experiments.student import StudentSpec
    levels = tuple(levels or TUNE_LEVELS)
    jobs = []
    for i in range(count):
        n = iteration * count + i
        spec = StudentSpec(weights=str(weights), decision_frames=decision_frames, turn_rate=turn_rate,
                           greedy=False, sample_seed=n)
        jobs.append((levels[n % len(levels)], 300 + n % TRAIN_SEEDS, spec) + (() if task == 'hunter' else (task,)))
    return jobs


def kl_coefficient(iteration, iterations, start=0.1, end=0.02, fade=0.5):
    span = max(fade * iterations, 1.0)
    return end if iteration >= span else start + (end - start) * iteration / span


def _rollout_job(job, shaping=None):
    import time
    from learning.tasks import task
    level, seed, spec, *named = job                   # a hunter job does not name its task
    started = time.perf_counter()
    flown = task(named[0] if named else 'hunter').adapter().environment().run(level, seed, spec, **(shaping or {}))
    flown['wall'] = time.perf_counter() - started
    return flown


def _indexed_job(fly, item):
    index, job = item
    return index, fly(job)


def longest_first(jobs, estimates):
    """Submission order for the pool: levels not yet timed first, then the slowest levels, so no long flight starts last."""
    return sorted(range(len(jobs)), key=lambda i: (jobs[i][0] in estimates, -estimates.get(jobs[i][0], 0.0), i))


class PoolRollout:
    """Flies jobs on a pool of worker processes that stays up between iterations.

    Results come back in job order whatever order the workers finish in, so the training batch does not depend on
    scheduling. Jobs go out slowest level first, judged by the previous iteration's wall times.
    """

    def __init__(self, workers, shaping=None, job=None):
        self.fly = job or functools.partial(_rollout_job, shaping=shaping)
        self.estimates = {}
        self._pool = None
        if workers > 1:
            import multiprocessing
            from experiments import run
            self._pool = multiprocessing.get_context('spawn').Pool(workers, initializer=run._use_dummy_drivers)

    def __call__(self, jobs, workers=None):
        order = longest_first(jobs, self.estimates)
        items = [(i, jobs[i]) for i in order]
        if self._pool is None:
            pairs = [_indexed_job(self.fly, item) for item in items]
        else:
            pairs = list(self._pool.imap_unordered(functools.partial(_indexed_job, self.fly), items, chunksize=1))
        results = [None] * len(jobs)
        for index, flown in pairs:
            results[index] = flown
        times = {}
        for job, flown in zip(jobs, results):
            times.setdefault(job[0], []).append(flown.get('wall', 0.0))
        self.estimates = {level: sum(t) / len(t) for level, t in times.items()}
        return results

    def close(self):
        if self._pool is not None:
            self._pool.close()
            self._pool.join()
            self._pool = None


def log_line(*parts):
    print(*parts, flush=True)         # a log redirected to a file is otherwise block-buffered


def train_rl(start, out_dir, iterations, episodes=96, workers=12, lr=1e-4, seed=0, decision_frames=6,
             turn_rate=1.0, rollout=None, log=log_line, shaping=None,
             levels=None, resume=False, gamma=0.99, gae_lambda=0.95, task='hunter', horizon=None):
    """PPO on the lab's own score, with optional reward shaping. No gate: every iteration saves a checkpoint.

    `task` is the hunter's or another task flown by the same student, such as 'campaign_hunter': its adapter flies the
    episodes, its senses layout is the policy's (a start policy in the hunter's is migrated to it), and its levels are
    the default. `horizon` overrides the shaping's.

    Each iteration also saves the resume state, and a PAUSE file in `out_dir` stops the run once the iteration is
    saved. With `resume`, the run in `out_dir` carries on from its resume state as if it had never stopped; `start`
    is still the run's original start, which the policy is held near."""
    import torch
    from experiments.score import mean_components
    from learning.tasks import task as task_named
    adapter = task_named(task).adapter()
    policy = start_policy(start, adapter.FEATURE_NAMES)   # raises InvalidPolicy: never train from nothing
    if task != 'hunter':
        levels = levels or adapter.default_levels()
    if horizon is not None:
        shaping = dict(shaping or {}, horizon=horizon)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    out_dir = Path(out_dir)
    saved = torch.load(out_dir / RESUME_FILE, weights_only=False) if resume else None   # absent: FileNotFoundError
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / STATE_FILE).unlink(missing_ok=True)
    net, ref = build_net(policy), build_net(policy)
    for parameter in ref.parameters():
        parameter.requires_grad_(False)
    optimiser = torch.optim.Adam(net.parameters(), lr=lr)
    first = 0
    if saved is not None:
        # Explicit legacy observation adapter, including Adam's moment buffers.
        old_weight = saved['net']['embed.weight']
        width = net.state_dict()['embed.weight'].shape[1]
        if old_weight.shape[1] < width:
            from hunter.neural.policy import older_layouts
            if old_weight.shape[1] not in {len(names) for names in older_layouts(adapter.FEATURE_NAMES)}:
                raise ValueError('unsupported resume observation schema')
            saved['net']['embed.weight'] = torch.nn.functional.pad(old_weight, (0, width-old_weight.shape[1]))
            parameter_ids = [key for group in saved['optimiser']['param_groups'] for key in group['params']]
            embedding_index = list(dict(net.named_parameters())).index('embed.weight')
            item = saved['optimiser']['state'].get(parameter_ids[embedding_index], {})
            for key, value in list(item.items()):
                if hasattr(value, 'shape') and value.shape == old_weight.shape:
                    item[key] = torch.nn.functional.pad(value, (0, width-old_weight.shape[1]))
        net.load_state_dict(saved['net'])
        optimiser.load_state_dict(saved['optimiser'])
        torch.set_rng_state(saved['torch_rng'])
        rng.bit_generator.state = saved['numpy_rng']
        first = saved['next_iteration']
        config_path = out_dir / 'config.json'
        if config_path.exists():
            import json
            run_config = json.loads(config_path.read_text())
            # Combat rules/tuning are immutable within a run. Old runs keep legacy combat.
            prior = run_config.get('shaping', {})
            if shaping is not None and task == 'hunter':
                shaping = dict(shaping)
                shaping['combat_ruleset'] = prior.get('combat_ruleset', 'legacy')
                shaping['curved_walls'] = prior.get('curved_walls', False)
                shaping['weapon_tuning'] = prior.get('weapon_tuning')
            episode = run_config.get('episode', {})
            if episode.get('rules_share') and 'level_cycle_from_iteration' not in episode:
                episode['level_cycle_from_iteration'] = first
            if (shaping or {}).get('random_maze') and not run_config.get('shaping', {}).get('random_maze'):
                run_config.setdefault('episode', {})['random_maze_from_iteration'] = first
                run_config.setdefault('shaping', {})['random_maze'] = True
            config_path.write_text(json.dumps(run_config, indent=1))
        current = out_dir / f'iter{first - 1:04d}.npz'
        log_file = out_dir / 'train.log'
        if log_file.exists() and not log_file.read_bytes().endswith(b'\n'):
            with open(log_file, 'a') as record:          # a line cut short by a crash: start ours on a new line
                record.write('\n')
    else:
        current = out_dir / 'iter_start.npz'
        export(net, policy, current)
    pooled = None if rollout else PoolRollout(workers, shaping)
    rollout = rollout or (lambda jobs, count: pooled(jobs))
    history = []
    paused = False
    try:
        for iteration in range(first, iterations):
            jobs = make_jobs(iteration, episodes, current, decision_frames, turn_rate, levels, task)
            t0 = time.perf_counter()
            flown = rollout(jobs, workers)
            t1 = time.perf_counter()
            stats = improve(net, ref, optimiser, flown, policy.mean, policy.scale,
                            kl_coefficient(iteration, iterations), rng, gamma=gamma, lam=gae_lambda)
            if task == 'hunter':
                summary = mean_components([e['result'] for e in flown])
                stats.update(iteration=iteration, mean_return=float(np.mean([e['rewards'].sum() for e in flown])),
                             score=summary['score'], kills_per_min=summary['kills_per_min'],
                             wall_quotient_per_min=summary['wall_quotient_per_min'],
                             hits_taken_per_min=summary['hits_taken_per_min'],
                             speed_fraction=float(np.mean([e.get('speed_fraction', 0.0) for e in flown])),
                             deaths_per_episode=float(np.mean([1.0 if e.get('died') else 0.0 for e in flown])),
                             powerups_per_episode=float(np.mean([e.get('powerups', 0) for e in flown])),
                             upgraded_fraction=float(np.mean([e.get('upgraded_fraction', 0.0) for e in flown])),
                             rollout_s=t1 - t0, update_s=time.perf_counter() - t1)
                arena = [e for e in flown if e.get('rules_arena')]
                stats.update(rules_episodes=len(arena),
                             rules_beaten=float(np.mean([1.0 if e.get('rules_beaten') else 0.0 for e in arena]))
                             if arena else 0.0)
                blind = sum(e.get('blind_steps', 0) for e in flown)
                stats.update(compass_follow=sum(e.get('following_steps', 0) for e in flown) / blind if blind else 0.0)
                unseen = sum(e.get('unseen_steps', 0) for e in flown)
                steps = sum(len(e.get('temperatures', ())) for e in flown)
                panicked = sum(int((np.asarray(e.get('temperatures', ())) > 1.0).sum()) for e in flown)
                stats.update(panic_fraction=panicked / steps if steps else 0.0)
                stats.update(hits_per_episode=float(np.mean([e.get('hits', 0) for e in flown])))
                stats.update(unaimed_shots_per_episode=float(np.mean([e.get('unaimed_shots', 0) for e in flown])))
                stats.update(bored_fraction=sum(e.get('bored_steps', 0) for e in flown) / unseen if unseen else 0.0)
            else:
                stats.update(iteration=iteration, mean_return=float(np.mean([e['rewards'].sum() for e in flown])),
                             **adapter.summary(flown), rollout_s=t1 - t0, update_s=time.perf_counter() - t1)
            current = out_dir / f'iter{iteration:04d}.npz'
            export(net, policy, current)
            history.append(stats)
            line = ' '.join(f'{k}={v:.3f}' if isinstance(v, float) else f'{k}={v}' for k, v in stats.items())
            log(line)
            save_resume_state(out_dir, net, optimiser, rng, iteration + 1, iterations)
            with open(out_dir / 'train.log', 'a') as record:     # the dashboard reads the newest run's train.log
                record.write(line + '\n')
            if (out_dir / PAUSE_FILE).exists() and iteration < iterations - 1:   # on the last one it just finishes
                (out_dir / PAUSE_FILE).unlink()
                write_state(out_dir, 'paused', iteration)
                paused = True
                break
        else:
            (out_dir / PAUSE_FILE).unlink(missing_ok=True)
            write_state(out_dir, 'finished', iterations - 1)
    finally:
        if pooled is not None:
            pooled.close()
    (out_dir / 'iter_start.npz').unlink(missing_ok=True)
    return {'last': current if history else None, 'history': history, 'paused': paused}


def start_policy(start, feature_names):
    """The policy in `start` in the `feature_names` layout; a hunter policy is carried over to a longer layout, and a
    policy in a known older version of the layout (such as the first campaign layout) loads padded to the current."""
    from hunter.senses import FEATURE_NAMES
    try:
        return NeuralPolicy.load(start, feature_names)
    except InvalidPolicy:
        if feature_names == FEATURE_NAMES:
            raise
        NeuralPolicy.load(start)                       # raises InvalidPolicy unless it is a hunter policy
        from tools.migrate_policy import migrated
        return migrated(start, feature_names)


RESUME_FILE = 'resume.pt'     # everything an iteration leaves behind that the next one needs
STATE_FILE = 'state.json'     # how the run ended: paused or finished
PAUSE_FILE = 'PAUSE'          # present: stop once the current iteration is saved


def save_resume_state(out_dir, net, optimiser, rng, next_iteration, iterations):
    """Save what the run needs to carry on at `next_iteration`, replacing the last iteration's (never half-written)."""
    import torch
    temporary = out_dir / (RESUME_FILE + '.tmp')
    torch.save({'net': net.state_dict(), 'optimiser': optimiser.state_dict(), 'torch_rng': torch.get_rng_state(),
                'numpy_rng': rng.bit_generator.state, 'next_iteration': next_iteration, 'iterations': iterations,
                'observation_version': OBSERVATION_VERSION},
               temporary)
    os.replace(temporary, out_dir / RESUME_FILE)


def write_state(out_dir, state, iteration):
    import json
    (Path(out_dir) / STATE_FILE).write_text(json.dumps({'state': state, 'iteration': iteration}))


def reward_summary(args):
    """The reward terms a run trains on, as plain numbers for the dashboard: (name, sign, value, unit) rows."""
    from experiments.harness import IDLE_AFTER_SECONDS
    from experiments.rollout import KILL_REWARD_TOTAL
    from experiments.score import Weights
    w = Weights()
    rows = [('Kills', '+', KILL_REWARD_TOTAL, 'points shared across all the enemies in an episode'),
            ('Powerup collected', '+', args.powerup_reward, 'points each'),
            ('Gun upgraded', '+', args.upgrade_bonus, 'points per second with a hunter power boost'),
            ('Wall impact', '-', w.quotient * args.wall_scale, 'points per full-speed hit (scales with the speed lost)'),
            ('Enemy hit', '-', w.hit * args.hit_scale, 'points each'),
            ('Idle', '-', w.idle / 60.0, f'points per second beyond {IDLE_AFTER_SECONDS:.0f} s without a kill, until the field is clear'),
            ('Flying below top speed', '-', args.speed_penalty, 'points per second at a standstill, in proportion to the shortfall'),
            ('Crawling under the still speed', '-', args.still_penalty, f'points per second below {args.still_speed:.0%} of top speed'),
            ('Starting or reversing a turn', '-', args.turn_change_penalty, 'points each'),
            ('Reversing a turn directly', '-', args.reversal_penalty, 'extra points each, left to right or back'),
            ('Death', '-', args.death_penalty, f'points when the hunter is destroyed (it survives {args.health} hits)'),
            ('Rules hunter destroyed', '+', args.rules_bounty, 'points, on top of its share of the kill reward'),
            ('Following the enemy compass', '+', getattr(args, 'compass_reward', 0.0),
             'points per maze width the route to the nearest enemy shortens while none is in sight'),
            ('Hit on an enemy', '+', getattr(args, 'hit_reward', 0.0), 'points per shot that reaches an enemy'),
            ('Turning', '-', getattr(args, 'turning_penalty', 0.0), 'points per second spent turning'),
            ('Unaimed shot', '-', getattr(args, 'shot_price', 0.0),
             'points each, unless within 20° of an enemy in sight'),
            ('Boredom', '-', getattr(args, 'boredom_penalty', 0.0),
             f'points per second with no enemy in sight in a cell visited in the last {getattr(args, "boredom_window", 10.0):g} s')]
    return [{'name': n, 'sign': s, 'value': round(v, 4), 'unit': u} for n, s, v, u in rows if v]


def _wall_energy():
    import config
    return config.SETTINGS.game.neuralWallEnergy


def parse_levels(text):
    """'1-20' or '4,7,13' (or a mix) as a tuple of level numbers."""
    levels = []
    for part in text.split(','):
        low, _, high = part.strip().partition('-')
        levels.extend(range(int(low), int(high or low) + 1))
    return tuple(levels)


def shaping_from(args):
    """The episode shaping a run's arguments ask for: what fly_rollout (or the task's episode) is called with."""
    from learning.tasks import task
    table = task(getattr(args, 'task', 'hunter')).reward_table()
    if table is not None:
        return {'horizon': args.horizon, **table.from_args(args)}
    from weapons.catalog import RULESET, CATALOG
    return {'wall_scale': args.wall_scale,
            'hit_scale': args.hit_scale,
            'still_penalty': args.still_penalty,
            'horizon': args.horizon,
            'random_start': not args.fixed_start,
            'still_speed': args.still_speed,
            'turn_change_penalty': args.turn_change_penalty,
            'reversal_penalty': args.reversal_penalty,
            'health': args.health,
            'death_penalty': args.death_penalty,
            'speed_penalty': args.speed_penalty,
            'powerup_reward': args.powerup_reward,
            'after_clear': args.after_clear,
            'upgrade_bonus': args.upgrade_bonus,
            'pursue_hunter': not args.decoy_player,
            'rules_bounty': args.rules_bounty,
            'compass_reward': args.compass_reward,
            'boredom_penalty': args.boredom_penalty,
            'boredom_window': args.boredom_window,
            'shot_price': args.shot_price,
            'turning_penalty': args.turning_penalty,
            'hit_reward': args.hit_reward,
            'panic_score': args.panic_score,
            'panic_temperature': args.panic_temperature,
            'duel_separation': True, 'random_maze': True,
            'combat_ruleset': RULESET,
            'curved_walls': True,
            'weapon_tuning': CATALOG.snapshot()}


def table_reward_summary(args):
    """The reward terms of a task that declares them in a reward table, as reward_summary gives the hunter's."""
    from learning.tasks import task
    table = task(args.task).reward_table()
    return table.summary(table.from_args(args))


def write_run_config(args):
    """Record what this run trains on beside its log, so the dashboard can list it."""
    import json
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.task == 'campaign_hunter':
        (out / 'config.json').write_text(json.dumps({
            'task': args.task, 'rewards': table_reward_summary(args),
            'episode': {'horizon_seconds': args.horizon, 'episodes_per_iteration': args.episodes,
                        'learning_rate': args.lr, 'levels': list(args.levels)},
            'start': str(args.start), 'shaping': shaping_from(args),
            'observation_version': OBSERVATION_VERSION, 'decision_frames': 6, 'turn_rate': 1.0,
            'args': {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}}, indent=1))
        return
    (out / 'config.json').write_text(json.dumps({
        'rewards': reward_summary(args),
        'episode': {'horizon_seconds': args.horizon, 'after_clear_seconds': args.after_clear,
                    'episodes_per_iteration': args.episodes, 'learning_rate': args.lr,
                    'health': args.health, 'wall_energy': _wall_energy() if args.health else 0,
                    'levels': args.levels,
                    'panic_score': args.panic_score, 'panic_temperature': args.panic_temperature},
        'start': str(args.start), 'shaping': shaping_from(args),
        'observation_version': OBSERVATION_VERSION,
        'decision_frames': 6, 'turn_rate': 1.0,
        'args': {k: (list(v) if isinstance(v, tuple) else v) for k, v in vars(args).items()}}, indent=1))


def unit_interval(value):
    number = float(value)
    if not 0 <= number <= 1:
        raise argparse.ArgumentTypeError('must be between 0 and 1')
    return number


def build_parser():
    class TaskParser(argparse.ArgumentParser):
        def parse_args(self, args=None, namespace=None):
            values = list(sys.argv[1:] if args is None else args)
            result = super().parse_args(values, namespace)
            if result.task == 'flocker_team' and not any(v.split('=')[0]=='--bootstrap-episodes' for v in values):
                result.bootstrap_episodes = 0
            if result.task == 'inchworm':
                from learning.tasks import task
                explicit = {v.split('=')[0] for v in values if v.startswith('--')}
                for dest, default in task('inchworm').fresh_defaults.items():
                    if '--'+dest.replace('_','-') not in explicit:
                        setattr(result,dest,default)
            return result
    parser = TaskParser(description=__doc__)
    from learning.tasks import TASKS
    parser.add_argument('--task', choices=tuple(TASKS), default='hunter', help='creature training task')
    parser.add_argument('--opponent', type=str, default=str(DEFAULT_START), help='frozen player weights for team training')
    parser.add_argument('--budget-seconds', type=float, default=7200, help='total team training and evaluation wall time; 0 means no time limit')
    parser.add_argument('--bootstrap-episodes', type=int, default=8, help='scripted imitation encounters for new team runs')
    parser.add_argument('--bootstrap-epochs', type=int, default=0,
                        help='passes over teacher demonstrations; 0 uses the task default')
    parser.add_argument('--seed', type=int, default=0, help='team model and training seed')
    from learning.reward_table import tables
    for table in tables().values():          # the tasks that declare their rewards in a reward table
        table.add_arguments(parser)
    parser.add_argument('--start', default=str(DEFAULT_START))
    parser.add_argument('--out', default=str(DEFAULT_OUT))
    parser.add_argument('--iterations', type=int, default=100, help='training iterations; 0 means unlimited for creature teams')
    parser.add_argument('--episodes', type=int, default=96)
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--gamma', type=unit_interval, default=0.99,
                        help='future reward discount, 0–1; higher values give distant rewards more weight')
    parser.add_argument('--gae-lambda', type=unit_interval, default=0.95,
                        help='GAE credit assignment, 0–1; higher values carry credit further back with more noise')
    parser.add_argument('--wall-scale', type=float, default=1.0,
                        help='multiplies the wall-collision penalty in the training reward (the score is unchanged)')
    parser.add_argument('--fixed-start', action='store_true',
                        help='start the hunter beside the player on every episode instead of at a random open cell')
    parser.add_argument('--still-speed', type=float, default=0.15,
                        help='below this share of top speed counts as near-motionless')
    parser.add_argument('--turn-change-penalty', type=float, default=0.0,
                        help='points lost each time a turn starts or reverses')
    parser.add_argument('--health', type=int, default=0,
                        help='hits the hunter survives before it is destroyed and the episode ends (0: indestructible)')
    parser.add_argument('--death-penalty', type=float, default=0.0,
                        help='points lost when the hunter is destroyed (needs --health)')
    parser.add_argument('--reversal-penalty', type=float, default=0.0,
                        help='extra points lost each time the turn flips straight from left to right or back')
    parser.add_argument('--horizon', type=float, default=DEFAULT_HORIZON,
                        help='seconds of game time per training episode')
    parser.add_argument('--still-penalty', type=float, default=0.0,
                        help='points lost per second spent near-motionless in the training reward')
    parser.add_argument('--speed-penalty', type=float, default=0.0,
                        help='points lost per second at standstill, scaling down linearly to zero at top speed')
    parser.add_argument('--powerup-reward', type=float, default=0.0,
                        help='points gained per powerup crystal the hunter collects in the training reward')
    parser.add_argument('--upgrade-bonus', type=float, default=0.0,
                        help='points gained per second the hunter\'s guns are upgraded in the training reward')
    parser.add_argument('--decoy-player', action='store_true',
                        help='leave the parked player as a target for enemies (by default they chase only the hunter)')
    parser.add_argument('--after-clear', type=float, default=0.0,
                        help='seconds an episode keeps going after the last enemy dies (less if no crystal is left)')
    parser.add_argument('--hit-scale', type=float, default=1.0,
                        help='multiplies the enemy-hit penalty in the training reward (the score is unchanged)')
    parser.add_argument('--rules-bounty', type=float, default=50.0,
                        help='points gained for destroying the Rules hunter in the training reward')
    parser.add_argument('--compass-reward', type=float, default=0.0,
                        help='points per maze width the enemy compass route shortens while no enemy is in sight')
    parser.add_argument('--boredom-penalty', type=float, default=0.0,
                        help='points lost per second with no enemy in sight in a recently visited maze cell')
    parser.add_argument('--boredom-window', type=float, default=10.0,
                        help='seconds before a visited maze cell counts as new again')
    parser.add_argument('--shot-price', type=float, default=0.0,
                        help='points lost per shot not within 20 degrees of an enemy in sight')
    parser.add_argument('--panic-score', type=float, default=None,
                        help='panic once the last 5 s score below this (e.g. -10); off when not given')
    parser.add_argument('--panic-temperature', type=float, default=3.0,
                        help='the temperature choices are sampled at while panicking')
    parser.add_argument('--hit-reward', type=float, default=0.0,
                        help='points gained per hunter shot that reaches an enemy')
    parser.add_argument('--turning-penalty', type=float, default=0.0,
                        help='points lost per second spent turning (circling costs; a quick correction is cheap)')
    parser.add_argument('--levels', type=parse_levels, default=None,
                        help='levels to train on, e.g. 1-20 or 4,7,13 (default: the tune set)')
    parser.add_argument('--actor-scale', type=float, default=1.,
                        help='one-time wallcrawler output-head damping when starting a new run (0 < scale <= 1)')
    parser.add_argument('--resume', action='store_true',
                        help='carry on the run in --out from its resume state')
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not 0<args.actor_scale<=1 or (args.task in PPO_TASKS and args.actor_scale!=1):
        raise ValueError('actor scale requires a wallcrawler run and 0 < scale <= 1')
    if args.task not in PPO_TASKS:
        from learning.run import train
        raw = sys.argv[1:] if argv is None else argv
        if not any(v.split('=')[0] == '--horizon' for v in raw):
            from learning.tasks import task
            args.horizon = task(args.task).fresh_defaults.get('horizon', 60.0)
        if args.workers < 1 or args.episodes < 1 or args.horizon <= 0 or args.budget_seconds < 0 or args.iterations < 0:
            raise ValueError('workers, episodes and horizon must be positive; budget and iterations must be nonnegative')
        result = train(args)
        print('paused' if result['paused'] else 'done', '| checkpoint:', result['last'])
        return
    if args.task == 'campaign_hunter':
        from learning.tasks import task
        raw = sys.argv[1:] if argv is None else argv
        if not any(v.split('=')[0] == '--horizon' for v in raw):
            args.horizon = task(args.task).fresh_defaults['horizon']
        args.levels = args.levels or task(args.task).adapter().default_levels()
    if not args.resume:
        write_run_config(args)
    from experiments import run
    run._use_dummy_drivers()
    result = train_rl(args.start, args.out, args.iterations, args.episodes, args.workers, lr=args.lr,
                      shaping=shaping_from(args),
                      levels=args.levels, resume=args.resume, gamma=args.gamma, gae_lambda=args.gae_lambda,
                      task=args.task)
    if result['paused']:
        print('paused | at iteration', result['history'][-1]['iteration'] if result['history'] else '-')
    else:
        print('done | last checkpoint:', result['last'])


if __name__ == '__main__':
    main(sys.argv[1:])
