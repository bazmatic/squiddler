"""Measure a Campaign Hunter policy: python -m tools.campaign_eval POLICY [--levels 1-40] [--seeds 3] [--workers N]

POLICY is a campaign-layout .npz, or a hunter-layout one (such as the shipped assets/neural/hunter.npz, or the word
`hunter`) that is carried over to the campaign senses in a temporary file. Each level is flown greedily with seeds
300.., as the trainer does. `--json PATH` keeps every episode's row, to compare runs.

The headline: demo-exit % (exits Demo would have shown: no spell longer than Demo's no-kill skip with enemies left
and none killed), cleared-exit % (exits with no enemy left) and scored-out %.
"""
import argparse
import dataclasses
import json
import os
import tempfile
import time
from pathlib import Path

FIRST_SEED = 300
HORIZON = 180.0


def parse_levels(text):
    """'1-40', '3', or '1,5,9-12': levels from 1, each range rising."""
    levels = []
    for part in text.split(','):
        first, _, last = part.partition('-')
        try:
            low, high = int(first), int(last or first)
        except ValueError:
            raise ValueError(f'bad levels {text!r}: expected e.g. 1-40, 3 or 1,5,9-12') from None
        if low < 1 or high < low:
            raise ValueError(f'bad levels {part!r} in {text!r}: levels start at 1 and a range must rise')
        levels.extend(range(low, high + 1))
    return levels


def campaign_weights(policy, directory):
    """A path to `policy` in the campaign layout: itself, or its migration (written in `directory`)."""
    from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES
    from hunter.neural.loader import DEFAULT_PATH
    from hunter.neural.policy import InvalidPolicy, NeuralPolicy
    from tools.migrate_policy import migrate
    path = str(DEFAULT_PATH if policy == 'hunter' else policy)
    try:
        NeuralPolicy.load(path, CAMPAIGN_FEATURE_NAMES)
        return path
    except InvalidPolicy:
        converted = os.path.join(directory, 'campaign_start.npz')
        migrate(path, converted, CAMPAIGN_FEATURE_NAMES)
        return converted


def _fly(job):
    from experiments.campaign_rollout import fly_campaign
    from experiments.student import StudentSpec
    level, seed, weights, horizon, pursuit_lesson, sampled, sticky_targets, target_recovery = job
    started = time.perf_counter()
    flown = fly_campaign(level, seed, StudentSpec(weights=weights, greedy=not sampled, sample_seed=seed * 101 + level),
                         horizon=horizon, greedy=not sampled, pursuit_lesson=pursuit_lesson,
                         sticky_targets=sticky_targets, target_recovery=target_recovery)
    return dict(dataclasses.asdict(flown['result']), **flown['stats'], reward_totals=flown['reward_totals'],
                sampled=sampled, wall=time.perf_counter() - started)


def evaluate(policy, levels, seeds=3, horizon=HORIZON, workers=1, first_seed=FIRST_SEED,
             pursuit_lesson=False, sampled=False, sticky_targets=True, target_recovery=False):
    """(rows, summary): one row per level and seed, in that order, and the overall summary."""
    from experiments import run
    run._use_dummy_drivers()       # headless, in this process as in the pool's
    with tempfile.TemporaryDirectory() as directory:
        weights = campaign_weights(policy, directory)
        jobs = [(level, first_seed + s, weights, horizon, pursuit_lesson, sampled, sticky_targets, target_recovery)
                for level in levels for s in range(seeds)]
        if workers > 1:
            import multiprocessing
            with multiprocessing.get_context('spawn').Pool(workers, initializer=run._use_dummy_drivers) as pool:
                rows = pool.map(_fly, jobs, chunksize=1)
        else:
            rows = [_fly(job) for job in jobs]
    return rows, summarize(rows)


def demo_exit(row):
    """Whether Demo would have shown this exit: its no-kill skip (noKillSeconds, 0 for off) never fired first."""
    from config import SETTINGS
    limit = SETTINGS.autoplay.noKillSeconds
    return row['end'] == 'exited' and (not limit or row['max_no_kill_gap'] <= limit)


def summarize(rows):
    """Shares of episodes by end, mean own kills, mean own kills per enemy at the start (at most 1 an episode), mean
    seconds. A cleared exit left by the exit with no enemy left (`cleared` alone is also true on scoring out); a Demo
    exit is one Demo's no-kill skip would have let happen."""
    n = len(rows) or 1

    def mean(values):
        return sum(values) / n

    summary = dict(episodes=len(rows),
                demo_exit=mean([demo_exit(r) for r in rows]),
                exit=mean([r['end'] == 'exited' for r in rows]),
                cleared_exit=mean([r['end'] == 'exited' and r['cleared'] for r in rows]),
                scored_out=mean([r['end'] == 'scored_out' for r in rows]),
                horizon=mean([r['end'] == 'horizon' for r in rows]),
                own_kills=mean([r['own_kills'] for r in rows]),
                kills_fraction=mean([min(1.0, r['own_kills'] / r['enemies_at_start']) if r['enemies_at_start']
                                     else 1.0 for r in rows]),
                elapsed=mean([r['elapsed'] for r in rows]))
    searches = [r for r in rows if r.get('started_unseen')]
    if searches:
        found = [r for r in searches if r['first_sighting_seconds'] is not None]
        summary.update(search_episodes=len(searches), search_found=len(found) / len(searches),
                       search_seconds=sum(r['first_sighting_seconds'] if r['first_sighting_seconds'] is not None
                                          else r.get('search_horizon_seconds', r['elapsed'])
                                          for r in searches) / len(searches))
    if rows and 'wall_crashes' in rows[0]:
        summary['wall_crashes'] = mean([r['wall_crashes'] for r in rows])
    if rows and 'wall_impulse' in rows[0]:
        summary['wall_impulse'] = mean([r['wall_impulse'] for r in rows])
    if rows and 'unseen_seconds' in rows[0]:
        summary['unseen_seconds'] = mean([r['unseen_seconds'] for r in rows])
    if rows and rows[0].get('pursuit_lesson'):
        summary['lesson_contact'] = mean([r['end'] == 'contact' for r in rows])
        summary['visible_starts'] = sum(r['end'] == 'visible_start' for r in rows)
    return summary


def print_report(rows, summary):
    if 'lesson_contact' in summary:
        print(f'{summary["episodes"]} pursuit lessons: contact {summary["lesson_contact"]:.0%}, '
              f'mean first sighting {summary.get("search_seconds", 0):.1f}s '
              f'(misses count the full time limit); visible starts {summary["visible_starts"]}')
        for level in sorted({r['level'] for r in rows}):
            part = summarize([r for r in rows if r['level'] == level])
            print(f'  level {level}: contact {part["lesson_contact"]:.0%}, '
                  f'first sighting {part.get("search_seconds", 0):.1f}s')
        return
    print('Per level, the end columns count episodes out of eps.')
    print(f'{"level":>5} {"eps":>3} {"demo-exit":>9} {"exit":>5} {"c-exit":>6} {"scored":>6} {"horiz":>5} '
          f'{"kills":>6} {"kills/start":>11} {"secs":>6}')
    for level in sorted({r['level'] for r in rows}):
        mine = [r for r in rows if r['level'] == level]
        part = summarize(mine)
        n = len(mine)
        counts = [sum(demo_exit(r) for r in mine), sum(r['end'] == 'exited' for r in mine),
                  sum(r['end'] == 'exited' and r['cleared'] for r in mine),
                  sum(r['end'] == 'scored_out' for r in mine), sum(r['end'] == 'horizon' for r in mine)]
        cells = [f'{c}/{n}' for c in counts]
        print(f'{level:>5} {n:>3} {cells[0]:>9} {cells[1]:>5} {cells[2]:>6} {cells[3]:>6} {cells[4]:>5} '
              f'{part["own_kills"]:>6.1f} {part["kills_fraction"]:>11.2f} {part["elapsed"]:>6.1f}')
    print(f'\n{summary["episodes"]} episodes: demo-exit {summary["demo_exit"]:.0%}, '
          f'cleared-exit {summary["cleared_exit"]:.0%}, scored-out {summary["scored_out"]:.0%} '
          f'(any exit {summary["exit"]:.0%}, horizon {summary["horizon"]:.0%}); '
          f'mean own kills {summary["own_kills"]:.2f}, mean kills/start {summary["kills_fraction"]:.2f}, '
          f'mean elapsed {summary["elapsed"]:.1f}s')
    if 'search_found' in summary:
        print(f'Hidden-start search: contact {summary["search_found"]:.0%}, '
              f'mean first sighting {summary["search_seconds"]:.1f}s (misses count the full time limit); '
              f'wall strikes {summary.get("wall_crashes", 0):.1f}/episode')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('policy')
    parser.add_argument('--levels', default='1-40')
    parser.add_argument('--seeds', type=int, default=3)
    parser.add_argument('--first-seed', type=int, default=FIRST_SEED,
                        help='first evaluation seed; use 400 or higher to exclude training seeds 300-399')
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--horizon', type=float, default=HORIZON)
    parser.add_argument('--json')
    parser.add_argument('--pursuit-lesson', action='store_true', help='end each episode at first enemy sight')
    parser.add_argument('--sampled', action='store_true', help='sample controls with fixed per-episode seeds')
    parser.add_argument('--nearest-targets', action='store_true', help='use historical nearest-enemy targeting')
    parser.add_argument('--target-recovery', action='store_true', help='recover stalled persistent chases')
    args = parser.parse_args(argv)
    if args.target_recovery and args.nearest_targets:
        parser.error('--target-recovery requires persistent targeting')
    try:
        levels = parse_levels(args.levels)
    except ValueError as exc:
        parser.error(str(exc))
    started = time.perf_counter()
    rows, summary = evaluate(args.policy, levels, args.seeds, args.horizon, args.workers, args.first_seed,
                             args.pursuit_lesson, args.sampled, not args.nearest_targets, args.target_recovery)
    print_report(rows, summary)
    print(f'{time.perf_counter() - started:.0f}s on {args.workers} workers')
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(dict(policy=args.policy, summary=summary, rows=rows), indent=1))


if __name__ == '__main__':
    main()
