"""Paired, fixed-policy acceptance check for persistent-target stall recovery.

    venv/bin/python -m tools.campaign_target_eval --out demo-out/target-recovery

Saves per-decision traces and the predeclared acceptance results. Never changes defaults or weights.
"""
import argparse
import dataclasses
import hashlib
import json
import math
import multiprocessing
from pathlib import Path

LEVELS = (3, 8, 11, 14, 17, 23, 28, 29, 31, 32, 34)


def fly(job):
    from experiments.run import _use_dummy_drivers
    _use_dummy_drivers()
    import config
    from experiments.campaign_rollout import fly_campaign
    from experiments.student import StudentSpec
    from hunter.campaign_pilot import CampaignPilot
    from hunter.neural.policy import NeuralPolicy
    from hunter.campaign_senses import CAMPAIGN_FEATURE_NAMES
    level, seed, recovery, weights, directory = job
    # Seed before constructing Game, not only after level setup in fly_campaign.
    # Each job runs in a fresh worker so a previous episode cannot leak state.
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    policy = NeuralPolicy.load(weights, CAMPAIGN_FEATURE_NAMES)
    pilot = CampaignPilot(policy, config.SETTINGS.game.neuralDecisionFrames, target_recovery=recovery)
    senses = pilot._senses
    rows, identities = [], {}
    previous, progress_cost, progress_at, hits = None, math.inf, 0.0, None
    stalled, switches, previous_time = 0.0, 0, 0.0

    def identity(enemy):
        if enemy is None:
            return None
        if enemy not in identities:
            identities[enemy] = len(identities)
        return identities[enemy]

    def trace(seconds, game):
        nonlocal previous, progress_cost, progress_at, hits, stalled, switches, previous_time
        # Observe the same decision cadence for both treatments; ignore terminal animation frames.
        if (pilot.controller.frames - 1) % pilot.controller.decision_frames:
            return
        if game.game_frozen or game.exit_explosion_active or game.state != config.STATE_PLAYING:
            return
        target = pilot.target_enemy
        alive = game.roster.alive()
        for enemy in alive:
            identity(enemy)
        cost = senses._route_cost((game.ship.x, game.ship.y), game.maze, target) if target in alive else math.inf
        hp = senses._remaining_hits(target) if target in alive else None
        damage = max(0.0, hits - hp) if target is previous and hits is not None and hp is not None else 0.0
        change = target is not previous
        if change:
            switches += int(previous is not None and target is not None)
        if change or damage or progress_cost - cost >= .25 * max(game.maze.cell_size_x, game.maze.cell_size_y):
            progress_cost, progress_at = cost, seconds
        idle = bool(alive and target in alive and seconds - progress_at >= 4.0)
        if idle:
            stalled += seconds - previous_time
        rows.append(dict(seconds=seconds, target=identity(target), kind=target.kind.key if target else None,
                         switch_reason=(senses.target_switch_reason or 'legacy') if change else None,
                         route_cost=cost if math.isfinite(cost) else None, damage=damage, hits=hp,
                         stalled=idle, controls=pilot.controller.last_controls,
                         alive=[identity(e) for e in alive], kills=game.scoring.enemies_destroyed,
                         wall_crashes=game.scoring.wall_collisions))
        previous, hits, previous_time = target, hp, seconds

    flown = fly_campaign(level, seed, StudentSpec(weights=weights, greedy=True), horizon=180,
                         greedy=True, pilot=pilot, trace=trace, sticky_targets=True, target_recovery=recovery)
    name = f'level{level}-seed{seed}-recovery{int(recovery)}'
    Path(directory, name + '.json').write_text(json.dumps(rows, separators=(',', ':')))
    return dict(dataclasses.asdict(flown['result']), target_recovery=recovery,
                stalled_seconds=stalled, target_switches=switches, trace=name + '.json')


def acceptance(rows):
    def groups(level2):
        return [[r for r in rows if (r['level'] == 2) == level2 and r['target_recovery'] == mode]
                for mode in (False, True)]

    def mean(rows, key):
        return sum(r[key] for r in rows) / len(rows)

    base2, new2 = groups(True)
    base, new = groups(False)
    before_stall, after_stall = sum(r['stalled_seconds'] for r in base2), sum(r['stalled_seconds'] for r in new2)
    checks = {'level2_stall_reduction': after_stall <= before_stall * .5,
              'exits': sum(r['end'] == 'exited' for r in new) >= sum(r['end'] == 'exited' for r in base),
              'cleared_exits': sum(r['end'] == 'exited' and r['cleared'] for r in new) >=
                               sum(r['end'] == 'exited' and r['cleared'] for r in base),
              'kills': mean(new, 'own_kills') >= .95 * mean(base, 'own_kills'),
              'wall_crashes': mean(new, 'wall_crashes') <= mean(base, 'wall_crashes') +
                              max(.1 * mean(base, 'wall_crashes'), .5)}
    return dict(passed=all(checks.values()), checks=checks,
                level2_stalled_seconds=[before_stall, after_stall],
                broad={key: [mean(base, key), mean(new, key)] for key in
                       ('own_kills', 'wall_crashes', 'stalled_seconds', 'target_switches')},
                broad_exits=[sum(r['end'] == 'exited' for r in group) for group in (base, new)],
                broad_cleared_exits=[sum(r['end'] == 'exited' and r['cleared'] for r in group) for group in (base, new)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--policy', default='assets/neural/campaign_hunter.npz')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(Path(args.policy).read_bytes()).hexdigest()
    episodes = [(2, s) for s in (102, 300, 301, 302)] + [(level, s) for level in LEVELS for s in (400, 401, 402)]
    jobs = [(level, seed, mode, args.policy, str(out)) for level, seed in episodes for mode in (False, True)]
    rows = []
    with multiprocessing.get_context('spawn').Pool(args.workers, maxtasksperchild=1) as pool:
        for row in pool.imap_unordered(fly, jobs):
            rows.append(row)
            print(f"{len(rows)}/{len(jobs)} level {row['level']} seed {row['seed']} recovery={row['target_recovery']} "
                  f"{row['end']} kills={row['own_kills']} stalled={row['stalled_seconds']:.1f}s", flush=True)
    assert digest == hashlib.sha256(Path(args.policy).read_bytes()).hexdigest(), 'policy changed during comparison'
    rows.sort(key=lambda r: (r['level'], r['seed'], r['target_recovery']))
    report = dict(policy=args.policy, sha256=digest, fresh_process_per_episode=True,
                  acceptance=acceptance(rows), rows=rows)
    (out / 'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report['acceptance'], indent=2))


if __name__ == '__main__':
    main()
